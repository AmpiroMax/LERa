# Copyright (c) Aishwarya Kamath & Nicolas Carion.
# Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
NB: The original file was modified by commenting or deleting redundant code and
by adapting to VQA task in ALFRED.

MDETR model and criterion classes.
"""
from typing import Dict, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

# import util.dist as dist
# from util import box_ops
# from util.metrics import accuracy
from fiqa.task_handlers.vqa_models.util.misc import NestedTensor, interpolate

from fiqa.task_handlers.vqa_models.backbone import build_backbone
# from .matcher import build_matcher
# from .postprocessors import build_postprocessors
# from .segmentation import DETRsegm, dice_loss, sigmoid_focal_loss
from fiqa.task_handlers.vqa_models.transformer import build_transformer


class MDETR(nn.Module):
    """This is the MDETR module that performs modulated object detection."""

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        qa_dataset,
        aux_loss=False,
        contrastive_hdim=64,
        contrastive_loss=False,
        # contrastive_align_loss=False,
        split_qa_heads=True,
        # predict_final=False,
    ):
        """Initializes the model.

        Parameters
        ----------
        backbone
            Torch module of the backbone to be used. See backbone.py.
        transformer
            Torch module of the transformer architecture. See transformer.py.
        num_classes
            Number of object classes.
        num_queries
            Number of object queries, ie detection slot. This is
            the maximal number of objects MDETR can detect in a single image.
            For COCO, we recommend 100 queries.
        qa_dataset
            Train a QA head for the target dataset (TODO: here only available what?)
        aux_loss
            True if auxiliary decoding losses (loss at each decoder layer)
            are to be used.
        contrastive_hdim
            Dimension used for projecting the embeddings
            before computing contrastive loss.
        contrastive_loss
            If true, perform image-text contrastive learning.
        split_qa_heads
            If true, use several head for each question type.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # self.isfinal_embed = nn.Linear(hidden_dim, 1) if predict_final
        # else None
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        nb_heads = 8
        self.qa_embed = nn.Embedding(
            nb_heads if split_qa_heads else 1, hidden_dim
        )

        self.input_proj = nn.Conv2d(
            backbone.num_channels, hidden_dim, kernel_size=1
        )
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.contrastive_loss = contrastive_loss
        if contrastive_loss:
            self.contrastive_projection_image = nn.Linear(
                hidden_dim, contrastive_hdim, bias=False
            )
            self.contrastive_projection_text = nn.Linear(
                self.transformer.text_encoder.config.hidden_size,
                contrastive_hdim, bias=False
            )
        # self.contrastive_align_loss = contrastive_align_loss
        # if contrastive_align_loss:
        #     self.contrastive_align_projection_image = nn.Linear(
        #         hidden_dim, contrastive_hdim
        #     )
        #     self.contrastive_align_projection_text = nn.Linear(
        #         hidden_dim, contrastive_hdim
        #     )

        self.qa_dataset = qa_dataset
        self.split_qa_heads = split_qa_heads
        if split_qa_heads:
            # TODO: make this more general
            if qa_dataset == "alfred":
                self.answer_type_head = nn.Linear(hidden_dim, 7)
                self.answer_existence_head = nn.Linear(hidden_dim, 1)
                self.answer_pickupable_head = nn.Linear(hidden_dim, 1)
                self.answer_picked_up_head = nn.Linear(hidden_dim, 1)
                self.answer_receptacle_head = nn.Linear(hidden_dim, 1)
                self.answer_opened_head = nn.Linear(hidden_dim, 1)
                self.answer_toggled_on_head = nn.Linear(hidden_dim, 1)
                self.answer_sliced_head = nn.Linear(hidden_dim, 1)
            else:
                assert False, f"Invalid qa dataset {qa_dataset}"
        else:
            # TODO: make this more general
            if qa_dataset == "alfred":
                self.answer_head = nn.Linear(hidden_dim, 1)
            else:
                assert False, f"Invalid qa dataset {qa_dataset}"

    def forward(
        self, samples: NestedTensor, captions, encode_and_save=True,
        memory_cache=None
    ):
        """The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W],
            containing 1 on padded pixels

        Parameters
        ----------
        samples
            A NestedTensor (see util/misc.py) containing batched images and
            their masks.
        captions
            Encoded or raw image captions.
        encode_and_save
            A flag indicating the first or the second forward() stage.
        memory_cache
            A dictionary for storing info after the first stage of forward().

        Returns
        -------
        dict
            A dictionary with the following elements:
                - "pred_logits": the classification logits (including no-object)
                for all queries.
                Shape = [batch_size x num_queries x (num_classes + 1)]
                - "pred_boxes": The normalized boxes coordinates
                for all queries, represented as
                (center_x, center_y, height, width). These values are
                normalized in [0, 1], relative to the size of each individual
                image (disregarding possible padding). See PostProcess for
                information on how to retrieve the unnormalized bounding box.
                - "aux_outputs": Optional, only returned when auxiliary losses
                are activated. It is a list of dictionaries containing
                the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        if encode_and_save:
            assert memory_cache is None
            features, pos = self.backbone(samples)
            src, mask = features[-1].decompose()

            query_embed = self.query_embed.weight
            query_embed = torch.cat([query_embed, self.qa_embed.weight], 0)

            memory_cache = self.transformer(
                self.input_proj(src),
                mask,
                query_embed,
                pos[-1],
                captions,
                encode_and_save=True,
                text_memory=None,
                img_memory=None,
                text_attention_mask=None,
            )

            if self.contrastive_loss:
                memory_cache["text_pooled_op"] = \
                    self.contrastive_projection_text(
                        memory_cache["text_pooled_op"]
                    )
                memory_cache["img_pooled_op"] = \
                    self.contrastive_projection_image(
                        memory_cache["img_pooled_op"]
                    )

            return memory_cache

        else:
            assert memory_cache is not None
            # hs.shape = (num_layers=6, b, num_queries + nb_heads, 256)
            hs = self.transformer(
                mask=memory_cache["mask"],
                query_embed=memory_cache["query_embed"],
                pos_embed=memory_cache["pos_embed"],
                encode_and_save=False,
                text_memory=memory_cache["text_memory_resized"],
                img_memory=memory_cache["img_memory"],
                text_attention_mask=memory_cache["text_attention_mask"],
            )
            out = {}
            if self.split_qa_heads:
                if self.qa_dataset == "alfred":
                    answer_embeds = hs[0, :, -8:]
                    hs = hs[:, :, :-8]
                    out["pred_answer_type"] = self.answer_type_head(
                        answer_embeds[:, 0]
                    )
                    out["pred_answer_existence"] = self.answer_existence_head(
                        answer_embeds[:, 1]
                    ).squeeze(-1)
                    out["pred_answer_pickupable"] = self.answer_pickupable_head(
                        answer_embeds[:, 2]
                    ).squeeze(-1)
                    out["pred_answer_picked_up"] = self.answer_picked_up_head(
                        answer_embeds[:, 3]
                    ).squeeze(-1)
                    out["pred_answer_receptacle"] = self.answer_receptacle_head(
                        answer_embeds[:, 4]
                    ).squeeze(-1)
                    out["pred_answer_opened"] = self.answer_opened_head(
                        answer_embeds[:, 5]
                    ).squeeze(-1)
                    out["pred_answer_toggled_on"] = self.answer_toggled_on_head(
                        answer_embeds[:, 6]
                    ).squeeze(-1)
                    out["pred_answer_sliced"] = self.answer_sliced_head(
                        answer_embeds[:, 7]
                    ).squeeze(-1)
                else:
                    assert False, f"Invalid qa dataset {self.qa_dataset}"
            else:
                answer_embeds = hs[0, :, -1]
                hs = hs[:, :, :-1]
                if self.qa_dataset == "alfred":
                    out["pred_answer"] = self.answer_head(
                        answer_embeds
                    ).squeeze(-1)
                else:
                    out["pred_answer"] = self.answer_head(answer_embeds)

            # class_embed.weight.shape = (d_model=256, num_classes+1=255+1)
            # output_class.shape =
            # = (num_layers=6, b, num_queries=100, num_classes+1=256)
            outputs_class = self.class_embed(hs)
            # bbox_embed(hs).shape = (num_layers=6, b, num_queries=100, 4)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            out.update(
                {
                    "pred_logits": outputs_class[-1],
                    "pred_boxes": outputs_coord[-1],
                }
            )
            # outputs_isfinal = None
            # if self.isfinal_embed is not None:
            #     outputs_isfinal = self.isfinal_embed(hs)
            #     out["pred_isfinal"] = outputs_isfinal[-1]
            # proj_queries, proj_tokens = None, None
            # if self.contrastive_align_loss:
            #     proj_queries = F.normalize(
            #         self.contrastive_align_projection_image(hs), p=2, dim=-1)
            #     proj_tokens = F.normalize(
            #         self.contrastive_align_projection_text(
            #             memory_cache["text_memory"]).transpose(0, 1), p=2,
            #         dim=-1
            #     )
            #     out.update(
            #         {
            #             "proj_queries": proj_queries[-1],
            #             "proj_tokens": proj_tokens,
            #             "tokenized": memory_cache["tokenized"],
            #         }
            #     )
            if self.aux_loss:
                # if self.contrastive_align_loss:
                #     assert proj_tokens is not None and \
                #            proj_queries is not None
                #     out["aux_outputs"] = [
                #         {
                #             "pred_logits": a,
                #             "pred_boxes": b,
                #             "proj_queries": c,
                #             "proj_tokens": proj_tokens,
                #             "tokenized": memory_cache["tokenized"],
                #         }
                #         for a, b, c in
                #         zip(outputs_class[:-1], outputs_coord[:-1],
                #             proj_queries[:-1])
                #     ]
                # else:
                out["aux_outputs"] = [
                    {
                        "pred_logits": a,
                        "pred_boxes": b,
                    }
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]
                # if outputs_isfinal is not None:
                #     assert \
                #         len(outputs_isfinal[:-1]) == len(out["aux_outputs"])
                #     for i in range(len(outputs_isfinal[:-1])):
                #         out["aux_outputs"][i]["pred_isfinal"] = \
                #             outputs_isfinal[i]
            return out


class ContrastiveCriterion(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, pooled_text, pooled_image):

        normalized_text_emb = F.normalize(pooled_text, p=2, dim=1)
        normalized_img_emb = F.normalize(pooled_image, p=2, dim=1)

        logits = torch.mm(
            normalized_img_emb, normalized_text_emb.t()
        ) / self.temperature
        labels = torch.arange(logits.size(0)).to(pooled_image.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        loss = (loss_i + loss_t) / 2.0
        return loss


class QACriterionAlfred(nn.Module):
    # Taken from ALFRED-QAD dataloader
    type2id = {
        "existence": 0, "pickupable": 1, "picked_up": 2, "receptacle": 3,
        "opened": 4, "toggled_on": 5, "sliced": 6
    }
    id2type = list(type2id.keys())

    def __init__(self, split_qa_heads):
        super().__init__()
        self.split_qa_heads = split_qa_heads
        self.gl_avg = {}
        self.reset_stats()

    def reset_stats(self):
        """Resets accuracy statistic. Should be called before evaluation."""
        if self.split_qa_heads:
            self.gl_avg = {key: [0, 0] for key in self.type2id.keys()}
            self.gl_avg.update(ans_type=[0, 0])
        else:
            self.gl_avg = {
                "total": [0, 0]
            }

    def calc_acc_from_stats(self):
        """Calculates accuracies for every question type
        (if split_qa_heads=True) or total accuracy (if split_qa_heads=False).
        """
        return {key: val[0] / val[1] for key, val in self.gl_avg.items()}

    def forward(self, output, answers):
        loss = {}
        if not self.split_qa_heads:
            loss["loss_answer_total"] = F.binary_cross_entropy_with_logits(
                output["pred_answer"], answers["answer"].float(),
                reduction="mean"
            )
            attr_total = \
                (output["pred_answer"].sigmoid() > 0.5) == answers["answer"]
            loss["accuracy_answer_total"] = attr_total.float().mean()

            if not self.training:
                self.gl_avg["total"][0] += attr_total.sum().item()
                self.gl_avg["total"][1] += attr_total.numel()

            return loss

        device = output["pred_answer_type"].device

        loss["loss_answer_type"] = F.cross_entropy(
            output["pred_answer_type"], answers["answer_type"]
        )
        type_acc = \
            output["pred_answer_type"].argmax(-1) == answers["answer_type"]
        loss["accuracy_answer_type"] = type_acc.float().mean()

        correct_ans = torch.zeros(type_acc.shape, dtype=torch.bool)
        for type_name, type_id in self.type2id.items():
            is_type, acc_type = QACriterionAlfred._calc_loss_and_acc(
                device, output, answers, type_id, type_name, loss
            )
            correct_ans += is_type * acc_type

            # Since the `loss["accuracy_answer_total"]` is not precise
            # (1 / batch_size limits it), let's accumulate results
            # from every batch and calculate real global average
            if not self.training:
                self.gl_avg[type_name][0] += acc_type[is_type].sum().item()
                self.gl_avg[type_name][1] += is_type.sum().item()

        if not self.training:
            self.gl_avg["ans_type"][0] += type_acc.sum().item()
            self.gl_avg["ans_type"][1] += type_acc.numel()

        loss["accuracy_answer_total"] = \
            (type_acc * correct_ans).sum() / type_acc.numel()
        return loss

    @staticmethod
    def _calc_loss_and_acc(device, output, answers, type_id, type_name, loss):
        is_type = answers["answer_type"] == type_id
        norm_type = is_type.sum() if is_type.any() \
            else torch.as_tensor(1.0, device=device)
        loss[f"loss_answer_{type_name}"] = (
                F.binary_cross_entropy_with_logits(
                    output[f"pred_answer_{type_name}"],
                    answers[f"answer_{type_name}"].float(),
                    reduction="none"
                ).masked_fill(~is_type, 0).sum()
                / norm_type
        )
        acc_type = (
            (output[f"pred_answer_{type_name}"].sigmoid() > 0.5) ==
            answers[f"answer_{type_name}"]
        )
        loss[f"accuracy_answer_{type_name}"] = \
            acc_type[is_type].sum() / norm_type
        return is_type, acc_type


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 255

    # assert not args.masks or args.mask_model != "none"

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = MDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        contrastive_hdim=args.contrastive_loss_hdim,
        contrastive_loss=args.contrastive_loss,
        qa_dataset=args.qa_dataset,
        split_qa_heads=args.split_qa_heads,
    )

    # if args.contrastive_loss:
    #     contrastive_criterion = ContrastiveCriterion(
    #         temperature=args.temperature_NCE)
    #     contrastive_criterion.to(device)
    # else:
    #     contrastive_criterion = None

    if args.qa_dataset == "alfred":
        qa_criterion = QACriterionAlfred(args.split_qa_heads)
    else:
        assert False, f"Invalid qa dataset {args.qa_dataset}"

    return model, qa_criterion
