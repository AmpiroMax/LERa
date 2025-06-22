import argparse
import torch
import torchvision.transforms as T
from typing import List, Tuple


class DummyVQAModel(torch.nn.Module):
    """A dummy model for debugging purposes.s"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, rgb: torch.Tensor, questions: List[str]) -> bool:
        return True


def build_dummy_vqa_model(
    args: argparse.Namespace
) -> Tuple[torch.nn.Module, T.Compose]:
    return DummyVQAModel(), T.Compose([T.ToTensor()])


def prepare_model(
    args: argparse.Namespace
) -> Tuple[torch.nn.Module, T.Compose]:
    if args.vqa_model == 'none':
        return build_dummy_vqa_model(args)

    elif args.vqa_model == 'mdetr':
        from fiqa.task_handlers.vqa_models.mdetr import build as build_mdetr

        model, _ = build_mdetr(args)
        checkpoint = torch.load(
            'fiqa/checkpoints/mdetr_best_checkpoint.pth', map_location='cpu'
        )
        model.load_state_dict(checkpoint['model'], strict=True)

        n_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(
            'MDETR was loaded from fiqa/checkpoints/mdetr_best_checkpoint.pth',
            f'. Number of params: {n_parameters}', 
            sep=''
        )

        # We use a much simplified version of MDETR's make_coco_transforms()
        img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return model, img_transform

    else:
        assert False, f'Unknown VQA-model name: {args.vqa_model}'


def do_forward_pass(
    vqa_model: torch.nn.Module, vqa_model_name: str,
    img_transformed: torch.Tensor, questions: List[str]
) -> bool:
    """NB: currently, the function supports only len(questions) = 1."""
    if vqa_model_name == 'none':
        return vqa_model(img_transformed, questions)

    elif vqa_model_name == 'mdetr':
        from fiqa.task_handlers.vqa_models.mdetr import QACriterionAlfred

        memory_cache = vqa_model(
            img_transformed, questions, encode_and_save=True
        )
        outputs = vqa_model(
            img_transformed, questions,
            encode_and_save=False, memory_cache=memory_cache
        )
        # We use multi-head version since its performance is better
        ques_type_pred = outputs['pred_answer_type'].argmax(-1).item()
        ques_type_pred = QACriterionAlfred.id2type[ques_type_pred]
        prob = outputs[f'pred_answer_{ques_type_pred}'].sigmoid().item()
        
        verdict = prob >= 0.5
        return verdict

    else:
        assert False, f'Unknown VQA-model name: {vqa_model_name}'
