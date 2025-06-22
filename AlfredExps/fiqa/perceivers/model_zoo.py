import argparse
import torch
from typing import Tuple, Optional

import torchvision.transforms as T

from fiqa.perceivers.basics_and_dummies import (
    SegModelBase, DepthModelBase,
    build_dummy_seg_model, build_dummy_depth_model
)
from fiqa.alfred_thor_env import AlfredThorEnv


# =============================== Segmentation ===============================
def build_seg_model(
    args: argparse.Namespace, env: Optional[AlfredThorEnv] = None
) -> Tuple[SegModelBase, T.Compose]:
    if args.seg_model == 'none':
        return build_dummy_seg_model()

    elif args.seg_model == 'oracle':
        from fiqa.perceivers.oracle_seg import build_oracle_seg_model

        assert 'tests' not in args.split, \
            'Oracle segmentation model is not allowed on test splits!'
        assert env is not None, \
            'Env must be set when using oracle segmentation!'
        return build_oracle_seg_model(env)

    elif args.seg_model == 'segformer':
        from fiqa.perceivers.segformer.segformer \
            import build as build_segformer

        return build_segformer(args)

    elif args.seg_model == 'maskrcnn':
        from fiqa.perceivers.maskrcnn.maskrcnn import build as build_maskrcnn

        return build_maskrcnn(args)

    elif args.seg_model == 'segformer_and_maskrcnn':
        from fiqa.perceivers.segformer_and_maskrcnn import \
            build as build_segformer_and_maskrcnn

        return build_segformer_and_maskrcnn(args)

    else:
        assert False, f'Unknown segmentation model name: {args.seg_model}'


# ============================= Depth estimation =============================
def build_depth_model(
    args: argparse.Namespace
) -> Tuple[DepthModelBase, T.Compose]:
    if args.depth_model == 'none':
        return build_dummy_depth_model()
    elif args.depth_model == 'leres':
        raise NotImplementedError()
    else:
        assert False, f'Unknown depth model name: {args.depth_model}'


def do_depth_forward_pass(
    depth_model: DepthModelBase, depth_model_name: str, rgb: torch.Tensor
) -> torch.Tensor:
    if depth_model_name == 'none':
        return depth_model(rgb)
    elif depth_model_name == 'leres':
        raise NotImplementedError()
    else:
        assert False, f'Unknown depth model name: {depth_model_name}'
