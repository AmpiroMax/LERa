import argparse
from typing import Tuple, Optional

import torchvision.transforms as T

from fiqa.alfred_thor_env import AlfredThorEnv

from fiqa.navigation.basics_and_dummies import NavigatorBase, RandomNavigator
from fiqa.task_handlers.info_retriever import InfoRetrieverBase, DummyInfoRetriever
from fiqa.task_handlers.interactor import InteractorBase


def build_navigator(
    args: argparse.Namespace, env: Optional[AlfredThorEnv] = None,
    interactor: Optional[InteractorBase] = None
) -> Tuple[NavigatorBase, T.Compose, InfoRetrieverBase]:
    if args.navigator == 'random':
        return RandomNavigator(args), T.Compose([]), DummyInfoRetriever()

    elif args.navigator == 'ddppo_resnet_gru':
        from fiqa.navigation.allenact.allenact_nav import DDPPOResNetGRU

        # These transforms are equivalent to the ones in RGBSensorThor
        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return DDPPOResNetGRU(args), transforms, DummyInfoRetriever()

    elif args.navigator == 'ddppo_clip_gru':
        from fiqa.navigation.allenact.allenact_nav import DDPPOClipGRU

        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        return DDPPOClipGRU(args), transforms, DummyInfoRetriever()

    elif args.navigator == 'film':
        from fiqa.navigation.film.film_nav import FILMNavigator, FILMNavInfoRetriever

        # Currently we have only segmentation based interactors and they have
        # `seg_model` and `img_transform_seg` fields
        seg_tuple = (interactor.seg_model, interactor.img_transform_seg)
        nav, transforms = FILMNavigator(args, seg_tuple), T.Compose([])
        return nav, transforms, FILMNavInfoRetriever(nav)

    elif args.navigator == 'oracle':
        from fiqa.navigation.oracle_nav import OracleNavigator

        assert 'tests' not in args.split, \
            'Oracle navigator is not allowed on test splits!'
        assert env is not None, 'Env must be set when using OracleNavigator!'
        return OracleNavigator(args, env), T.Compose([]), DummyInfoRetriever()

    else:
        assert False, f'Unknown navigator {args.navigator}'
