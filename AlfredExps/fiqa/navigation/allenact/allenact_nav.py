import argparse
from typing import cast, MutableMapping, Optional, List

import torch

from utils.logger import logger
from fiqa.language_processing.subtask import Subtask
from fiqa.navigation.basics_and_dummies import NavigatorBase

from fiqa.navigation.allenact.projects.objectnav_baselines.mixins \
    import ResNetPreprocessGRUActorCriticMixin
from fiqa.navigation.allenact.projects.objectnav_baselines.experiments.clip.mixins \
    import ClipResNetPreprocessGRUActorCriticMixin
from fiqa.navigation.allenact.allenact_plugins.clip_plugin.clip_preprocessors \
    import ClipResNetPreprocessor
from fiqa.navigation.allenact.allenact_plugins.ithor_plugin.ithor_sensors \
    import GoalObjectTypeThorSensor, RGBSensorThor
# from fiqa.navigation.allenact.projects.objectnav_baselines.experiments.ithor.objectnav_ithor_base \
#     import ObjectNaviThorBaseConfig
from fiqa.navigation.allenact.allenact.base_abstractions.preprocessor \
    import SensorPreprocessorGraph
from fiqa.navigation.allenact.allenact.base_abstractions.sensor \
    import SensorSuite
from fiqa.navigation.allenact.allenact.algorithms.onpolicy_sync.policy \
    import ActorCriticModel
from fiqa.navigation.allenact.allenact.base_abstractions.misc \
    import ActorCriticOutput


class AllenActNavBase(NavigatorBase):
    """
    A base class for the AllenAct baseline models for the ObjectNav task.
    A derived class should only define `detectable_obj_types`, `obj_types2id` 
    and `__init__()`.
    """

    detectable_obj_types = None
    obj_types2id = None
    # Actions were adapted from
    # fiqa.navigation.allenact.allenact_plugins.robothor_plugin.robothor_tasks.ObjectNavTask
    actions_list = [
        'MoveAhead', 'RotateLeft', 'RotateRight',
        'StopNav', 'LookUp', 'LookDown'
    ]

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.device = torch.device(
            f'cuda:{args.interactor_gpu}' if torch.cuda.is_available() else 'cpu'
        )

        self.sensor_preprocessor_graph = None
        self.actor_critic = None

        self.rnn_hidden_states = None
        self.actions = None
        self.not_done_masks = None
        self.horizon = 0
        self.new_objective = False
        self.rotations = 0
        self.retry_nav = False
        self.goal_obj = None

    def reset(self, subtask_queue: Optional[List[Subtask]] = None) -> None:
        # In all ALFRED train-split scenes, the initial angle is set to 30
        self.horizon = 30

    def reset_before_new_objective(
        self, subtask: Subtask, retry_nav: bool
    ) -> None:
        self.rnn_hidden_states = torch.zeros(
            1, self.actor_critic.num_recurrent_layers, 512, device=self.device
        )
        self.actions = torch.zeros(
            1, 1, dtype=torch.long, device=self.device
        )
        self.not_done_masks = torch.zeros(
            1, 1, 1, dtype=torch.bool, device=self.device
        )
        self.new_objective = True
        self.retry_nav = retry_nav
        self.rotations = 2 if retry_nav else 0

        # TODO: better handling missing objects
        goal_obj = subtask.obj
        if goal_obj not in type(self).obj_types2id:
            logger.log_warning(f'Navigator does not have {goal_obj}!')
            if 'Sliced' in goal_obj:
                goal_obj = goal_obj[:-6]
        self.goal_obj = torch.tensor(
            type(self).obj_types2id.get(goal_obj, 1)
        )

    def _setup_0_horizon(self) -> int:
        if self.horizon > 0:
            self.horizon -= 15
            sampled_action = AllenActNavBase.actions_list.index('LookUp')
        elif self.horizon < 0:
            self.horizon += 15
            sampled_action = AllenActNavBase.actions_list.index('LookDown')

        self.new_objective = self.horizon != 0
        return sampled_action

    def __call__(self, rgb: torch.Tensor) -> str:
        # If it is required to retry navigation, we want the navigator
        # to change the view of the target object, so we have to encourage it
        # to "choose" another trajectory
        if self.rotations:
            self.rotations -= 1
            return 'RotateLeft'
        # The model was trained to start the search from 0-horizon
        if self.new_objective and self.horizon != 0:
            return AllenActNavBase.actions_list[self._setup_0_horizon()]
        self.new_objective = False

        # Since ResNet is used (ResNetPreprocessor or
        # ClipResNetPreprocessor), the images must have already been
        # preprocessed. Moreover, rgb.shape must be (B, H, W, C).
        obs = {
            'rgb_lowres': rgb.permute((1, 2, 0)).unsqueeze(0).to(self.device),
            'goal_object_type_ind': self.goal_obj.unsqueeze(0).to(self.device)
        }
        obs = self.sensor_preprocessor_graph.get_observations(obs)
        # We also have to add `nstep` (i. e. T) dimension
        # to use forward encoder
        obs = {k: v.unsqueeze(0) for k, v in obs.items()}

        # 1.1 use perception model (i.e. encoder) to get observation embeddings
        obs_embeds = self.actor_critic.forward_encoder(obs)
        # 1.2 use embedding model to get the previous action embeddings
        prev_actions_embeds = self.actor_critic.prev_action_embedder(
            self.actions
        )
        joint_embeds = torch.cat((obs_embeds, prev_actions_embeds), dim=-1)
        # 2. use RNNs to get single/multiple beliefs
        beliefs_dict = {}
        for key, model in self.actor_critic.state_encoders.items():
            beliefs_dict[key], self.rnn_hidden_states = model(
                joint_embeds, self.rnn_hidden_states, self.not_done_masks
            )
        # 3. fuse beliefs for multiple belief models
        beliefs, _ = self.actor_critic.fuse_beliefs(
            beliefs_dict, obs_embeds
        )
        actor_critic_output = ActorCriticOutput(
            distributions=self.actor_critic.actor(beliefs),
            values=self.actor_critic.critic(beliefs),
            extras={}
        )
        distr = actor_critic_output.distributions
        self.actions = distr.sample()  # tensor with shape=(1, 1) and dtype=int
        sampled_action = AllenActNavBase.actions_list[self.actions.item()]

        # Indicate that the episode is still incomplete
        # (these masks are used for receiving a reward during training)
        if not self.not_done_masks.item():
            self.not_done_masks.fill_(True)

        if sampled_action in ['LookUp', 'LookDown']:
            self.horizon += 15 if sampled_action == 'LookDown' else -15
        return sampled_action


class DDPPOResNetGRU(AllenActNavBase):
    """
    One of the baseline models for the ObjectNav task. More info:
    https://github.com/allenai/allenact/tree/main/projects/objectnav_baselines

    NB. Since we are forced to use ai2thor==2.1.0, 
    the use of ai2thor >2.1.0 has been excluded.
    """

    # Already sorted:
    detectable_obj_types = [
        'AlarmClock', 'Apple', 'ArmChair', 'BaseballBat', 'BasketBall',
        'Bathtub', 'BathtubBasin', 'Bed', 'Blinds', 'Book', 'Boots', 'Bowl',
        'Box', 'Bread', 'ButterKnife', 'CD', 'Cabinet', 'Candle', 'CellPhone',
        'Chair', 'Cloth', 'CoffeeMachine', 'CoffeeTable', 'CounterTop',
        'CreditCard', 'Cup', 'Curtains', 'Desk', 'DeskLamp', 'Desktop',
        'DiningTable', 'DishSponge', 'DogBed', 'Drawer', 'Dresser',
        'Dumbbell', 'Egg', 'Faucet', 'Floor', 'FloorLamp', 'Footstool',
        'Fork', 'Fridge', 'GarbageBag', 'GarbageCan', 'HandTowel',
        'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain', 'Knife',
        'Ladle', 'Laptop', 'LaundryHamper', 'Lettuce', 'LightSwitch',
        'Microwave', 'Mirror', 'Mug', 'Newspaper', 'Ottoman', 'Painting',
        'Pan', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow',
        'Plate', 'Plunger', 'Poster', 'Pot', 'Potato', 'RemoteControl',
        'Safe', 'SaltShaker', 'ScrubBrush', 'Shelf', 'ShelvingUnit',
        'ShowerCurtain', 'ShowerDoor', 'ShowerGlass', 'ShowerHead',
        'SideTable', 'Sink', 'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa',
        'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'Stool', 'TVStand',
        'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'Toaster',
        'Toilet', 'ToiletPaperHanger', 'Tomato', 'Towel', 'TowelHolder',
        'VacuumCleaner', 'Vase', 'Watch', 'WateringCan', 'Window', 'WineBottle'
    ]
    obj_types2id = {
        obj_type: i for i, obj_type in enumerate(detectable_obj_types)
    }

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        SENSORS = [
            RGBSensorThor(
                # height=ObjectNaviThorBaseConfig.SCREEN_SIZE,
                height=224,
                # width=ObjectNaviThorBaseConfig.SCREEN_SIZE,
                width=224,
                use_resnet_normalization=True,
                uuid='rgb_lowres'
            ),
            GoalObjectTypeThorSensor(
                object_types=DDPPOResNetGRU.detectable_obj_types
            )
        ]
        preprocessing_and_model = ResNetPreprocessGRUActorCriticMixin(
            sensors=SENSORS,
            resnet_type='RN18',
            screen_size=224,
            goal_sensor_type=GoalObjectTypeThorSensor
        )
        self.sensor_preprocessor_graph = SensorPreprocessorGraph(
            source_observation_spaces=SensorSuite(
                [*SENSORS]
            ).observation_spaces,
            preprocessors=preprocessing_and_model.preprocessors()
        ).to(self.device)
        self.actor_critic = cast(
            ActorCriticModel,
            preprocessing_and_model.create_model(
                num_actions=6,
                sensor_preprocessor_graph=self.sensor_preprocessor_graph
            )
        )
        pretrained_state = torch.load(
            'fiqa/checkpoints/exp_ObjectNav-RoboTHOR-RGB-ResNet18GRU-DDPPO'
            + '__stage_00__steps_000165242880.pt',
            map_location='cpu'
        )['model_state_dict']
        self.actor_critic.load_state_dict(
            # state_dict=cast('OrderedDict[str, torch.Tensor]', pretrained_state)
            state_dict=cast(
                'MutableMapping[str, torch.Tensor]', pretrained_state
            )
        )
        self.actor_critic.to(self.device)


class DDPPOClipGRU(AllenActNavBase):
    """
    One of the baseline models for the ObjectNav task. More info:
    https://github.com/allenai/allenact/tree/main/projects/objectnav_baselines

    NB. Since we are forced to use ai2thor==2.1.0, 
    the use of ai2thor >2.1.0 has been excluded. 
    Also, CLIP is required to be installed.
    """

    # Already sorted:
    detectable_obj_types = [
        'AlarmClock', 'Apple', 'ArmChair', 'BaseballBat', 'BasketBall',
        'BathtubBasin', 'Bed', 'Book', 'Bowl', 'Box', 'Bread', 'ButterKnife',
        'CD', 'Cabinet', 'Candle', 'CellPhone', 'Cloth', 'CoffeeMachine',
        'CoffeeTable', 'CounterTop', 'CreditCard', 'Cup', 'Desk',
        'DiningTable', 'DishSponge', 'Drawer', 'Dresser', 'Egg', 'Faucet',
        'FloorLamp', 'Fork', 'Fridge', 'GarbageCan', 'HandTowel', 'Kettle',
        'KeyChain', 'Knife', 'Ladle', 'Laptop', 'Lettuce', 'Microwave', 'Mug',
        'Newspaper', 'Ottoman', 'Pan', 'Pen', 'Pencil', 'PepperShaker',
        'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'RemoteControl',
        'Safe', 'SaltShaker', 'Shelf', 'SideTable', 'SinkBasin', 'SoapBar',
        'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle', 'Statue',
        'TennisRacket', 'TissueBox', 'Toilet', 'ToiletPaperHanger', 'Tomato',
        'Vase', 'Watch', 'WateringCan', 'WineBottle'
    ]
    obj_types2id = {
        obj_type: i for i, obj_type in enumerate(detectable_obj_types)
    }

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        SENSORS = [
            # mean and stdev are required to be equal to the CLIP's values
            # when using the CLIP-based model, so they are set but not used,
            # since we emulate sensors by `transforms`
            RGBSensorThor(
                # height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
                height=224,
                # width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
                width=224,
                use_resnet_normalization=False,
                uuid='rgb_lowres',
                mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,  # not used
                stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,  # not used
            ),
            GoalObjectTypeThorSensor(
                object_types=DDPPOClipGRU.detectable_obj_types
            )
        ]
        preprocessing_and_model = ClipResNetPreprocessGRUActorCriticMixin(
            sensors=SENSORS,
            clip_model_type='RN50',
            # screen_size=ObjectNavRoboThorBaseConfig.SCREEN_SIZE
            screen_size=224,
            goal_sensor_type=GoalObjectTypeThorSensor
        )
        self.sensor_preprocessor_graph = SensorPreprocessorGraph(
            source_observation_spaces=SensorSuite(
                [*SENSORS]
            ).observation_spaces,
            preprocessors=preprocessing_and_model.preprocessors(
                use_normalization=False
            )
        ).to(self.device)
        self.actor_critic = cast(
            ActorCriticModel,
            preprocessing_and_model.create_model(
                num_actions=6,
                add_prev_actions=True,
                sensor_preprocessor_graph=self.sensor_preprocessor_graph
            )
        )
        pretrained_state = torch.load(
            'fiqa/checkpoints/exp_ObjectNav-RoboTHOR-RGB-ClipResNet50GRU-DDPPO'
            + '__stage_00__steps_000300001536.pt',
            map_location='cpu'
        )['model_state_dict']
        self.actor_critic.load_state_dict(
            state_dict=cast(
                'MutableMapping[str, torch.Tensor]', pretrained_state
            )
        )
        self.actor_critic.to(self.device)
