import argparse


def parse_args(str_args: str = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='FIQA and AI2THOR parameters')

    # =============== General pipeline arguments ===============
    parser.add_argument(
        '--seed', type=int, default=1, help='Random seed (default: 1)'
    )
    parser.add_argument(
        '--run_name', type=str, default='first_run',
        help='The name of the directory inside results/[data_split]/ ' \
            + 'where all logs will be saved. Must be unique.'
    )
    parser.add_argument(
        '--save_imgs', action='store_true',
        help='Whether to save images from the simulator during the episodes'
    )
    parser.add_argument(
        '--dataset', type=str, default='alfred', choices=['alfred', 'teach'],
        help='The dataset to run on'
    )
    parser.add_argument(
        '--planner', type=str, choices=['no_replan', 'with_replan'],
        help='Which planner to use'
    )
    parser.add_argument(
        '--subdataset_type', type=str, default='none',
        choices=['changing_states', 'pose_corrections'],
        help='What type of the collected subdataset to use'
    )

    # =============== ALFRED related arguments ===============
    parser.add_argument(
        '--from_idx', type=int, default=0, help='Episode index to start from'
    )
    parser.add_argument(
        '--to_idx', type=int, default=204, help='Episode index to end on'
    )
    parser.add_argument(
        '--reward_config', default='alfred_utils/models/config/rewards.json'
    )
    parser.add_argument(
        '--max_fails', type=int, default=10,
        help='Max API execution failures before episode termination'
    )
    parser.add_argument(
        '--max_steps', type=int, default=1000,
        help='Max steps before episode termination'
    )

    # =============== SubtaskChecker and VQA related arguments ===============
    # General arguments
    parser.add_argument(
        '--checker', type=str, default='none',
        choices=[
            'none', 'oracle', 'oracle_with_noise', 'vqa', 'seg_and_vqa',
            'frames_diff_based'
        ],
        help='Type of SubtaskChecker to use'
    )
    parser.add_argument(
        '--existence_only_checker', action='store_true',
        help='Whether to check only existence or not'
    )
    parser.add_argument(
        '--interaction_only_checker', action='store_true',
        help='Whether to check only existence or not'
    )
    parser.add_argument(
        '--checker_correctness_prob', type=float,
        help='OracleSubtaskCheckerWithNoise correctness probability. '
            + 'Should be between 0.0 and 1.0'
    )
    parser.add_argument(
        '--vqa_model', type=str, default='none',
        choices=['none', 'mdetr'],
        help='VQA model to use in subtask checkers that use VQA'
    )
    parser.add_argument(
        '--vqa_gpu', type=int, default=0,
        help='GPU to use for the VQA model'
    )

    # MDETR Backbone
    parser.add_argument(
        '--backbone', default='resnet101', type=str,
        help='Name of the convolutional backbone to use ' \
            + 'such as resnet50 resnet101 timm_tf_efficientnet_b3_ns',
    )
    parser.add_argument(
        '--dilation', action='store_true',
        help='If true, we replace stride with dilation ' \
            + 'in the last convolutional block (DC5)',
    )
    parser.add_argument(
        '--position_embedding', default='sine', type=str,
        choices=('sine', 'learned'),
        help='Type of positional embedding to use on top' 
        + 'of the image features',
    )
    # MDETR text encoder
    parser.add_argument(
        '--freeze_text_encoder', action='store_true',
        help='Whether to freeze the weights of the text encoder'
    )
    parser.add_argument(
        '--text_encoder_type', default='roberta-base',
        choices=('roberta-base', 'distilroberta-base', 'roberta-large'),
    )
    # MDETR transformer
    parser.add_argument(
        '--enc_layers', default=6, type=int,
        help='Number of encoding layers in the transformer',
    )
    parser.add_argument(
        '--dec_layers', default=6, type=int,
        help='Number of decoding layers in the transformer',
    )
    parser.add_argument(
        '--dim_feedforward', default=2048, type=int,
        help='Intermediate size of the feedforward layers ' \
            + 'in the transformer blocks',
    )
    parser.add_argument(
        '--hidden_dim', default=256, type=int,
        help='Size of the embeddings (dimension of the transformer)',
    )
    parser.add_argument(
        '--dropout', default=0.1, type=float,
        help='Dropout applied in the transformer'
    )
    parser.add_argument(
        '--nheads', default=8, type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        '--num_queries', default=100, type=int, help='Number of query slots'
    )
    parser.add_argument(
        '--pre_norm', action='store_true',
        help='Whether to use pre-LayerNorm'
    )
    parser.add_argument(
        '--no_pass_pos_and_query', dest='pass_pos_and_query',
        action='store_false',
        help='Disables passing the positional encodings ' \
            + 'to each attention layers',
    )
    # MDETR general
    parser.add_argument(
        '--split_qa_heads', action='store_true',
        help='Whether to use a separate head per question type in vqa'
    )
    parser.add_argument(
        '--qa_dataset', type=str, default='alfred'
    )
    # MDETR training hyper-parameters
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    # MDETR losses
    parser.add_argument(
        '--no_aux_loss', dest='aux_loss', action='store_false',
        help='Disables auxiliary decoding losses (loss at each layer)',
    )
    parser.add_argument(
        '--contrastive_loss', action='store_true',
        help='Whether to add contrastive loss'
    )
    parser.add_argument(
        '--contrastive_loss_hdim', type=int, default=64,
        help='Projection head output size before computing normalized ' \
            + 'temperature-scaled cross entropy loss',
    )
    parser.add_argument(
        '--temperature_NCE', type=float, default=0.07,
        help='Temperature in the  temperature-scaled cross entropy loss'
    )
    # MDETR Segmentation
    parser.add_argument('--masks', action='store_true')  # TODO: do we need it?

    # =============== Language processing arguments ===============
    parser.add_argument(
        '--split', type=str,
        choices=[
            'train', 'valid_unseen', 'valid_seen', 'tests_seen', 'tests_unseen'
        ],
        required=True,
        help='ALFRED data split'
    )
    parser.add_argument(
        '--path_to_scene_names', type=str, default='alfred_utils/data/splits',
        help='Path to the oct21.json folder'
    )
    parser.add_argument(
        '--lp_data', type=str, default='alfred_utils/data',
        help='Path to json_2.1.0 folder'
    )
    parser.add_argument(
        '--path_to_instructions', type=str,
        default='fiqa/language_processing/processed_instructions',
        help='Path to preprocessed instructions folder'
    )
    parser.add_argument(
        '--instr_type', type=str, default='recept',
        choices=[
            'film', 'no_recept', 'recept', 'recept+nav'
        ],
        help='Instructions processing type. ' \
            + 'See fiqa/language_processing/model.py'
    )
    parser.add_argument(
        '--use_gt_instrs', action='store_true', help='Whether to use gt instructions'
    )

    # ============== Subtask executor arguments (navigation + interaction) ==============
    parser.add_argument(
        '--navigator_gpu', type=int, default=0, help='GPU to use for navigation'
    )
    parser.add_argument(
        '--interactor_gpu', type=int, default=0,
        help='GPU to use for interaction (segmentation)'
    )
    parser.add_argument(
        '--debug', action='store_true', help='Whether to print debug information or not'
    )
    parser.add_argument(
        '--draw_debug_imgs', action='store_true',
        help='Whether to draw debug images or not'
    )
    parser.add_argument(
        '--navigator', type=str, default='random',
        choices=[
            'ddppo_resnet_gru', 'ddppo_clip_gru', 'random', 'film', 'oracle'
        ],
        help='What navigation algorithm to use'
    )
    parser.add_argument('--allow_retry_nav_for_oracle_nav', action='store_true')
    parser.add_argument('--film_use_stop_analysis', action='store_true')
    parser.add_argument(
        '--interactor', type=str, default='seg_based',
        choices=[
            'trivial_seg_based', 'advanced_seg_based', 'oracle_seg_based'
        ],
        help='What interactor to use'
    )
    parser.add_argument(
        '--seg_model', type=str, default='none',
        choices=[
            'none', 'oracle', 'segformer', 'maskrcnn', 
            'segformer_and_maskrcnn'
        ],
        help='Model for segmentation'
    )
    parser.add_argument(
        '--depth_model', type=str, default='none',
        choices=['none', 'leres'],
        help='Model for depth estimation'
    )

    args = parser.parse_args(args=str_args)
    args.change_states = args.subdataset_type == 'changing_states'
    return args
