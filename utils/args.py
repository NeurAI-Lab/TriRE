from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_arguments(parser):
    # parser.add_argument('--train_ssl', action='store_true',
    #                     help='Use SSL for task agnostic learning.')
    # parser.add_argument('--ssl_algo', type=str, default='supcontrast',
    #                     help='SSL training algorithm')
    # parser.add_argument('--multitask', action='store_true',
    #                     help='Use rotation as multitask')
    # parser.add_argument('--empty_buffer', action='store_true',
    #                     help='Do not use any buffer for experience replay')
    # parser.add_argument('--ce_weight', type=float, default=1,
    #                     help='multitask weight for cross entropy')
    # parser.add_argument('--rot_weight', type=float, default=1,
    #                     help='multitask weight for rotation')
    parser.add_argument('--img_size', type=int, required=True,
                        help='Input image size')
    # parser.add_argument('--ssl_train_percentage', type=float, default=0.9,
    #                     help='Percentage of training time for SSL. Rest is for multitasking')
    # parser.add_argument('--eval_c', action='store_true',
    #                     help='Use trained model for evaluation on natural corruption datasets')
    # parser.add_argument('--slim_factor', type=float, default=0.9,
    #                     help='Slim factor for split networks')
    # parser.add_argument('--retain_revise', action='store_true',
    #                     help='After every task, retain part of the backbone and rest reset')
    # parser.add_argument('--rr_method', type=str, default="wels", choices=['llf', 'wels', 'kels', 'spr',
    #                                                                       'twoone','rifle', 'imp', 'dsd', 'lw',
    #                                                                       'rrr_ema', 'rrr'],
    #                     help='Approach for retain and revise of backbone')
    # parser.add_argument('--re_init', type=str, default="xavier_uniform",
    #                     help='Technique for re-initializing the part of the network')
    # parser.add_argument('--llf_threshold', type=int, default=10,
    #                     help='LLF threshold (10 for starting from block-3, 14 from block-4 in ResNet-18)')
    # parser.add_argument('--lw_threshold', type=int, default=1,
    #                     help='LW threshold (10 for starting from block-3, 14 from block-4 in ResNet-18)')
    # parser.add_argument('--lw_mask_blocks', type=int, default=1,
    #                     help='LW - number of blocks to mask at once after each generation')
    # parser.add_argument('--dsd_sparsity', type=float, default=0.3,
    #                     help='DSD sparsity threshold. These % of weights will be set to zero '
    #                          'in each layer during sparse training')
    # parser.add_argument('--wels_split_rate', type=float, default=0.3,
    #                     help='WELS split rate')
    # parser.add_argument('--rrr_retain_rate', type=float, default=0.7,
    #                     help='RRR split rate')
    # parser.add_argument('--rrr_fisher_ema_gamma', type=float, default=0.95,
    #                     help='RRR gamma value for the exponential moving average')
    # parser.add_argument('--kels_split_rate', type=float, default=0.3,
    #                     help='KELS split rate')
    # parser.add_argument('--spr_lambda_scale', type=float, default=0.9,
    #                     help='SPR lambda scale')
    # parser.add_argument('--spr_noise_scale', type=float, default=0.00002,
    #                     help='SPR noise scale')
    parser.add_argument('--sparsity', type=float, default=0.2,
                        help='Percentage of top-k weights from kwinner neurons/filters')
    parser.add_argument('--kwinner_sparsity', type=float, default=0.2,
                        help='Percentage of top-k neurons/ filters to be selected after each task')
    parser.add_argument('--slow_lr_multiplier', type=float, default=1,
                        help='Learning rate multiplier for sparse set in RR (o-ewc)')
    parser.add_argument('--slow_lr_multiplier_decay', action='store_true',
                        help='Decay slow LR multiplier over the course of training')
    parser.add_argument('--train_budget_1', type=float, default=0.6,
                        help='Training budget for first for loop in retain and revise')
    parser.add_argument('--train_budget_2', type=float, default=0.2,
                        help='Training budget for second for loop in retain and revise')
    parser.add_argument('--pruning_technique', type=str, default="magnitude_pruning",
                       help='Pruning technique for sparsifying the network')
    parser.add_argument('--non_overlaping_kwinner', action='store_true',
                        help='During structured pruning stage, do not use any '
                             'overlapping kwinner masks. Be sure that the net has enough capacity.'
                             'set sparsity < (1 / num_tasks). ')
    parser.add_argument('--reinit_technique', type=str, default="xavier",
                        help='Technique for re-initializing the part of the network')
    parser.add_argument('--reparameterize', action='store_true',
                        help='Whether to reparameterize the weights after each task')
    parser.add_argument('--mask_non_sparse_weights', action='store_true',
                        help='Whether to mask non sparse weights during forward pass of second loop')
    parser.add_argument('--mask_cum_sparse_weights', action='store_true',
                        help='Whether to mask cumulative sparse weights during forward pass of third loop')
    parser.add_argument('--reservoir_buffer', action='store_true',
                        help='Updates buffer in each iteration. If False, updates at task boundary. ')
    parser.add_argument('--reset_act_counters', action='store_true',
                        help='Reset activation counters after each task')
    parser.add_argument('--reg_weight', type=float, default=0.10,
                        help='EMA regularization weight')
    parser.add_argument('--stable_model_update_freq', type=float, default=0.05,
                        help='EMA update frequency')
    parser.add_argument('--stable_model_alpha', type=float, default=0.999,
                        help='EMA alpha')
    parser.add_argument('--rewind_tuning_incl', action='store_true', help='whether to finetune the observe_3')
    parser.add_argument('--sparse_model_finetuning', action='store_true', help='whether to finetune the sparse model '
                                                                               'exclusively')
    # parser.add_argument('--n_epochs', type=int, required=True,
    #                     help='The number of epochs for each task.')


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size.')
    parser.add_argument('--n_epochs', type=int, required=True,
                        help='The number of epochs for each task.')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, required=True,
                        help='The batch size of the memory buffer.')
