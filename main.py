import importlib
import os
import sys
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')

from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args, add_arguments
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
from torchsummary import summary


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def main():
    lecun_fix()
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    parser.add_argument('--het_drop', type=float, default=1, help='heterogenous dropout weight')
    parser.add_argument("--lr_fl", type=float, default=0.0005, help="first loop learning rate")
    parser.add_argument("--lr_sl", type=float, default=0.0001, help="second loop learning rate")
    parser.add_argument('--use_cl_mask', action='store_true', default=False, help='use CL mask or not')
    parser.add_argument("--output_folder", type=str, default="output-ai/temp_results")
    parser.add_argument("--use_het_drop", action='store_true', default=False, help='use heterogenous dropout or not')
    parser.add_argument("--forget_perc", type=float, default=0.9, help="Epoch to where the model is rewinded")

    add_arguments(parser)
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
        if hasattr(args, 'buffer_size'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        add_arguments(parser)
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.model == 'mer':
        setattr(args, 'batch_size', 1)
    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args)
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)


if __name__ == '__main__':
    main()
