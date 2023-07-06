import torch
from datasets import get_gcl_dataset
from models import get_model
from utils.status import progress_bar
from utils.tb_logger import *
from utils.status import create_fake_stash
from models.utils.continual_model import ContinualModel
from argparse import Namespace


def evaluate(model: ContinualModel, dataset, eval_ema=False) -> float:
    """
    Evaluates the final accuracy of the model.
    :param model: the model to be evaluated
    :param dataset: the GCL dataset at hand
    :return: a float value that indicates the accuracy
    """
    curr_model = model.net
    if eval_ema:
        print('setting evaluation model to EMA model')
        curr_model = model.ema_model
    curr_model.eval()
    correct, total = 0, 0
    while not dataset.test_over:
        inputs, labels = dataset.get_test_data()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs[0].data, -1)
        correct += torch.sum(predicted == labels).item()
        total += labels.shape[0]

    acc = correct / total * 100
    return acc


def train(args: Namespace):
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """

    if args.csv_log:
        from utils.loggers import CsvLogger
    dataset = get_gcl_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    model.net.to(model.device)

    model_stash = create_fake_stash(model, args)

    ema_results, ema_results_mask_classes = [], []

    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        # model_stash['tensorboard_name'] = tb_logger.get_name()
        # csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, tb_logger.get_log_dir())
        task_perf_path = os.path.join(tb_logger.get_log_dir(), 'task_performance.txt')
        task_perf_path_ema = os.path.join(tb_logger.get_log_dir(), 'task_performance_ema.txt')

    model.net.train()
    epoch, i = 0, 0
    t, lr = 0, 0

    while not dataset.train_over:
        inputs, labels, not_aug_inputs = dataset.get_train_data()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        not_aug_inputs = not_aug_inputs.to(model.device)

        if args.retain and dataset.completed_rounds < 9:
            loss, loss_rot = model.observe_1(inputs, labels, not_aug_inputs, t, lr)
        elif args.revise and dataset.completed_rounds in range(9, 18):
            loss, loss_rot = model.observe_2(inputs, labels, not_aug_inputs, t, lr)
        else:
            loss, loss_rot = model.observe_3(inputs, labels, not_aug_inputs, t, lr)

        if args.tensorboard:
            tb_logger.log_loss_gcl(loss, i)
        progress_bar(i, dataset.LENGTH // args.batch_size, epoch, 'C', loss)

        i += 1

    if model.NAME == 'joint_gcl':
      model.end_task(dataset)

    acc = evaluate(model, dataset)
    print('Accuracy:', acc)

    dataset = get_gcl_dataset(args)
    ema_accs = evaluate(model, dataset, eval_ema=True)
    print('Accuracy:', ema_accs)

    try:
        f = open(os.path.join(tb_logger.get_log_dir(), 'ece.txt'), "w")
        f.write('Accuracy score: {}'.format(acc))
        f.close()
    except:
        raise Exception('Unable to write accuracy to a file!')
