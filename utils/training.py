import torch
from torch.nn import DataParallel
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from utils.buffer_tricks import Buffer
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys
import math
from tqdm import tqdm
import numpy as np
from torch.utils.data import TensorDataset
import statistics
from copy import deepcopy
import pickle


def save_task_perf(savepath, results, n_tasks):

    results_array = np.zeros((n_tasks, n_tasks))
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i >= j:
                results_array[i, j] = results[i][j]

    np.savetxt(savepath, results_array, fmt='%.2f')


def adjust_learning_rate(model: ContinualModel, lr, t):
    for param_group in model.opt.param_groups:
        param_group['lr'] = lr
        print("learning rate: ", param_group['lr'])


def adjust_learning_rate_sparse(model: ContinualModel, lr, t):
    for param_group in model.opt_sparse.param_groups:
        param_group['lr'] = lr
        print("learning rate: ", param_group['lr'])


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, eval_ema=False, eval_sparse=False, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    curr_model = model.net
    if eval_ema:
        # print('setting evaluation model to EMA model')
        curr_model = model.ema_model
    elif eval_sparse:   # for sparse model
        curr_model = model.sparse_model

    status = curr_model.training
    curr_model.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = curr_model(inputs, k)
            else:
                outputs = curr_model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    curr_model.train(status)
    return accs, accs_mask_classes


def update_buffer_one_epoch(train_loader, dataset, model):
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            if hasattr(dataset.train_loader.dataset, 'logits'):
                inputs, labels, not_aug_inputs, logits = data
                inputs = inputs.to(model.device)
                labels = labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)
                logits = logits.to(model.device)
            else:
                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)

            outputs = model(inputs)
            model.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels,
                                 logits=outputs.data)


def train_one_epoch(train_loader, dataset, model, t, epoch, lr, cl_mask, observe=1):
    loss_main, loss_aux = 0, 0
    for i, data in enumerate(train_loader):
        if hasattr(dataset.train_loader.dataset, 'logits'):
            inputs, labels, not_aug_inputs, logits = data
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            not_aug_inputs = not_aug_inputs.to(model.device)
            logits = logits.to(model.device)
            if observe == 1:
                loss, loss_rot = model.observe_1(inputs, labels, not_aug_inputs, logits, t, lr, cl_mask)
            elif observe == 2:
                loss, loss_rot = model.observe_2(inputs, labels, not_aug_inputs, logits, t, lr, cl_mask)
            else:
                loss, loss_rot = model.observe_3(inputs, labels, not_aug_inputs, logits, t, lr, cl_mask)

        else:
            inputs, labels, not_aug_inputs = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            not_aug_inputs = not_aug_inputs.to(model.device)
            if observe == 1:
                loss, loss_rot = model.observe_1(inputs, labels, not_aug_inputs, t, lr, cl_mask)
            elif observe == 2:
                loss, loss_rot = model.observe_2(inputs, labels, not_aug_inputs, t, lr, cl_mask)
            else:
                loss, loss_rot = model.observe_3(inputs, labels, not_aug_inputs, t, lr, cl_mask)

        loss_main += loss
        loss_aux += loss_rot
        progress_bar(i, len(train_loader), epoch, t, loss)

    return loss_main, loss_aux


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []
    results_ema, results_mask_classes_ema = [], []
    model_stash = create_stash(model, args, dataset)
    lr_fl = args.lr_fl  # first loop lr
    lr_sl = args.lr_sl  # second loop lr

    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, tb_logger.get_log_dir())
        task_perf_path = os.path.join(tb_logger.get_log_dir(),  'task_performance.txt')
        task_perf_path_ema = os.path.join(tb_logger.get_log_dir(), 'task_performance_ema.txt')

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    if model.NAME != 'icarl' and model.NAME != 'pnn':   # and model.NAME != 'rr_tricks'
        random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    inter_acc = {}
    inter_acc_ema = {}
    for t in range(dataset.N_TASKS):
        stage_acc = []
        stage_acc_ema = []  # ema
        model.net.train()

        # Single Head Classifier
        if args.use_cl_mask:
            cur_classes = np.arange(t*dataset.N_CLASSES_PER_TASK, (t+1)*dataset.N_CLASSES_PER_TASK)
            cl_mask = np.setdiff1d(np.arange(dataset.N_CLASSES_PER_TASK * dataset.N_TASKS), cur_classes)
        else:
            cl_mask = None

        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

            accs_ema = evaluate(model, dataset, last=True, eval_ema=True)
            results_ema[t - 1] = results_ema[t - 1] + accs_ema[0]
            results_mask_classes_ema[t - 1] = results_mask_classes_ema[t - 1] + accs_ema[1]

        n_epochs = args.n_epochs
        epochs_obs1 = math.floor(args.train_budget_1 * args.n_epochs)   # Training budget of retain phase
        print("first loop: ", epochs_obs1)
        adjust_learning_rate(model, lr_fl, t)   # learning rate of retain phase

        for epoch in range(epochs_obs1):
            loss_main, loss_aux = train_one_epoch(train_loader, dataset, model, t, epoch, lr_fl, cl_mask,
                                                  observe=1)
            if epoch == args.forget_perc * epochs_obs1:
                model.net_epoch_k = deepcopy(model.net)  # Weight saved for Rewind step
            if args.tensorboard:
                acc = round(statistics.mean(evaluate(model, dataset)[0]), 2)
                tb_logger.log_accuracy_epoch(acc, n_epochs, epoch, t)
                acc_ema = round(statistics.mean(evaluate(model, dataset, eval_ema=True)[0]), 2)
                tb_logger.log_accuracy_epoch(acc_ema, n_epochs, epoch, t, eval_ema=True)
                tb_logger.log_loss(loss_main / len(train_loader), n_epochs, epoch, t)
                tb_logger.log_loss_rotation(loss_aux / len(train_loader), n_epochs, epoch, t)

        stage_acc.append(round(statistics.mean(evaluate(model, dataset)[0]), 2))  # stage 1
        print("stage 1: ", stage_acc)
        stage_acc_ema.append(round(statistics.mean(evaluate(model, dataset, eval_ema=True)[0]), 2))  # stage 1
        print("stage 1 (ema): ", stage_acc_ema)

        if t != dataset.N_TASKS + 1:
            #   Compute fisher information matrix
            if args.pruning_technique == 'fisher_pruning':
                model.net_copy = deepcopy(model.net)
                model.compute_fisher(dataset)

            stage_acc.append(round(statistics.mean(evaluate(model, dataset)[0]), 2))  # stage 2
            print("stage 2: ", stage_acc)
            stage_acc_ema.append(round(statistics.mean(evaluate(model, dataset, eval_ema=True)[0]), 2))  # stage 2
            print("stage 2 (ema): ", stage_acc_ema)

            #  Create a new sparse model for the current task
            model.extract_new_sparse_model(t)

            # Create a copy of the weights before masking out current non-sparse weights
            model.net_copy = deepcopy(model.net)

            # Training budget of Revise depending on whether Rewind phase is there or not.
            if args.rewind_tuning_incl and t < dataset.N_TASKS - 1:
                epochs_obs2 = math.floor(args.train_budget_2 * n_epochs)
            else:
                epochs_obs2 = n_epochs - epochs_obs1
            print("second loop: ", epochs_obs2)
            adjust_learning_rate(model, lr_sl, t)   # learning rate of revise - ideally a lower value than lr_fl

            for epoch in range(epochs_obs2):
                loss_main, loss_aux = train_one_epoch(train_loader, dataset, model, t, epoch + epochs_obs1,
                                                      lr_sl, cl_mask, observe=2)
                if args.tensorboard:
                    acc = round(statistics.mean(evaluate(model, dataset)[0]), 2)
                    tb_logger.log_accuracy_epoch(acc, n_epochs, epoch + epochs_obs1, t)
                    acc_ema = round(statistics.mean(evaluate(model, dataset, eval_ema=True)[0]), 2)
                    tb_logger.log_accuracy_epoch(acc_ema, n_epochs, epoch + epochs_obs1, t, eval_ema=True)
                    tb_logger.log_loss(loss_main / len(train_loader), n_epochs, epoch + epochs_obs1, t)
                    tb_logger.log_loss_rotation(loss_aux / len(train_loader), n_epochs, epoch + epochs_obs1, t)

            stage_acc.append(round(statistics.mean(evaluate(model, dataset)[0]), 2))  # stage 4
            print("stage 4: ", stage_acc)
            stage_acc_ema.append(round(statistics.mean(evaluate(model, dataset, eval_ema=True)[0]), 2))  # stage 4
            print("stage 4 (ema): ", stage_acc_ema)

            # Check the amount of overlap with current and cumulative sparse sets
            overlap = model.measure_overlap()
            if args.tensorboard:
                tb_logger.log_sparse_overlap(overlap, t+1)

            # Update the cumulative sparse set
            model.update_sparse_set()

            if args.rewind_tuning_incl and t < dataset.N_TASKS - 1:  # Rewind is not needed in the last task

                if args.reparameterize:
                    model.reparameterize_non_sparse()   # rewind to the saved weights
                    
                print("model rewinded and reparameterized")
                stage_acc.append(round(statistics.mean(evaluate(model, dataset)[0]), 2))  # stage 5
                print("stage 5: ", stage_acc)
                stage_acc_ema.append(round(statistics.mean(evaluate(model, dataset, eval_ema=True)[0]), 2))  # stage 5
                print("stage 5 (ema): ", stage_acc_ema)

                epochs_obs3 = n_epochs - epochs_obs1 - epochs_obs2  # rewind
                print("third loop: ", epochs_obs3)
                adjust_learning_rate(model, lr_fl, t)
                for epoch in range(0, epochs_obs3):
                    loss_main, loss_aux = train_one_epoch(train_loader, dataset, model, t, epoch + epochs_obs1 + epochs_obs2,
                                                          lr_fl, cl_mask, observe=3)
                    if args.tensorboard:
                        acc = round(statistics.mean(evaluate(model, dataset)[0]), 2)
                        tb_logger.log_accuracy_epoch(acc, n_epochs, epoch + epochs_obs1 + epochs_obs2, t)
                        acc_ema = round(statistics.mean(evaluate(model, dataset, eval_ema=True)[0]), 2)
                        tb_logger.log_accuracy_epoch(acc_ema, n_epochs, epoch + epochs_obs1 + epochs_obs2, t, eval_ema=True)
                        tb_logger.log_loss(loss_main / len(train_loader), n_epochs, epoch + epochs_obs1 + epochs_obs2, t)
                        tb_logger.log_loss_rotation(loss_aux / len(train_loader), n_epochs, epoch + epochs_obs1 +
                                                    epochs_obs2, t)

                stage_acc.append(round(statistics.mean(evaluate(model, dataset)[0]), 2))  # stage 6
                print("stage 6: ", stage_acc)
                stage_acc_ema.append(round(statistics.mean(evaluate(model, dataset, eval_ema=True)[0]), 2))  # stage 6
                print("stage 6 (ema): ", stage_acc_ema)

            sparsity_measure = model.measure_amount_of_sparsity()
            if args.tensorboard:
                tb_logger.log_measure_sparsity(sparsity_measure, t+1)

            # update the buffer at task boundary
            if not args.reservoir_buffer:
                update_buffer_one_epoch(train_loader, dataset, model)

            if args.reset_act_counters:
                for name, module in model.net.named_modules():
                    if 'kwinner' in name:
                        if hasattr(module, 'act_count'):
                            module.act_count *= 0

        # add all stage accuracies dictionary
        inter_acc[t+1] = stage_acc
        inter_acc_ema[t + 1] = stage_acc_ema

        if hasattr(model, 'end_task'):
            model.end_task(dataset, tb_logger)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        accs_ema = evaluate(model, dataset, eval_ema=True)
        results_ema.append(accs_ema[0])
        results_mask_classes_ema.append(accs_ema[1])    # for logging ema results

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        mean_acc_ema = np.mean(accs_ema, axis=1)
        print_mean_accuracy(mean_acc_ema, t + 1, dataset.SETTING)

        model_stash['mean_accs'].append(mean_acc)

        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)
            tb_logger.log_stage_accuracy(np.array(stage_acc), args, t)
            csv_logger.log(mean_acc)
            csv_logger.log(mean_acc_ema, ema=True)

    # Write stage accuracy dict to file
    fname = os.path.join(tb_logger.get_log_dir(), 'stage_accuracies.txt')
    with open(fname, 'w') as f:
        for key, value in inter_acc.items():
            f.write('%s:%s\n' % (key, value))

    fname_ema = os.path.join(tb_logger.get_log_dir(), 'stage_accuracies_ema.txt')
    with open(fname_ema, 'w') as f:
        for key, value in inter_acc_ema.items():
            f.write('%s:%s\n' % (key, value))

    # Write k-winner masks dict to file
    fname = os.path.join(tb_logger.get_log_dir(), 'kwinner_masks.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(model.kwinner_mask, f)

    if args.tensorboard:
        tb_logger.close()
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)

        if model.NAME != 'icarl' and model.NAME != 'pnn':   # and model.NAME != 'rr_tricks'
            csv_logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)
        csv_logger.write(vars(args))

    # save checkpoint
    fname = os.path.join(tb_logger.get_log_dir(), 'checkpoint.pth')
    torch.save(model.net.state_dict(), fname)

    fname = os.path.join(tb_logger.get_log_dir(), 'checkpoint_ema.pth')
    torch.save(model.ema_model.state_dict(), fname)
