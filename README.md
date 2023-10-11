# TriRE: A Multi-Mechanism Learning Paradigm for Continual Knowledge Retention and Promotion

The official repository for NeurIPS'23 paper. We extended the original repo [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) with our method.

## How to run?
+ python main.py --dataset seq-cifar100 --model trire --buffer_size 200 --load_best_args --img_size 32 --tensorboard --reservoir_buffer --kwinner_sparsity 0.3 --pruning_technique CWI --sparsity 0.2 --lr_fl 0.002 --lr_sl 0.0001 --reset_act_counters --train_budget_1 0.6 --train_budget_2 0.2 --reparameterize --reinit_technique rewind --use_cl_mask --reg_weight 0.05 --stable_model_update_freq 0.1 --rewind_tuning_incl --use_het_drop
        
## Setup

+ Use `./utils/main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters from the paper.
+ New models can be added to the `models/` folder.
+ New datasets can be added to the `datasets/` folder.

## Datasets

**Class-Il / Task-IL settings**

+ Seq-CIFAR10
+ Seq-CIFAR100
+ SeqTinyImageNet
