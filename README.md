# TriRE: A Multi-Mechanism Learning Paradigm for Continual Knowledge Retention and Promotion

The official repository for NeurIPS'23 paper. We extended the original repo [DER++](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) with our method.

<img width="721" alt="method_readme" src="https://github.com/NeurAI-Lab/TriRE/assets/57964849/d4870e27-f593-480a-b77a-9eb8bea2efaa">

## How to run TriRE?

+ Use `python main.py` to run experiments.
+ For example, for dataset Seq-CIFAR100, run \
  `python main.py --dataset seq-cifar100 --model trire --buffer_size 200 --load_best_args --img_size 32 --tensorboard --reservoir_buffer --kwinner_sparsity 0.3 --pruning_technique CWI --sparsity 0.2 --lr_fl 0.002 --lr_sl 0.0001 --reset_act_counters --train_budget_1 0.6 --train_budget_2 0.2 --reparameterize --reinit_technique rewind --use_cl_mask --reg_weight 0.05 --stable_model_update_freq 0.1 --rewind_tuning_incl --use_het_drop`

  where, `kwinner_sparsity` and `sparsity` represents the percentage of most activated neurons and corresponding most important weights to be retained at the end of Retain stage respectively.\
  `pruning_technique` : {'CWI', 'Magnitude Pruning', 'Fisher Information'}\
  `lr_fl` is the learning rate for Retain and Revise stages and `lr_sl` is the slowed learning rate for Revise stage. \
  `train_budget_1` and `train_budget_2` are the percentages of epochs dedicated to Retain and Revise stages which also implicates that the rest of the epochs are used for Rewind stage.\
  `use_cl_mask` indicates that the model is using a single head classifier.\
  `reinit_technique` : {xavier, rewind}
        
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

## Cite Our Work

If you find the code useful in your research, please consider citing our paper:

    
