# InDiD: Instant Disorder Detection via a Principled Neural Network
This repository contains the source code to reproduce experiments from the [paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548182). The code can be used for the experiments with our sequence-to-sequence models ('InDiD', 'BCE' and 'BCE+InDiD'), KL-CPD and TS-CP&#x00B2; baselines, as well as the classic CPD baselines from the [ruptures](https://github.com/deepcharles/ruptures) package.

In addition, we provide a full version of Appendix with the expanded classic methods result, data examples and implementation details which were cut from the paper due to space limitations.

## Environment
You may find a list of the necessary packages in `requirements.txt` file.

## Download data
To reproduce experiments from the paper, you need to load the necessary datasets [here](https://disk.yandex.ru/d/_PQyni3AhyLu5g) and place them into the `data/` directory. Note that SyntheticNormal datasets are generated on-the-run so there is no data to be downloaded for these experiments.

The mark-up files for the video datasets are to be found in the corresponding directories as well. In addition, we provide the notebook `generate_MNIST_sequences_with_CVAE.ipynb` with the source code for the generation of MNIST sequences dataset.

## Training a model
To train a model, you should use `train.py` script. For example, the following command 
```
python train.py --model_type seq2seq -- loss_type indid --experiments_name mnist
``` 
will train a seq2seq model with our principled InDiD loss on the dataset of MNIST images. The trained model marked by a current timestamp will be saved into `saves/models/mnist/` directory.

The full list of options for `train.py`:
  * --model_type - type of the model to be trained (available options are 'seq2seq', 'kl_cpd' and 'tscp');
  * --loss_type - loss function for 'seq2seq' models (available options are 'indid', 'bce' and 'combined');
  * --experiments_name - dataset to be used (available options are 'synthtic_1D', 'synthetic_100D', 'human_activity', 'mnist', 'explosion' and 'road_accidents');
  * --seed - random seed to be fixed for reproducibility (otional, default is 102);
  * --dryrun - if True, do not save the model (optional, default is False);
  * --gpus - number of GPUs available for training (optional, default is 0);
  * --num_workers - number of available CPUs (optional, default is 2).

## Evaluating a model
To evaluate a pre-trained model, you should use `test.py` script. For example, the following command
```
python test.py --model_type seq2seq -- loss_type indid --experiments_name mnist --timestamp 230104T075800
```
will load a pre-trained model named 'model_seq2seq_indid_230104T075800.pth' saved in the `saves/models/mnist/` directory.

The full list of options for `test.py`:
  * --model_type - type of the model to be trained (available options are 'seq2seq', 'kl_cpd' and 'tscp');
  * --loss_type - loss function for 'seq2seq' models (available options are 'indid', 'bce' and 'combined');
  * --experiments_name - dataset to be used (available options are 'synthtic_1D', 'synthetic_100D', 'human_activity', 'mnist', 'explosion' and 'road_accidents');
  * --timestamp - timestamp identifying the model;
  * --threchold_number (or -tn) - number of alarm thresholds for the evaluation (optional, default is 100);
  * --device - 'cpu' or 'cuda' if available (optional, default is 'cpu');
  * --verbose - if true, the result metrics will be printed to the console (optional, default is False).

## Running a full experiment
To run a full experiment (initialize, train, save and evaluate a model), you may use `run.py` script. For example, the following command
```
python run.py --model_type seq2seq -- loss_type indid --experiments_name mnist 
```
will run `train.py` and `test.py` scripts with the corresponding arguments in sequence.

The full list of options for `run.py` :
  * --model_type - type of the model to be trained (available options are 'seq2seq', 'kl_cpd' and 'tscp');
  * --loss_type - loss function for 'seq2seq' models (available options are 'indid', 'bce' and 'combined');
  * --experiments_name - dataset to be used (available options are 'synthtic_1D', 'synthetic_100D', 'human_activity', 'mnist', 'explosion' and 'road_accidents');
  * --seed - random seed to be fixed for reproducibility (otional, default is 102);
  * --dryrun - if True, do not save the model (optional, default is False);
  * --gpus - number of GPUs available for training (optional, default is 0);
  * --num_workers - number of available CPUs (optional, default is 2);
  * --threchold_number (or -tn) - number of alarm thresholds for the evaluation (optional, default is 100);
  * --verbose - if true, the result metrics will be printed to the console (optional, default is False).

## Specifying model's architecture and training parameters
For each experiment, the models' architectures and the other loss and training parameters are specified in config files in `configs/` directory. These parameters may be changed if needed.

## Experiments with classic baselines.
We use 3 classic baselines (Binseg, Pelt and KernelCPD) implemented in [ruptures](https://github.com/deepcharles/ruptures) package. To reproduce this experiments, you should use `classic_baselines.py` script. For example, the following command
```
python classic_baselines.py --model_type classic_kernel --experiments_name synthetic_1D --kernel linear --pen 21
```
will create the KernelCPD model with 'linear' kernel and penalty parameter equal to 21 and evaluate it on the 1-dimensional SyntheticNormal dataset.

The full list of options for `classic_baselines.py` :
  * --model_type - type of the classic baseline (available options are 'classic_binseg', 'classic_pelt' and 'classic_kernel');
  * --experiments_name - dataset to be used (for classic baselines, available options are 'synthtic_1D', 'synthetic_100D', 'human_activity' and 'mnist);
  * --n_pred - number of change points to be predicted (optional, default is None);
  * --pen - penalty parameter  for a ruptures model (optional, default is None);
  * --kernel - type of the kernel to be used in the KernelCPD model (available options are 'linear', 'rbf' and cosine; default is 'linear');
  * --core_model - model parameter for Binseg and Pelt (available options are 'l1', 'l2' and 'rbf'; default is 'l2');
  * --seed - random seed to be fixed for reproducibility (otional, default is 102);
  * --verbose - if true, the result metrics will be printed to the console (optional, default is False).
  
Please, note that for the classic baselines, either 'n_pred' or 'pen' parameter should be specified.

## Experiments with custom models and datasets
You may experiment with custom core models or/and datasets using our CPD utilities. To do so, please, look at the examples provided in the jupyter notebook `custom_experiment.ipynb`.
