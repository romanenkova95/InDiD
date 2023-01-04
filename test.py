"""Script that loads and evaluates a pre-trained CPD model."""
from utils import datasets, model_utils, metrics
from utils.arg_parsing import get_test_parser

import warnings
warnings.filterwarnings("ignore")

from typing import Dict

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def main(args_local: Dict):
    # create datasets
    experiments_name = args_local["experiments_name"]
    train_dataset, test_dataset = datasets.CPDDatasets(experiments_name=experiments_name).get_dataset_()

    model_name = args_local["model_type"]
    if model_name == "seq2seq":
        model_name = model_name + "_" + args_local["loss_type"]

    timestamp = args_local["timestamp"]
    save_path = Path("saves/models") / experiments_name / f'model_{model_name}_{timestamp}.pth'
    assert save_path.exists(), f"Checkpoint {str(save_path)} doesn't exist"

    # load checkpoint with the pre-trained model
    checkpoint = torch.load(save_path)
    state_dict = checkpoint["checkpoint"]
    args = checkpoint["args"]

    seed = args["seed"]
    model_utils.fix_seeds(seed)

    # initialize the model
    # take the last model in list for evaluation 
    # (the only one in case of 'indid', 'bce', 'kl_cpd' or 'tscp' models)
    model = model_utils.get_models_list(args, train_dataset, test_dataset)[-1]

    # load the weights
    model.load_state_dict(state_dict)

    threshold_number = args_local["threshold_number"]
    threshold_list = np.linspace(-5, 5, threshold_number)
    threshold_list = 1 / (1 + np.exp(-threshold_list))
    threshold_list = [-0.001] + list(threshold_list) + [1.001]

    # scale is needed only for KL-CPD and TS-CP2 baselines
    if args["model_type"] in ["kl_cpd", "tscp"]:
        scale = args["predictions"]["scale"]
    else:
        scale = None

    # compute metrics
    metrics_local, delay_list, fp_delay_list = \
        metrics.evaluation_pipeline(model,
                                    model.val_dataloader(),
                                    threshold_list,
                                    device=args_local["device"],
                                    model_type=args_local["model_type"],
                                    verbose=args_local["verbose"], 
                                    scale=scale)

    path_to_saves = Path("saves/results") / experiments_name
    path_to_metric = path_to_saves / "metrics"
    path_to_metric.mkdir(parents=True, exist_ok=True)

    # write metrics to .txt file
    metrics.write_metrics_to_file(
        filename=f'{str(path_to_metric)}/{model_name}.txt',
        metrics=metrics_local,
        seed=seed,
        timestamp=timestamp
        )

    # Plot the detection curve and save the figure
    plt.figure(figsize=(12, 12))
    plt.plot(fp_delay_list.values(), delay_list.values(), '-o', markersize=8, label=model_name)
    plt.xlabel('Mean Time to False Alarm', fontsize=28)
    plt.ylabel('Mean Detection Delay', fontsize=28)
    plt.title("Detection Curve", fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='upper left', fontsize=26);

    path_to_figure = path_to_saves / "figures"
    path_to_figure.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_figure / f'{model_name}_{timestamp}.png', dpi=300)

    # save metrics
    path_to_pickle = Path("saves/results/pickles/")
    path_to_pickle.mkdir(parents=True, exist_ok=True)
    metrics.dump_results(
        metrics_local, pickle_name=f'{str(path_to_pickle)}/{model_name}_{experiments_name}_{timestamp}.pickle'
    )

if __name__ == '__main__':
    parser = get_test_parser()
    args_local = parser.parse_args()
    args_local = dict(vars(args_local)) 
    main(args_local)