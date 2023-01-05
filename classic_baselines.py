"""Script that initializes and uses a Classic Baseline model for a particular experiment."""
from utils import datasets, model_utils, metrics
from utils.arg_parsing import get_classic_baseline_parser

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

from torch.utils.data import DataLoader

def main(args: dict) -> str:
    experiments_name = args["experiments_name"]

    if experiments_name in ["explosion", "road_accidents"]:
        raise ValueError("Classic baselines are not applicable for video datasets.")

    model_name = args["model_type"]

    timestamp = datetime.now().strftime("%y%m%dT%H%M%S")

    # fix random seed
    seed = args["seed"]
    model_utils.fix_seeds(seed)

    # get validation dataset and create the DataLoader
    train_dataset, test_dataset = datasets.CPDDatasets(experiments_name=experiments_name).get_dataset_()
    val_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # only 1 model in the list for classic baselines
    baseline_model = model_utils.get_models_list(args, train_dataset, test_dataset)[-1]

    # calculate metrics
    metrics_local = metrics.calculate_baseline_metrics(baseline_model, val_dataloader, verbose=args["verbose"])

    path_to_saves = Path("saves/results") / experiments_name
    path_to_metric = path_to_saves / "metrics"

    model_name_for_save = f'{model_name}_pen_{args["pen"]}_n_pred_{args["n_pred"]}_kernel_{args["kernel"]}.txt'

    path_to_metric.mkdir(parents=True, exist_ok=True)
    
    metrics.write_metrics_to_file(
        filename=f'{str(path_to_metric)}/'+ model_name_for_save,
        metrics=metrics_local,
        seed=seed,
        timestamp=timestamp
    )  

if __name__ == '__main__':
    parser = get_classic_baseline_parser()
    args = parser.parse_args()
    args = dict(vars(args)) 
    main(args)
