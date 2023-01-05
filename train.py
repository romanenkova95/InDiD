"""Script that initializes, trains and saves a CPD model for a particular experiment."""
from utils import datasets, model_utils
from utils.arg_parsing import get_train_parser, get_args

import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pathlib import Path
from datetime import datetime

import torch

def main(args: dict) -> str:
    experiments_name = args["experiments_name"]
    model_name = args["model_type"]

    if model_name == "seq2seq":
        model_name = model_name + "_" + args["loss_type"]

    timestamp = ""
    if not args["dryrun"]:
        timestamp = datetime.now().strftime("%y%m%dT%H%M%S")
        save_path = Path("saves/models") / experiments_name
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f'model_{model_name}_{timestamp}.pth'
        assert not save_path.exists(), f'Checkpoint {str(save_path)} already exists'

    train_dataset, test_dataset = datasets.CPDDatasets(experiments_name=experiments_name).get_dataset_()

    seed = args["seed"]
    model_utils.fix_seeds(seed)

    # 2 models in the list in case of "combined" loss, otherwise, 1 model in the list
    models_list = model_utils.get_models_list(args, train_dataset, test_dataset)

    logger = TensorBoardLogger(save_dir=f'logs/{experiments_name}',
                               name=model_name)

    for model in models_list:
        trainer = pl.Trainer(
            max_epochs=args["learning"]["epochs"]//len(models_list),
            gpus=args["gpus"],
            benchmark=True,
            check_val_every_n_epoch=1,
            gradient_clip_val=args["learning"]['grad_clip'],
            logger=logger,
            callbacks=EarlyStopping(**args["early_stopping"])
        )
        trainer.fit(model)

    # Note that we save the last trained model - OK
    if not args["dryrun"]:
        esp = args.pop("early_stopping", {})
        esp = {f'early_stopping_{key}': value for key, value in esp.items()}
        args.update(esp)
        torch.save({"checkpoint": model.state_dict(), "args": args}, save_path)
    
    return timestamp

if __name__ == '__main__':
    parser = get_train_parser()
    args = get_args(parser)
    timestamp = main(args)
    print(timestamp)