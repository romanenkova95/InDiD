"""Module with utility functions for models initialization."""
from typing import Dict, Optional, Sequence, Tuple, Union, List
import torch
from torch.utils.data import Dataset
import yaml
import pytorch_lightning as pl

from . import core_models, cpd_models
from . import klcpd, tscp
from .cpd_models import fix_seeds

def get_models_list(
    args: dict,
    train_dataset: Dataset,
    test_dataset: Dataset,
) -> List[pl.LightningModule]:
    """Initialize CPD models for a particular experiment.

    :param args: dict with all the parameters
    :param train_dataset: training data
    :param test_dataset: testing data
    :return models_list: list with 2 models in case of a 'seq2seq' model with the combined loss,
                         list with 1 model, otherwise
    """
    if args["model_type"] == "seq2seq":
        models_list = get_seq2seq_models_list(args, train_dataset, test_dataset)

    elif args["model_type"] == "kl_cpd":
        models_list = get_kl_cpd_models_list(args, train_dataset, test_dataset)

    elif args["model_type"] == "tscp":
        models_list = get_tscp_models_list(args, train_dataset, test_dataset)

    else:
        raise ValueError(f'Unknown model {args["model"]}.')

    return models_list

def get_seq2seq_models_list(
    args: dict,
    train_dataset: Dataset,
    test_dataset: Dataset
) -> List[cpd_models.CPDModel]:
    """Initialize seq2seq models for a particular experiment.

    :param args: dict with all the parameters
    :param train_dataset: training data
    :param test_dataset: testing data
    :returns: list with 2 CPD models in case of 'combined loss', 
              list with 1 CPD model in case of 'indid' or 'bce' loss
    """
    # Initialize core models for synthetic_1D, _100D and human_activity experiments
    if args["experiments_name"] in ["synthetic_1D", "synthetic_100D", "human_activity"]:
        # initialize default base model for Synthetic Normal 1D experiment
        core_model = core_models.BaseRnn(
            input_size=args["model"]["input_size"],
            hidden_dim=args["model"]["hidden_dim"],
            n_layers=args["model"]["n_layers"],
            drop_prob=args["model"]["drop_prob"]
            )
            
    elif args["experiments_name"] == "mnist":        
        # initialize default base model for MNIST experiment
        core_model = core_models.MnistRNN(
            input_size=args["model"]["input_size"],
            hidden_rnn=args["model"]["hidden_rnn"],
            rnn_n_layers=args["model"]["rnn_n_layers"],
            linear_dims=args["model"]["linear_dims"],
            rnn_dropout=args["model"]["rnn_dropout"],
            dropout=args["model"]["dropout"],
            rnn_type=args["model"]["rnn_type"]
        )
    
    elif args["experiments_name"] in ["explosion", "road_accidents"]:
        core_model = core_models.CombinedVideoRNN(
            input_dim=args["model"]["input_size"],
            rnn_hidden_dim=args["model"]["hidden_rnn"],
            num_layers=args["model"]["rnn_n_layers"],
            rnn_dropout=args["model"]["rnn_dropout"],
            dropout=args["model"]["dropout"]
        )

    else:
        raise ValueError("Wrong experiment name.")

    # Initialize CPD models
    if args["loss_type"] in ["indid", "bce"]:
        model = cpd_models.CPDModel(
            loss_type = args["loss_type"],
            args=args,
            model=core_model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            )
        models_list = [model]

    elif args["loss_type"] == "combined":
        model_1 = cpd_models.CPDModel(
            loss_type="bce",
            args=args,
            model=core_model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            )
        model_2 = cpd_models.CPDModel(
            loss_type="indid",
            args=args,
            model=core_model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            )
        models_list = [model_1, model_2]
    else:
        raise ValueError("Wrong loss type. Choose 'indid', 'bce' or 'combined'.'")
    return models_list

def get_kl_cpd_models_list(
    args: dict,
    train_dataset: Dataset,
    test_dataset: Dataset
) -> List[klcpd.KLCPD]:
    """Initialize KL-CPD models for a particular experiment.

    :param args: dict with all the parameters
    :param train_dataset: training data
    :param test_dataset: testing data
    :returns: list with 1 KL-CPD model
    """
    # the same model structures for all the experiments, args are read from different configs
    if args["experiments_name"] in ["explosion", "road_accidents"]:
        netD = klcpd.VideoNetD(args)
        netG = klcpd.VideoNetG(args)

    elif args["experiments_name"] in ["synthetic_1D", "synthetic_100D", "mnist", "human_activity"]:
        netD = klcpd.NetD(args)
        netG = klcpd.NetG(args)
    else:
        raise ValueError("Wrong experiments name.")
    
    model = klcpd.KLCPD(
        args=args,
        net_generator=netG,
        net_discriminator=netD,
        train_dataset=train_dataset,
        test_dataset=test_dataset
        )
    models_list = [model]
    print(model)
    return models_list

def get_tscp_models_list(
    args: dict,
    train_dataset: Dataset,
    test_dataset: Dataset
    )->  List[tscp.TSCP_model]:
    """Initialize TS-CP2 models for a particular experiment.

    :param args: dict with all the parameters
    :param train_dataset: training data
    :param test_dataset: testing data
    :returns: list with 1 TS-CP2 model
    """
    # universal encoder for all the experiments
    encoder = tscp.BaseTSCPEncoder(args)
        
    model = tscp.TSCP_model(
        args=args,
        model=encoder,
        train_dataset=train_dataset, 
        test_dataset=test_dataset
    )

    models_list = [model]
    return models_list