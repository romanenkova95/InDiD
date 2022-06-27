"""Module for trainig Change Point Detection models."""

import pytorch_lightning as pl
import torch.nn as nn

from typing import Tuple

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from CPD_explosion import models

from CPD_explosion.models import fix_seeds

def train_model(
    model: models.CPD_model,
    experiments_name: str,
    max_epochs: int = 100,
    patience: int = None,
    gpus: int = 0,
    gradient_clip_val: float = 0.0,
    seed: int = 0,
    monitor: str = "val_loss",
    min_delta: float = 0.0,
    check_val_every_n_epoch: int = 1
) -> models.CPD_model:
    
    """Initialize logger, callbacks, trainer and TRAIN CPD model
    
    :param model: CPD_model for training
    :param experiments_name: name of the conducted experiment
    :param max_epochs: maximum # of epochs to train (default=100)
    :param patience: # of epochs to eait before early stopping (no early stopping if patience=None)
    :param gpus: # of available GPUs for trainer (default=0 -> train on CPU)
    :param gradient_clip_val: parameter for Gradient Clipping (if 0, no Gradient Clipping)
    
    :return: trained model
    """    
    callbacks = []
    
    # initialize TensorBoard logger
    logger = TensorBoardLogger(save_dir='logs/', name=experiments_name)
    
    # initialize Checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/{}'.format(experiments_name),
                                          filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}', )
    callbacks.append(checkpoint_callback)
    
    if patience is not None:
        # initialize EarlyStopping callback
        early_stop_callback = EarlyStopping(monitor=monitor, min_delta=min_delta,
                                            patience=patience, verbose=True, mode="min")
        callbacks.append(early_stop_callback)

    # fixing all the seeds
    fix_seeds(seed)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=gpus,
        benchmark=True,
        check_val_every_n_epoch=check_val_every_n_epoch,
        gradient_clip_val=gradient_clip_val,
        logger=logger,
        callbacks=callbacks
    )
    
    trainer.fit(model)
       
    return model
    
def run_experiments(
    experiments_name: str,
    max_epochs: int = 100,
    patience: int = None,
    gpus: int = 0,
    gradient_clip_val: float = 0.0,
    ) -> Tuple[models.CPD_model, models.CPD_model, models.CPD_model]:
    
    """Run default experiments.
    
    :param experiments_name: name of the conducted experiment (available now: synthetic_1D,
                                                                              synthetic_100D,
                                                                              mnist,
                                                                              human_activity,
                                                                              explosion,
                                                                              car_accidents)
    :param max_epochs: maximum # of epochs to train (default=100)
    :param patience: # of epochs to eait before early stopping (no early stopping if patience=None)
    :param gpus: # of available GPUs for trainer (default=0 -> train on CPU)
    :param gradient_clip_val: parameter for Gradient Clipping (if 0, no Gradient Clipping)
    :return: tuple of
            - trained CPD_seq2seq model
            - trained BCE_seq2seq model
            - trained InDiD (aka combined) model
    """
    print('Runing {} experiment.'.format(experiments_name))
    print('Training CPD_seq2seq model...')

    cpd_model = models.CPD_model(experiments_name=experiments_name, loss_type='CPD')
    cpd_model = train_model(model=cpd_model, experiments_name=experiments_name, max_epochs=max_epochs,
                            patience=patience, gpus=gpus, gradient_clip_val=gradient_clip_val)
    
    print('CPD_seq2seq model is trained!')
    print('Training BCE_seq2seq model...')
    
    bce_model = models.CPD_model(experiments_name=experiments_name, loss_type='BCE')
    bce_model = train_model(model=bce_model, experiments_name=experiments_name, max_epochs=max_epochs,
                            patience=patience, gpus=gpus, gradient_clip_val=gradient_clip_val)
    
    print('BCE_seq2seq model is trained!')
    print('Training InDiD model...')
    
    indid_model_1 = models.CPD_model(experiments_name=experiments_name, loss_type='BCE')
    indid_model_2 = models.CPD_model(experiments_name=experiments_name, loss_type='CPD', model=indid_model_1.model)
    
    indid_model_1 = train_model(model=indid_model_1, experiments_name=experiments_name, max_epochs=max_epochs//2,
                                patience=patience, gpus=gpus, gradient_clip_val=gradient_clip_val)
    indid_model_2 = train_model(model=indid_model_2, experiments_name=experiments_name, max_epochs=max_epochs//2,
                                patience=patience, gpus=gpus, gradient_clip_val=gradient_clip_val)

    print('InDiD model is trained!')

    return cpd_model, bce_model, indid_model_2

def write_metrics_to_file(filename, metrics, seed):
    best_th_f1, best_time_to_FA, best_delay, auc, best_conf_matrix, best_f1, best_cover, best_th_cover, max_cover = metrics
    
    with open(filename, 'a') as f:
        f.writelines('SEED: {}\n'.format(seed))
        f.writelines('AUC: {}\n'.format(auc))
        f.writelines('Time to FA {}, delay detection {} for best-F1 threshold: {}\n'. format(round(best_time_to_FA, 4), 
                                                                                       round(best_delay, 4), 
                                                                                       round(best_th_f1, 4)))
        f.writelines('TN {}, FP {}, FN {}, TP {} for best-F1 threshold: {}\n'. format(best_conf_matrix[0],
                                                                               best_conf_matrix[1],
                                                                               best_conf_matrix[2],
                                                                               best_conf_matrix[3],
                                                                               round(best_th_f1, 4)))
        f.writelines('Max F1 {}: for best-F1 threshold {}\n'.format(round(best_f1, 4), round(best_th_f1, 4)))
        f.writelines('COVER {}: for best-F1 threshold {}\n'.format(round(best_cover, 4), round(best_th_f1, 4)))

        f.writelines('Max COVER {}: for threshold {}\n'.format(max_cover, best_th_cover))
        f.writelines('----------------------------------------------------------------------\n')
