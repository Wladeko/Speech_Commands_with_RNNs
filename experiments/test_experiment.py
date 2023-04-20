import os
import wandb
import tensorflow as tf
import sys

from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

sys.path.append("./")
from utils.dataset import make_train_dataset, make_val_dataset
from utils.utils import set_seeds, make_configs, step_decay
from models.test_model import TestGRU


ENTITY = 'wladeko'
PROJECT = 'dl_rnn'
GROUP = 'test'
NAME = 'rnn'
SAVE_PATH = 'weights/'

base_config = {
    'dataloader': {
        'sample_rate': 8000,
        'unknown_silence_samples': 2000,
        'seed': 0,
        'batch_size': 128,
        'convert_to_image': False,
    },
    'training': {
        'n_epochs': 50,
        'dropout': 0.3,
    },
    'compile':{
        'loss': 'sparse_categorical_crossentropy',
        'optimizer': 'adam',
        'metrics': ['accuracy', 'sparse_categorical_accuracy']
    },
    'model': {
        'architecture': 'TestGRU',
        'id': None,
        'save_path': None,
    },
    'early_stopper':{
        'monitor': 'val_sparse_categorical_accuracy',
        'min_delta': 0.001,
        'patience': 4,
        'verbose': 1,
        'start_from_epoch': 10,
        'restore_best_weights': True,
    },
    'checkpointer':{
        'monitor': 'val_sparse_categorical_accuracy',
        "verbose": 1,
        'save_best_only': True
    },
    'scheduler': LearningRateScheduler(step_decay),
    'other':{
            'num_classes':12,
    }
}

combinations = {
    'seeds': {
        'dict_path': ['dataloader', 'seed'],
        'values': [0, 1, 2, 3, 4]
    },
}



configs = make_configs(base_config, combinations)

for i, config in enumerate(configs):

    set_seeds(config['dataloader']['seed'])
    config['model']['id'] = i
    NAME = config['model']['architecture'] + str(config['model']['id'])

    wandb.init(
        project = PROJECT,
        entity = ENTITY,
        group = GROUP,
        name = NAME,
        config = config)
    
    l = len(configs)
    print(f"---------------\nConfig {i+1}/{l}\n---------------\n\n")
    print('Running config:', config, "\n")

    X_t, y_t = make_train_dataset(**base_config['dataloader'])
    print('--- Training data loaded ---\n')
    X_v, y_v = make_val_dataset(**base_config['dataloader'])
    print('--- Validation data loaded ---\n')

    input_shape = X_t.shape[1:]
    # input_shape = (None, 8000)

    model = TestGRU(input_shape=input_shape, output_nodes=config['other']['num_classes'], dropout=config["training"]['dropout'])

    model.compile(**config["compile"])
    earlystopper = EarlyStopping(**config["early_stopper"])
    checkpointer = ModelCheckpoint(NAME+'.h5', **config["checkpointer"])
    lrate = config["scheduler"]

    history = model.fit(
                X_t,
                y_t,
                epochs=config['training']['n_epochs'],
                validation_data=(X_v, y_v),
                batch_size=config['dataloader']['batch_size'],
                shuffle=True,
                callbacks=[
                    earlystopper, 
                    checkpointer, 
                    lrate,
                    WandbMetricsLogger(log_freq=5),
                    WandbModelCheckpoint("models")
                ])
    save_path = os.path.join(SAVE_PATH, NAME)
    model.save(save_path)

    wandb.finish()