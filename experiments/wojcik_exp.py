import os
import sys

import numpy as np
import tensorflow as tf
import wandb
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from numpy import load
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

sys.path.append("./")
import absl.logging

from models.CNN_LSTM import CNN_LSTM1
from models.GRU import GRU1, GRU2, GRU3
from models.LSTM import LSTM1, LSTM2, LSTM3
from models.simpleRNN import simpleRNN1, simpleRNN2, simpleRNN3
from utils.utils import make_configs, set_seeds, step_decay

absl.logging.set_verbosity(absl.logging.ERROR)

ENTITY = "wo-team"
PROJECT = "dl-rnn-audio"
GROUP = "simple-lstm-gru"
NAME = "_"
SAVE_PATH = "weights/"

models = {
    "SimpleRNN1": simpleRNN1,
    "SimpleRNN2": simpleRNN2,
    "SimpleRNN3": simpleRNN3,
    "LSTM1": LSTM1,
    "LSTM2": LSTM2,
    "LSTM3": LSTM3,
    "GRU1": GRU1,
    "GRU2": GRU2,
    "GRU3": GRU3,
}

base_config = {
    "dataloader": {
        "sample_rate": 8000,
        "unknown_silence_samples": 2000,
        "seed": 0,
        "batch_size": 128,
        "convert_to_image": False,
    },
    "training": {
        "n_epochs": 50,
        "dropout": 0.3,
    },
    "compile": {
        "loss": "sparse_categorical_crossentropy",
        "optimizer": "adam",
        "metrics": ["accuracy", "sparse_categorical_accuracy"],
    },
    "model": {
        "architecture": "SimpleRNN1",
        "model_init": None,
        "id": None,
        "save_path": None,
    },
    "early_stopper": {
        "monitor": "val_sparse_categorical_accuracy",
        "min_delta": 0.001,
        "patience": 4,
        "verbose": 1,
        "start_from_epoch": 10,
        "restore_best_weights": True,
    },
    "checkpointer": {
        "monitor": "val_sparse_categorical_accuracy",
        "verbose": 1,
        "save_best_only": True,
    },
    "scheduler": LearningRateScheduler(step_decay),
    "other": {
        "num_classes": 12,
    },
}

combinations = {
    "seeds": {"dict_path": ["dataloader", "seed"], "values": [0, 1, 2]},
    "architecture": {
        "dict_path": ["model", "architecture"],
        "values": [
            "SimpleRNN1",
            "SimpleRNN2",
            "SimpleRNN3",
            "LSTM1",
            "LSTM2",
            "LSTM3",
            "GRU1",
            "GRU2",
            "GRU3",
        ],
    },
    "dropout": {"dict_path": ["training", "dropout"], "values": [0.3, 0.5]},
}

configs = make_configs(base_config, combinations)

LOAD_PTH = "./saved_data/"
X_t = load(LOAD_PTH + "X_t.npy")
y_t = load(LOAD_PTH + "y_t.npy")
X_v = load(LOAD_PTH + "X_v.npy")
y_v = load(LOAD_PTH + "y_v.npy")
y_t = np.argmax(y_t, axis=1).transpose()
y_v = np.argmax(y_v, axis=1).transpose()

start_config = int(input("Provide ID of first config: "))

for i, config in enumerate(configs):
    if i < start_config:
        continue

    set_seeds(config["dataloader"]["seed"])
    config["model"]["id"] = i
    NAME = (
        str(config["model"]["id"])
        + config["model"]["architecture"]
        + "_seed"
        + str(config["dataloader"]["seed"])
        + "_dropout"
        + str(config["training"]["dropout"]).replace(".", ",")
    )
    config["model"]["model_init"] = models[config["model"]["architecture"]]

    wandb.init(project=PROJECT, entity=ENTITY, group=GROUP, name=NAME, config=config)

    l = len(configs)
    print(f"---------------\nConfig {i+1}/{l}\n---------------\n\n")
    print("Running config:", config, "\n")

    input_shape = X_t.shape[1:]

    model = config["model"]["model_init"](
        input_shape=input_shape, dropout=config["training"]["dropout"]
    )

    model.compile(**config["compile"])
    earlystopper = EarlyStopping(**config["early_stopper"])
    checkpointer = ModelCheckpoint(NAME + ".h5", **config["checkpointer"])
    lrate = config["scheduler"]

    history = model.fit(
        X_t,
        y_t,
        epochs=config["training"]["n_epochs"],
        validation_data=(X_v, y_v),
        batch_size=config["dataloader"]["batch_size"],
        shuffle=True,
        callbacks=[
            earlystopper,
            checkpointer,
            lrate,
            WandbMetricsLogger(log_freq=5),
            WandbModelCheckpoint("weights/wandb"),
        ],
    )
    save_path = os.path.join(SAVE_PATH, NAME)
    model.save(save_path)

    wandb.finish()
