import copy
import itertools
import random
import tensorflow as tf
import numpy as np
import math


def set_seeds(seed):
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_configs(base_config, combinations):
    product_input = [p['values'] for p in combinations.values()]
    product = [p for p in itertools.product(*product_input)]
    configs = []
    for p in product: # for each combination
        config = copy.deepcopy(base_config)
        for i, parameter in enumerate(combinations.values()): # for each parameter in config
            for name in parameter['dict_path'][:-1]: # finish when pointing at second-last element from path
                pointer = config[name]
            pointer[parameter['dict_path'][-1]] = p[i] # set desired value
        configs.append(config)
    return configs

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.4
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,  
            math.floor((1+epoch)/epochs_drop))
    
    if (lrate < 1e-5):
        lrate = 1e-5
      
    print('Changing learning rate to {}'.format(lrate))
    return lrate