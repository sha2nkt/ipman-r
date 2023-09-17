import os
import time
import shutil
import yaml
import itertools
# from flatten_dict import flatten, unflatten
from typing import Dict, List, Union, Any
from loguru import logger
import operator
from functools import reduce
from rich_cluster import execute_task_on_cluster
from yacs.config import CfgNode as CN
SMPL_MODEL_DIR = '/ps/project/common/smplifyx/models/'

def get_hparams_defaults():
    """Get a yacs hparamsNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return hparams.clone()

def update_hparams_from_dict(cfg_dict):
    hparams = get_hparams_defaults()
    cfg = hparams.load_cfg(str(cfg_dict))
    hparams.merge_from_other_cfg(cfg)
    return hparams.clone()

def get_grid_search_configs(config, excluded_keys=[]):
    """
    :param config: dictionary with the configurations
    :return: The different configurations
    """

    def bool_to_string(x: Union[List[bool], bool]) -> Union[List[str], str]:
        """
        boolean to string conversion
        :param x: list or bool to be converted
        :return: string converted thinghat
        """
        if isinstance(x, bool):
            return [str(x)]
        for i, j in enumerate(x):
            x[i] = str(j)
        return x

    # exclude from grid search

    flattened_config_dict = flatten(config, reducer='path')
    hyper_params = []
    for k,v in flattened_config_dict.items():
        if isinstance(v,list):
            if k in excluded_keys:
                flattened_config_dict[k] = ['+'.join(v)]
            elif len(v) > 1:
                hyper_params += [k]

        if isinstance(v, list) and isinstance(v[0], bool) :
            flattened_config_dict[k] = bool_to_string(v)

        if not isinstance(v,list):
            if isinstance(v, bool):
                flattened_config_dict[k] = bool_to_string(v)
            else:
                flattened_config_dict[k] = [v]

    keys, values = zip(*flattened_config_dict.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for exp_id, exp in enumerate(experiments):
        for param in excluded_keys:
            exp[param] = exp[param].strip().split('+')
        for param_name, param_value in exp.items():
            # print(param_name,type(param_value))
            if isinstance(param_value, list) and (param_value[0] in ['True', 'False']):
                exp[param_name] = [True if x == 'True' else False for x in param_value]
            if param_value in ['True', 'False']:
                if param_value == 'True':
                    exp[param_name] = True
                else:
                    exp[param_name] = False


        experiments[exp_id] = unflatten(exp, splitter='path')

    return experiments, hyper_params

def run_grid_search_experiments(
        args,
    script='main.py'):
    # cfg = yaml.safe_load(open(cfg_file))
    # parse config file to get a list of configs and the name of hyperparameters that change
    # different_configs, hyperparams = get_grid_search_configs(
    #     cfg,
    #     excluded_keys=['TRAIN/DATASETS_2D', 'TRAIN/DATASETS_3D', 'TRAIN/DATASET_EVAL'],
    # )
    # logger.info(f'Grid search hparams: \n {hyperparams}')

    # different_configs = [update_hparams_from_dict(c) for c in different_configs]
    # logger.info(f'======> Number of experiment configurations is {len(different_configs)}')


    config_to_run = CN(args)

    if config_to_run.cluster: # if running on cluster
        execute_task_on_cluster(
            script=script,
            exp_name=config_to_run.exp_name,
            input_folder=config_to_run.input_folder,
            cluster_bs=config_to_run.cluster_bs,
            num_queue=config_to_run.num_queue,
            condor_dir=config_to_run.condor_dir,
            cfg_file=config_to_run.config,
            bid_amount=config_to_run.bid,
            num_workers=4,
            memory=config_to_run.memory,
            exp_opts=config_to_run.opts,
            gpu_min_mem=config_to_run.gpu_min_mem,
            gpu_arch=config_to_run.gpu_arch,
        )
        exit()

    return config_to_run