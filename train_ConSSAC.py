# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:14:07 2019
@author: truthless
"""

import sys
import os
import logging
import time
import numpy as np
import torch
import random

#from convlab.policy.ppo import PPO
from convlab.policy.cql.ConSSAC import ConSSAC
#from convlab.policy.ppo.multiwoz.ppo_policy import PPOPolicy
from convlab.policy.rlmodule import Memory
from torch import multiprocessing as mp
from argparse import ArgumentParser
from convlab.util.custom_util import set_seed, init_logging, save_config, move_finished_training, env_config, \
    eval_policy, log_start_args, save_best, load_config_file, get_config
from datetime import datetime
import matplotlib.pyplot as plt
from create_new_database import read_memory_file

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEVICE

try:
    mp.set_start_method('spawn', force=True)
    mp = mp.get_context('spawn')
except RuntimeError:
    pass

memory_file = 'newdata.pkl'

def update(env, policy, num_dialogues, epoch, process_num, seed=0):

    # sample data
    if epoch==1:
        transactions = read_memory_file(memory_file)
        policy.update_memory(transactions)
    policy.update(env, policy_sys, epoch)

if __name__ == '__main__':

    time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    begin_time = datetime.now()
    parser = ArgumentParser()
    parser.add_argument("--config_name", type=str, default='RuleUser-Semantic-RuleDST',
                        help="Name of the configuration")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the policy parameter initialization")
    parser.add_argument("--mode", type=str, default='info',
                        help="Set level for logger")
    parser.add_argument("--save_eval_dials", type=bool, default=False,
                        help="Flag for saving dialogue_info during evaluation")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs',
                        f'{parser.parse_args().config_name}.json')
    seed = parser.parse_args().seed
    mode = parser.parse_args().mode
    save_eval = parser.parse_args().save_eval_dials

    logger, tb_writer, current_time, save_path, config_save_path, dir_path, log_save_path = \
        init_logging(os.path.dirname(os.path.abspath(__file__)), mode)

    args = [('model', 'seed', seed)] if seed is not None else list()

    environment_config = load_config_file(path)
    save_config(vars(parser.parse_args()), environment_config, config_save_path)

    conf = get_config(path, args)
    seed = conf['model']['seed']
    logging.info('Train seed is ' + str(seed))
    set_seed(seed)

    
    #policy_sys = Discrete_SAC(True, seed=conf['model']['seed'], vectorizer=conf['vectorizer_sys_activated'],use_masking= True, load_path = "from_pretrained")
    policy_sys = CQL(True, seed=conf['model']['seed'], vectorizer=conf['vectorizer_sys_activated'],use_masking= True, load_path = "")    
   
    # Load model
    if conf['model']['use_pretrained_initialisation']:
        logging.info("Loading supervised model checkpoint.")
        policy_sys.load_from_pretrained(conf['model'].get('pretrained_load_path', ""),"","")
        #policy_sys.load_from_pretrained(conf['model']['pretrained_load_path'],"best_ppo","best_ppo")
        #policy_sys.load_from_pretrained(conf['model']['pretrained_load_path'],"best_SAC","best_SAC")
    elif conf['model']['load_path']:
        try:
            print(conf['model']['load_path'])
            policy_sys.load(conf['model']['load_path'])
        except Exception as e:
            logging.info(f"Could not load a policy: {e}")
    else:
        logging.info("Policy initialised from scratch")

    log_start_args(conf)
    logging.info(f"New episodes per epoch: {conf['model']['num_train_dialogues']}")

    env, sess = env_config(conf, policy_sys)


    policy_sys.current_time = current_time
    policy_sys.log_dir = config_save_path.replace('configs', 'logs')
    policy_sys.save_dir = save_path
    
    logging.info(f"Evaluating at start - {time_now}" + '-'*60)
    time_now = time.time()
    eval_dict = eval_policy(conf, policy_sys, env, sess, save_eval, log_save_path)
    logging.info(f"Finished evaluating, time spent: {time.time() - time_now}")

    for key in eval_dict:
        tb_writer.add_scalar(key, eval_dict[key], 0)
    best_complete_rate = eval_dict['complete_rate']
    best_success_rate = eval_dict['success_rate_strict']
    best_return = eval_dict['avg_return']
    '''
    best_complete_rate = {}
    best_success_rate = {}
    best_return = {}
    '''
    logging.info("Start of Training: " +
                 time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    #policy_sys.target_copy()

    success_rate_strict = []
    complete_rate=[]
    avg_return=[]
    avg_actions=[]
    y = 0
    for i in range(conf['model']['epoch']):
        idx = i + 1
        print("Epoch :{}".format(str(idx)))
        update(env, policy_sys, conf['model']['num_train_dialogues'], idx, conf['model']['process_num'], seed=seed)
        policy_sys.save(save_path, "model" + str(idx))
        #if (idx % conf['model']['eval_frequency'] == 0 and idx != 0) :
        if (True) :
            time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.info(f"Evaluating after Dialogues: {idx * (conf['model']['num_train_dialogues'] )} - {time_now}" + '-' * 60)

            eval_dict = eval_policy(conf, policy_sys, env, sess, save_eval, log_save_path)
            
            #print("eval_dict['success_rate_strict']",eval_dict["success_rate_strict"])
            
            complete_rate.append(eval_dict["complete_rate"])
            avg_return.append(eval_dict["avg_return"])
            success_rate_strict.append(eval_dict["success_rate_strict"])
            avg_actions.append(eval_dict["avg_actions"])
            y = y + 1 
             			
            best_complete_rate, best_success_rate, best_return = \
                save_best(policy_sys, best_complete_rate, best_success_rate, best_return,
                          eval_dict["complete_rate"], eval_dict["success_rate_strict"],
                          eval_dict["avg_return"], save_path)
            policy_sys.save(save_path, "last")
            for key in eval_dict:
                tb_writer.add_scalar(key, eval_dict[key], idx * conf['model']['num_train_dialogues'])

    logging.info("End of Training: " +
                 time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    #f = open(os.path.join(dir_path, "time.txt"), "a")
    with open(os.path.join(dir_path, "time.txt"), "a") as f:
        f.write(str(datetime.now() - begin_time))
        f.close()

  
