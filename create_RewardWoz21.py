import random
import sys
import os
import logging
import time
import numpy as np
import torch
import random
import json
import pickle
from torch import multiprocessing as mp
from convlab.policy.cql.CQL import CQL
from convlab.policy.rlmodule import Memory
from convlab.util.custom_util import set_seed, init_logging, save_config, move_finished_training, env_config, \
    eval_policy, log_start_args, save_best, load_config_file, get_config, flatten_acts
import pickle
import torch.utils.data as data
from copy import deepcopy
from convlab.policy.vector.vector_binary import VectorBinary
from convlab.util import load_policy_data, load_dataset, relative_import_module_from_unified_datasets, load_unified_data
from convlab.policy.rule.multiwoz.policy_agenda_multiwoz import Goal
from convlab.policy.rule.multiwoz.policy_agenda_multiwoz import act_dict_to_flat_tuple, unified_format
reverse_da, normalize_domain_slot_value = relative_import_module_from_unified_datasets(
    'multiwoz21', 'preprocess.py', ['reverse_da', 'normalize_domain_slot_value'])
from transform_goal import create_data_goal
from convlab.dst.rule.multiwoz.dst import RuleDST
from convlab.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab.util.multiwoz.dbquery import Database
from convlab.policy.rlmodule import MemoryReplay
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEVICE

try:
    mp.set_start_method('spawn', force=True)
    mp = mp.get_context('spawn')
except RuntimeError:
    pass
mapping = {'Restaurant': {'Addr': 'address', 'Area': 'area', 'Food': 'food', 'Name': 'name', 'Phone': 'phone',
                          'Post': 'postcode', 'Price': 'price range', 'Ref': 'ref', 'Day': 'book day', 'People': 'book people', 'Time': 'book time'},
           'Hotel': {'Addr': 'address', 'Area': 'area', 'Internet': 'internet', 'Parking': 'parking', 'Name': 'name',
                     'Phone': 'phone', 'Post': 'postcode', 'Price': 'price range', 'Stars': 'stars', 'Type': 'type', 'Ref': 'ref',
                     'Day': 'book day', 'People': 'book people', 'Stay':'book stay','none':None},
           'Attraction': {'Addr': 'address', 'Area': 'area', 'Fee': 'entrance fee', 'Name': 'name', 'Phone': 'phone',
                          'Post': 'postcode', 'Type': 'type'},
           'Train': {'Id': 'trainID', 'Arrive': 'arrive by', 'Day': 'day', 'Depart': 'departure', 'Dest': 'destination',
                     'Time': 'duration', 'Leave': 'leave at', 'Ticket': 'price', 'Ref': 'ref', 'People': 'book people'},
           'Taxi': {'Car': 'car type', 'Phone': 'phone', 'Arrive': 'arrive by', 'Depart': 'departure', 'Dest': 'destination', 'Leave': 'leave at'},
           'Hospital': {'Post': 'postcode', 'Phone': 'phone', 'Addr': 'address', 'Department': 'department'},
           'Police': {'Post': 'postcode', 'Phone': 'phone', 'Addr': 'address','Name': 'name'}}

def _replace(slot):
    #replace da_act name with the suitable name for searching in database.
    slot = slot.replace("price range", "pricerange").replace("leave at","leaveAt").replace("arrive by","arriveBy")
    return slot

dist_file = 'newdata.pkl'

def resetmemory(dist_file):
    # Overwrite the pickle file with an empty list
    with open(dist_file, 'wb') as pickle_file:
        pickle.dump([], pickle_file)
    print("Pickle file cleared.")
    '''
    file_size = os.path.getsize(dist_file)
    print(f"memory_size::{file_size}")
    if file_size:
        with open(dist_file, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        print(f"data::{data}")
    '''

# Write the lists to a pickle file
def write_to_memory_file(dist_file, new_data):
    try:
        # Check if the file exists and is not empty
        print(f'the size of dist_file is::{os.path.getsize(dist_file)}')
        if os.path.exists(dist_file) and os.path.getsize(dist_file) > 0:
            # Read the existing data from the file
            with open(dist_file, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
            print("*****************")        
            # Append the new data
            data.append(item for item in buff)
            '''
            if isinstance(data, list):
                data.append(new_data)
            elif isinstance(data, dict):
                data.update(new_data)
            else:
                raise ValueError("Unsupported pickle format")
            '''
            # Write the updated data back to the file
            with open(dist_file, 'wb') as pickle_file:
                pickle.dump(data, pickle_file)
        
            print("Data appended successfully.")

        elif os.path.exists(dist_file) and os.path.getsize(dist_file) == 0:
            with open(dist_file, 'wb') as pickle_file:
                pickle.dump([new_data], pickle_file)
                print("New file created and data added.")

    except FileNotFoundError:
        print("File not found. Creating a new file.")
        with open(dist_file, 'wb') as pickle_file:
            pickle.dump([new_data], pickle_file)
        print("New file created and data added.")
    
    except pickle.PickleError:
        print("Error decoding pickle. Ensure the file contains valid pickle.")
    
    except Exception as e:
        print(f"An error occurred: {e}")


def read_memory_file(file):
    try:
        file_size = os.path.getsize(file)
        # Load Memory object from a pickle file
        #print(f"memory_size::{file_size}")
        # Ensure the file is not empty
        if file_size == 0:
            raise EOFError("File is empty")
        # Load Memory object from a pickle file
        with open(file, 'rb') as file:
            file.seek(0)
            data = pickle.load(file)
            print(f'len(data[0])::{len(data[0])}')
        return data

    except FileNotFoundError:
        print("File not found.")
    
    except EOFError as e:
        print(f"EOFError: {e}")
    
    except pickle.UnpicklingError:
        print("Error unpickling the data. Ensure the file contains valid pickle data.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    
    #resetmemory(dist_file)
    '''
    #memory = MemoryReplay(max_size=10000)
    memory = read_memory_file(dist_file)
    if memory:
        print("Memory object loaded successfully.")
    else:
        print("Failed to load Memory object.")
    '''
    dataset_name = 'multiwoz21'
    dataset = load_dataset(dataset_name)
    loaded_data = load_unified_data(dataset, data_split="all", speaker='all', 
                                     utterance=True, dialogue_acts=True, state=True, terminated=True, db_results=True, goal=True, split_to_turn= False, use_context= True, context_window_size=1 )	
    print(f"the lengh of traning data is:::{len(loaded_data['train'])}")
    #buff = Memory()
    memory = MemoryReplay(max_size=70000)
    vector = VectorBinary()
    sys_dst = RuleDST()
    db = Database()
    dialog = 0
    #for dialogue in loaded_data['train']:
    for dialogue in loaded_data['train'][:1]:
        print(f"dialogue number is ::: {dialog}")
        #__init__
        evaluator = MultiWozEvaluator()
        sys_dst.init_session()
        buff = Memory()
        # Transform the goal
        goal = create_data_goal(dialogue)
        goal = goal.transform_goal(dialogue['goal'])
        evaluator.add_goal(goal)
        done = False
        user_dialog_act = reverse_da(dialogue['turns'][0]['dialogue_acts'])

        user_dialog_act = act_dict_to_flat_tuple(user_dialog_act)
        user_dialog_act = [[intent.lower(), domain.lower(), mapping[domain][slot], value.lower()] for intent, domain, slot, value in  user_dialog_act if slot!='none']
        evaluator.add_usr_da(user_dialog_act)
        sys_dst.state['user_action'] = user_dialog_act
        sys_dst.state['history'].append(["sys", []])
        sys_dst.state['history'].append(["user", user_dialog_act])

        state = sys_dst.update(user_dialog_act)
        

        s = deepcopy(state)
        #s = sys_dst.state
        for t in range(1,len(dialogue['turns']),2):

            s_vec, action_mask = vector.state_vectorize(s)
            s_vec = torch.Tensor(s_vec)
            action_mask = torch.Tensor(action_mask)
            #print("s::",s)            
            #print("s_vec.numpy()::",s_vec.numpy())

            sys_dialog_act = reverse_da(dialogue['turns'][t]['dialogue_acts'])
            sys_dialog_act = act_dict_to_flat_tuple(sys_dialog_act)
            sys_dialog_act = [[i.lower(), d.lower(), s.lower(), v.lower()] for i, d, s, v in sys_dialog_act]
            assert isinstance(sys_dialog_act, list)
            sys_dst.state['system_action'] = sys_dialog_act


            new_acts = list()
            if type(sys_dialog_act) == list:
                for intent, domain, slot, value in sys_dialog_act:
                    if intent == 'book' and value=='none' and (domain == 'restaurant' or domain == 'hotel' or domain =='train'):
                        constraints = [[_replace(key), val] for key, val in state['belief_state'][domain].items() if val != '' and 'book' not in key ]
                        found = db.query(domain=domain, constraints=constraints)
                        ref = found[0]['Ref'] if found else ''
                        value = ref
                    new_acts.append([intent, domain, slot, value])
            sys_dialog_act = new_acts     
            evaluator.add_sys_da(sys_dialog_act, sys_dst.state['belief_state'])
            if t<len(dialogue['turns'])-1:
                user_dialog_act = reverse_da(dialogue['turns'][t+1]['dialogue_acts'])
                user_dialog_act = act_dict_to_flat_tuple(user_dialog_act)
                user_dialog_act = [[intent.lower(), domain.lower(), mapping[domain][slot], value.lower()] for intent, domain, slot, value in  user_dialog_act if
                                      intent.lower() in ['inform', 'recommend', 'offerbook', 'offerbooked'] and slot!='none']
            else:
                user_dialog_act = []

            evaluator.add_usr_da(user_dialog_act)

            sys_dst.state['user_action'] = user_dialog_act
            sys_dst.state['history'].append(["sys", sys_dialog_act])
            sys_dst.state['history'].append(["user", user_dialog_act])

            state = sys_dst.update(user_dialog_act)
            s = deepcopy(state)

            if t == (len(dialogue['turns'])-1 ):
                done = True

            terminated = done
            reward = evaluator.get_reward(terminated)
            #print( f" dialogue {dialog}, turn {t} :reward {reward}, terminated {terminated}")


            next_s_vec, next_action_mask = vector.state_vectorize(s)
            #print("next_s_vec::",next_s_vec)
            # a flag indicates ending or not
            mask = 0 if done else 1
            #a=s_vec.numpy()
            #b=vector.action_vectorize(sys_dialog_act)
            #c= reward
            #d=next_s_vec.numpy()
            #e=mask
            #f=action_mask.numpy()
            #j=next_action_mask.numpy()
            buff.push(s_vec.numpy(), vector.action_vectorize(sys_dialog_act), reward, next_s_vec, mask, action_mask.numpy(), next_action_mask)
            #transaction = (s_vec.numpy(), vector.action_vectorize(sys_dialog_act), reward, next_s_vec, mask, action_mask.numpy(), next_action_mask)
            #write_to_memory_file(dist_file, buff)
        dialog +=1
        print("========================================================================================")
        print(f'buffer::{buff}')
        memory.append(buff)
    write_to_memory_file(dist_file, memory)
        
    memory = read_memory_file(dist_file)
    if memory:
        print("Memory object loaded successfully.")
    else:
        print("Failed to load Memory object.")
    