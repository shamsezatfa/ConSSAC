# -*- coding: utf-8 -*-
import torch
from torch import optim
import numpy as np
import logging
import os
import json
from convlab.policy.policy import Policy
from convlab.policy.rlmodule import MultiDiscretePolicy, Value, MemoryReplay
from convlab.util.custom_util import model_downloader, set_seed
import zipfile
import sys
import torch.nn.functional as F


root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CQL(Policy):

    def __init__(self, is_train=False, dataset='Multiwoz', seed=0, vectorizer=None, load_path="", **kwargs):

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs','cql_config.json'), 'r') as f:
            cfg = json.load(f)
        print(f'{os.path.join(os.path.dirname(os.path.abspath(__file__)))}')
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['save_dir'])
        self.cfg = cfg
        self.save_per_epoch = cfg['save_per_epoch']
        self.update_round = cfg['update_round']
        self.training_iter = cfg['training_iter']
        self.training_batch_iter = cfg['training_batch_iter']
        self.optim_batchsz = cfg['batchsz']
        self.tau = cfg['tau']
        self.is_train = is_train
        self.automatic_entropy_tuning = True
        self.discount_rate=cfg['discount_rate']
        self.info_dict = {}
        self.vector = vectorizer


        # CQL params
        self.with_lagrange = True
        self.cql_log_alpha = torch.tensor(2.0, requires_grad=True, device = DEVICE)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=cfg['cql_lr']) 
        logging.info('cql seed ' + str(seed))
        set_seed(seed)

        if self.vector is None:
            logging.info("No vectorizer was set, using default..")
            from convlab.policy.vector.vector_binary import VectorBinary
            self.vector = VectorBinary()

        # construct actor and critic networks
        if dataset == 'Multiwoz':
            self.actor = MultiDiscretePolicy(self.vector.state_dim, cfg['h_dim'], self.vector.da_dim, seed).to(device=DEVICE)
            logging.info(f"ACTION DIM OF CQL: {self.vector.da_dim}")
            logging.info(f"STATE DIM OF CQL: {self.vector.state_dim}")

        #replay memory
        self.memory = MemoryReplay(cfg['memory_size'])

        
        self.critic_local = Value(self.vector.state_dim, self.vector.da_dim, cfg['hv_dim']).to(device=DEVICE)
        self.critic_local_2 = Value(self.vector.state_dim, self.vector.da_dim, cfg['hv_dim']).to(device=DEVICE)
        with torch.no_grad():
            self.critic_target = Value(self.vector.state_dim, self.vector.da_dim, cfg['hv_dim']).to(device=DEVICE)
            self.critic_target_2 = Value(self.vector.state_dim, self.vector.da_dim, cfg['hv_dim']).to(device=DEVICE)
                      
        self.automatic_entropy_tuning=cfg["automatic_entropy_tuning"]
        if self.automatic_entropy_tuning:
            #self.target_entropy = .98*-np.log((1.0/self.vector.da_dim))
            self.target_entropy = 1
            self.log_alpha = torch.zeros(1, requires_grad=True, device = DEVICE)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=cfg['policy_lr'])
        
        if is_train:
            print(f"cfg['policy_lr']:{cfg['policy_lr']}. cfg['value_lr']:{cfg['value_lr']}.cfg['cql_lr']:{cfg['cql_lr']} ")
            self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg['policy_lr'])
            self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=cfg['value_lr'])
            self.critic_optim_2 = optim.Adam(self.critic_local_2.parameters(), lr=cfg['value_lr'])

        self.copy_model_over(self.critic_local, self.critic_target)
        self.copy_model_over(self.critic_local_2, self.critic_target_2)
    
    def update_memory(self, sample):
        #self.memory.reset()
        self.memory.append(sample)
        print(f"len(self.memory)::{len(self.memory)}")



    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        while (True):
            #print("state", state)
            s, action_mask = self.vector.state_vectorize(state)
            s_vec = torch.Tensor(s).to(device=DEVICE)
            mask_vec = torch.Tensor(action_mask).to(device=DEVICE)
            a = self.actor.select_action(s_vec.to(device=DEVICE), False, action_mask=mask_vec.to(device=DEVICE) ).cpu()

            a_counter = 0
            while a.sum() == 0:
                a_counter += 1
                a = self.actor.select_action(
                    s_vec.to(device=DEVICE), True, action_mask=mask_vec.to(device=DEVICE)).cpu()
                if a_counter == 5:
                    break

            action = self.vector.action_devectorize(a.detach().numpy())
            '''
            action = [[intent,domain,(slot.replace('Ticket',"Price") if (domain == 'Hotel' or domain == 'Restaurant') else slot) ,value] for intent, domain, slot, value in action ]  

            for act in action:
                intent, domain, slot, value = act
                if intent == 'inform':
                    if value == 'not available':
                        action.remove(act)

            inform_da = {}      
            inform2_da = {}		
            for act in action:
                intent, domain, slot, value = act
                if intent == 'inform':
                    if not inform_da.get(domain):
                        inform_da[domain]=[]
                        inform2_da[domain]=[]
                    inform_da[domain].append([slot,value])
                    inform2_da[domain].append(slot)

            #print("inform2_da",inform2_da)
            pred_action = {}
            dells=[]
            for act in action:
                intent, domain, slot, value = act
                if  intent == 'recommend' and slot != "name" and domain in inform_da.keys() and [slot,value] in inform_da.get(domain) : 
                #if  intent == 'recommend'  and domain in inform_da.keys() and [slot,value] in inform_da.get(domain) : 
                    #print("inform_da",inform_da)
                    #print("recom_da",recom_da)
                    #print("delll", act)
                    #action.remove(act)                
                    dells.append(act)
            for act in dells:
                if len(action)>1:
                    action.remove(act)
            
            #print("action%%",action)
            dells=[]
            for act in action:
                intent, domain, slot, value = act
                #print("act", act)
                #if  intent == 'request':
                    #print("request_act", act)
                if  intent == 'request'   and domain in inform2_da.keys() and slot in inform2_da.get(domain) : 
                    #print("dell", act)
                    #action.remove(act)                
                    dells.append(act)
            for act in dells:
                if len(action)>1:
                   action.remove(act)
        
            dells=[]
            for pred_act in pred_action:
                for act in action: 
                    if act == pred_act:
                        intent, domain, slot, value = act
                        if  intent == 'request':
                            #action.remove(act)                
                            dells.append(act)
    
            for act in dells:
                if len(action)>1:
                    action.remove(act)
             
            pred_action = action
	        '''
            self.info_dict["action_used"] = action
            return action
        

    def init_session(self):
        """
        Restore after one session
        """
        pass
    

    
    def update(self, policy_sys, epoch):
        
        total_critic1_loss = 0.
        total_critic2_loss = 0.
        total_actor_loss = 0.	
        total_alpha_loss = 0.

        for i in range(self.training_iter):
            
            batch = self.memory.get_batch(batch_size = self.optim_batchsz)

            s_b = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
            a_b = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
            r_b = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
            next_s_b = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
            mask_b = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
            action_mask_b = torch.Tensor(np.stack(batch.action_mask)).to(device=DEVICE)
            next_action_mask_b = torch.Tensor(np.stack(batch.next_action_mask)).to(device=DEVICE)

            
            # iterate batch to optimize
            actor_loss, critic1_loss, critic2_loss, alpha_loss = 0., 0., 0., 0.
            
            for _ in range(self.training_batch_iter):
                                     
                # Set all the gradients stored in the optimisers to zero.
                self.critic_optim.zero_grad()
                self.critic_optim_2.zero_grad()
                self.actor_optim.zero_grad()
                self.alpha_optim.zero_grad()
                self.cql_alpha_optimizer.zero_grad()
                
                # 1. update critic networks 
                
                qf1_loss , qf2_loss, cql1_alpha_loss, cql2_alpha_loss, cql_alpha   = self.calculate_critic_losses(s_b, a_b, r_b, next_s_b, mask_b, action_mask_b, next_action_mask_b)

                q1_loss = qf1_loss + cql1_alpha_loss.detach()
                q2_loss = qf2_loss + cql2_alpha_loss.detach()

                critic1_loss = critic1_loss + q1_loss
                critic2_loss = critic2_loss + q2_loss
               
                #backprop
                q1_loss.backward()
                q2_loss.backward()
                self.critic_optim.step()
                self.critic_optim_2.step()

                cql_alpha_loss = ( cql1_alpha_loss + cql2_alpha_loss) * 0.5 

                #update cql_alpha
                (-cql_alpha*(cql_alpha_loss.detach())).backward()
                self.cql_alpha_optimizer.step()

                #update target networks				
                self.soft_update_of_target_network(self.critic_local, self.critic_target, self.tau)
                self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2, self.tau)
                
                # 2. update actor network
                
                p_loss, log_pi = self.calculate_actor_loss(s_b, action_mask_b)
                actor_loss = actor_loss + p_loss.item()
                # backprop
                p_loss.backward()
                # set the inf in the gradient to 0
                for p in self.actor.parameters():
                    p.grad[p.grad != p.grad] = 0.0
                self.actor_optim.step()

                
                # 3. update alpha network without clipping
                
                if self.automatic_entropy_tuning:
                    al_loss = self.calculate_entropy_tuning_loss(log_pi)
                    alpha_loss = alpha_loss + al_loss.item()
                    
                    # backprop
                    al_loss.backward()
                    self.alpha_optim.step()
                    #alpha = self.log_alpha.exp()
                else: al_loss = None
                
                   
            #print(self.actor.eval())
            
            critic1_loss /= self.training_batch_iter
            critic2_loss /= self.training_batch_iter
            actor_loss /= self.training_batch_iter
            alpha_loss /= self.training_batch_iter
            
            logging.debug('<<dialog critic1 cql>> epoch {}, iteration {}, critic1, loss {}'.format(epoch, i, critic1_loss))
            logging.debug('<<dialog critic2 cql>> epoch {}, iteration {}, critic2, loss {}'.format(epoch, i, critic2_loss))
            logging.debug('<<dialog actor cql>> epoch {}, iteration {}, actor, loss {}'.format(epoch, i, actor_loss))
            logging.debug('<<dialog alpha cql>> epoch {}, iteration {}, alpha, loss {}'.format(epoch, i, alpha_loss))
            
            total_critic1_loss += critic1_loss
            total_critic2_loss += critic2_loss
            total_actor_loss += actor_loss
            total_alpha_loss += alpha_loss    

        total_critic1_loss  /= ( self.training_iter)
        logging.debug('<<dialog total critic1 cql>> epoch {}, total_critic1_loss {}'.format(epoch, total_critic1_loss))
        total_critic2_loss  /= ( self.training_iter)
        logging.debug('<<dialog total critic2 cql>> epoch {}, total_critic2_loss {}'.format(epoch, total_critic2_loss))
        total_actor_loss /= ( self.training_iter)
        logging.debug('<<dialog total actor cql>> epoch {}, total_actor_loss {}'.format(epoch, total_actor_loss))
        total_alpha_loss  /= ( self.training_iter)
        logging.debug('<<dialog total alpha cql>> epoch {}, total_alpha_loss {}'.format(epoch, total_alpha_loss))

        
        
    def batch_select_action (self, state_batch,action_mask):
            action=[]
            for s,a_mask in zip(state_batch,action_mask):
                a = self.actor.select_action(s.to(device=DEVICE), False, action_mask=a_mask.to(device=DEVICE) ).cpu()
                a_counter = 0
                while a.sum() == 0:
                    a_counter += 1
                    a = self.actor.select_action(s.to(device=DEVICE), True, action_mask=a_mask.to(device=DEVICE)).cpu()
                    if a_counter == 5:
                        break
                action.append(a.tolist())
            action = torch.tensor(action).to(device=DEVICE)
            return action

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch, action_mask, next_action_mask):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""

        with torch.no_grad():
            next_state_action = self.batch_select_action(next_state_batch,next_action_mask)                     

            next_state_log_pi, action_probabilities = self.actor.get_log_prob(next_state_batch,next_state_action,next_action_mask)
            qf1_next_target = self.critic_target.forward(next_state_batch.float())
            qf2_next_target = self.critic_target_2.forward(next_state_batch.float())
            next_target = torch.min(qf1_next_target, qf2_next_target)
            alpha = self.log_alpha.exp()
            min_qf_next_target = action_probabilities * (next_target - alpha * next_state_log_pi)
            min_qf_next_target =  min_qf_next_target.sum(dim=1)
            next_q_value = reward_batch + (1.0 - mask_batch) * self.discount_rate * (min_qf_next_target)
        
        q1 = self.critic_local.forward(state_batch)
        q2 = self.critic_local_2.forward(state_batch)
        q1_ = q1.gather(1, action_batch.long()).sum(dim=1)
        q2_ = q2.gather(1, action_batch.long()).sum(dim=1)
        
        qf1 = self.critic_local.forward(state_batch).sum(dim=1)
        qf2 = self.critic_local_2.forward(state_batch).sum(dim=1)
        qf1_loss =  F.mse_loss(q1_, next_q_value.detach())
        qf2_loss =  F.mse_loss(q2_, next_q_value.detach())

        cql1_scaled_loss = torch.logsumexp(qf1, dim=-1).mean() - q1_.mean()
        cql2_scaled_loss = torch.logsumexp(qf2, dim=-1).mean() - q2_.mean()

        cql_alpha = self.cql_log_alpha.exp()
        cql1_alpha_loss = cql_alpha*( cql1_scaled_loss)
        cql2_alpha_loss = cql_alpha*( cql2_scaled_loss)

        return qf1_loss , qf2_loss, cql1_alpha_loss, cql2_alpha_loss, cql_alpha
        
    def calculate_actor_loss(self, state_batch, action_mask):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""

        action = self.batch_select_action(state_batch, action_mask)
        log_pi, action_probabilities = self.actor.get_log_prob(state_batch,action,action_mask)

        qf1_pi = self.critic_local.forward(state_batch)
        qf2_pi = self.critic_local_2.forward(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        alpha = self.log_alpha.exp()
        inside_term  = ((alpha * log_pi) -  min_qf_pi)
        actor_loss  = (action_probabilities * inside_term).sum(dim=1).mean()
        log_pi = torch.sum( log_pi * action_probabilities, dim=1)

        return actor_loss,  log_pi
        
    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss
        
    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
        
    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.critic_local.state_dict(), directory + '/' + str(epoch) + '.critic1.mdl')
        torch.save(self.critic_local_2.state_dict(), directory + '/' + str(epoch) + '.critic2.mdl')
        torch.save(self.actor.state_dict(), directory + '/' + str(epoch) + '.pol.mdl')

        logging.info('<<dialog actor>> epoch {}: saved network to mdl'.format(epoch))
   
    def load(self, filename):
        critic1_mdl_candidates = [
            filename + '.critic1.mdl',		
            filename + '_cql.critic1.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.critic1.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_cql.critic1.mdl'),
        ]
        
        for critic1_mdl in critic1_mdl_candidates:
            if os.path.exists(critic1_mdl):
                self.critic_local.load_state_dict(torch.load(critic1_mdl, map_location=DEVICE))
                logging.info('<<dialog actor>> loaded checkpoint from file: {}'.format(critic1_mdl))
                break
        
        critic2_mdl_candidates = [
            filename + '.critic2.mdl',			
            filename + '_cql.critic2.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.critic2.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_cql.critic2.mdl')
        ]
        
        for critic2_mdl in critic2_mdl_candidates:
            if os.path.exists(critic2_mdl):
                self.critic_local_2.load_state_dict(torch.load(critic2_mdl, map_location=DEVICE))
                logging.info('<<dialog actor>> loaded checkpoint from file: {}'.format(critic2_mdl))
                break
        
        actor_mdl_candidates = [
            filename + '.pol.mdl',
            filename + '_cql.pol.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.pol.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_cql.pol.mdl')
        ]
        for actor_mdl in actor_mdl_candidates:
            if os.path.exists(actor_mdl):
                self.actor.load_state_dict(torch.load(actor_mdl, map_location=DEVICE))
                logging.info('<<dialog actor>> loaded checkpoint from file: {}'.format(actor_mdl))
                break

    def load_from_pretrained(self, archive_file, model_file, filename):
        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for cql actor is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(os.path.join(model_dir, 'best_cql.pol.mdl')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)

        actor_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.pol.mdl')
        if os.path.exists(actor_mdl):
            self.actor.load_state_dict(torch.load(actor_mdl, map_location=DEVICE))
            logging.info('<<dialog actor>> loaded checkpoint from file: {}'.format(actor_mdl))

        critic1_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.critic1.mdl')
        critic2_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.critic2.mdl')
        if os.path.exists(critic1_mdl):
            self.critic_local.load_state_dict(torch.load(critic1_mdl, map_location=DEVICE))
            logging.info('<<dialog actor>> loaded checkpoint from file: {}'.format(critic1_mdl))
        if os.path.exists(critic2_mdl):
            self.critic_local_2.load_state_dict(torch.load(critic2_mdl, map_location=DEVICE))
            logging.info('<<dialog actor>> loaded checkpoint from file: {}'.format(critic2_mdl))
