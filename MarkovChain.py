#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:27:06 2019

@author: vijetadeshpande
"""

'''
sample code structure for the markov chain
'''
#from database import database
import numpy as np
import scipy as sp
import pandas as pd
from copy import deepcopy


class MarkovChain:
    
    def __init__(self, model):
        
        d = {}
        d["DsFreeSus"]  = {"Out": ["ARF0"], "In": []}
        d["ARF0"]       = {"Out": ["REM", "RHD0", "Deceased"], "In": ["DsFreeSus"]}
        d["REM"]        = {"Out": ["ARF1"], "In": ["ARF0", "ARF1"]}
        d["ARF1"]       = {"Out": ["REM", "RHD0", "Deceased"], "In": ['REM']}
        d["RHD0"]       = {"Out": ["STK", "RHD1"], "In": ["ARF0", "ARF1", "RHD1"]}
        d["STK"]        = {"Out": ["Deceased"], "In": ["RHD0"]}
        d["RHD1"]       = {"Out": ["RHD0", "Deceased"], "In": ["RHD0"]}
        d["Deceased"]   = {"Out": [], "In": ["ARF0", "ARF1", "STK", "RHD1"]}
        
        self.root                   = model
        self.states                 = ["DsFreeSus", "ARF0", "REM", "ARF1", "RHD0", "STK", "RHD1", "Deceased"]
        self.size                   = len(self.states)
        self.possible_transitions   = d
        self.action_set             = ['PP', 'SP', 'VS']
        self.base_tpm               = None
        self.set_base_tpm()
        self.cost                   = None
        self.set_cost()
        self.disability_w           = None
        self.set_disability_w()        

        
    def set_base_tpm(self):
        
        data = deepcopy((self.root.database.markov_chain_data)['Data for transition probabilities'])
        
        # the input data needs to be filtered here, because
        # 1. input data has set of fixed values and variables
        # 2. transition probability values are variables
        # 3. we'd like to base our calculations on the fixed values and then cross-check with variable values in data
        
        # take the base values of transition probabilities as fixed value
        p_val               = pd.DataFrame(data = data.iloc[2:, 3])
        p_val.columns       = ["base value"]
        state_names         = data.iloc[2:, 1].str.split("_", expand = True)
        state_names.columns = ["Prev", "Next"]
        replace_vector      = list(["DsFreeSus", "ARF0", "ARF0", "ARF0", "", "REM", "ARF1", "ARF1", "ARF1", "", "RHD0", "RHD0", "RHD0", "", "RHD1", "RHD1", "RHD1", "", "STK"])                       
        state_names["Prev"] = state_names["Prev"].replace(list(state_names["Prev"]), replace_vector)
        replace_vector      = list(["ARF0", "RHD0", "Deceased", "REM", "", "ARF1", "Deceased", "RHD0", "REM", "", "RHD1", "STK", "RHD0", "", "RHD0", "Deceased", "RHD1", "", "Deceased"])
        state_names["Next"] = state_names["Next"].replace(list(state_names["Next"]), replace_vector)
        
        # reindexng rows
        p_val       = p_val.reset_index(drop = True)
        state_names = state_names.reset_index(drop = True)
        
        # create a matrix
        base_tpm = pd.DataFrame(index = self.states, columns = self.states)
        
        # fill the matrix
        for i in range(0, len(p_val)):
            if (state_names["Prev"][i] == "" and state_names["Next"][i] == ""):
                continue
            else:
                base_tpm.loc[state_names["Prev"][i], state_names["Next"][i]] = np.array(p_val)[i, 0]
                
        # check row sum
        base_tpm = base_tpm.fillna(0)
        for i in self.states:
            if (base_tpm.sum(axis = 1) != 1)[i]:
                base_tpm.loc[i, i] = 1 - sum(np.array(base_tpm.loc[i, :]))
                
        # point to the appropriate attribute
        self.base_tpm = base_tpm
        
        return 

        
    def get_tpm(self, age, action = pd.DataFrame()):
        
        # import required data
        risk_reduction = deepcopy((self.root.database.markov_chain_data)['Data for risk reduction'])
        
        # get the base tpm to base the caculations on
        tpm = pd.DataFrame((self.base_tpm)[:][:], copy = True)
        
        # changing TP values according to age condition in Watkins 2016
        if age >= 14:
            if age >= 24:
                # incidence of ARF0 and ARF1 will change in this condition
                tpm["ARF0"]["DsFreeSus"]        = tpm["ARF0"]["DsFreeSus"] * np.exp(-0.1 * (age - 14))
                tpm["DsFreeSus"]["DsFreeSus"]   = 1 - sum(np.array(tpm.loc["DsFreeSus", tpm.columns != "DsFreeSus"]))
                tpm["ARF1"]["REM"]              = tpm["ARF1"]["REM"] * np.exp(-0.1 * (age - 24))
                tpm["REM"]["REM"]               = 1 - sum(np.array(tpm.loc["REM", tpm.columns != "REM"]))
            else:
                # incidence of ARF0 will change in this condition
                tpm["ARF0"]["DsFreeSus"]        = tpm["ARF0"]["DsFreeSus"] * np.exp(-0.1 * (age - 14))
                tpm["DsFreeSus"]["DsFreeSus"]   = 1 - sum(np.array(tpm.loc["DsFreeSus", tpm.columns != "DsFreeSus"]))

        # now we have to modify the TPM according to the action
        if action.empty:
            return tpm
        else:
            if risk_reduction.empty:
                print('Error in get_tpm function of MarkovChain class: function needs an input of risk reduction data')
                return
            
            # the transitions which are going to get affected due to the current action taken
            # 1. Healthy to ARF0
            # 2. Remission to ARF1
            # 3. RHD1 to RHD0
            # therefore, we'll only track and change rows corresponding to DsFreeSus, REM and RHD1
            
            # first create post matrix for intervention tpms
            post = pd.DataFrame(tpm, copy = True)
            
            # 1. consider the healthy to ARF0 transition
            post['ARF0']['DsFreeSus']       = ((1 - action.loc['PP',:]) * tpm['ARF0']['DsFreeSus']) + (action.loc['PP', :] * tpm['ARF0']['DsFreeSus'] * risk_reduction.iloc[0,0])        
            post['DsFreeSus']['DsFreeSus']  = 1 - post['ARF0']['DsFreeSus']
            
            # 2. REM to ARF1
            post['ARF1']['REM'] = ((1 - action.loc['SP', :]) * tpm['ARF1']['REM']) + (action.loc['SP', :] * tpm['ARF1']['REM'] * risk_reduction.iloc[1,0])
            post['REM']['REM']  = 1 - post['ARF1']['REM']
            
            # 3. RHD1 to RHD0
            post['RHD0']['RHD1']        = ((1 - action.loc['VS', :]) * tpm['RHD0']['RHD1']) + (action.loc['VS', :] * risk_reduction.iloc[2,0])
            post['Deceased']['RHD1']    = (1 - post['RHD0']['RHD1']) * post['Deceased']['RHD1']
            post['RHD1']['RHD1']        = (1 - post['RHD0']['RHD1']) * post['RHD1']['RHD1'] 
            
            if any(post.sum(axis = 1) != 1):
                print('Error in get_tpm function of MarkovChain class: rows are not adding to 1')
                return
            
            # return both tpms
            return post
            
    
    def get_stationary_dist(self, age, action = pd.DataFrame()):
        
        # get transition prob matrix for current age
        tpm = (self.get_tpm(age, action))[:][:]
        
        # alter the last row to make the birth-equal-death process
        tpm["Deceased"]["Deceased"]     = 0
        tpm["DsFreeSus"]["Deceased"]    = 1
        tpm                             = np.matrix(tpm)
        
        # get the steady state distribution
        pie = pd.DataFrame((np.linalg.matrix_power(tpm, 1000))[:][0])
        
        return pie
    
    def get_age_dist(self):
        
        return deepcopy((self.root.database.markov_chain_data)['Data for age distribution'])
    
    def get_prevance(self, rates_dict, age_dist = pd.DataFrame()):
        
        #check if the age distribution is defined or not
        if age_dist.empty:
            age_dist = self.get_age_dist()
            
        # assign a value to strting population
        cohort      = 100000
        start_pop   = cohort * age_dist
        
        # create a population matrix
        pop         = np.zeros(((age_dist.shape)[0], self.size))
        pop[:, 0]   = start_pop.iloc[:, 0]
        pop         = pd.DataFrame(pop, columns = self.states)
        
        # change data type of rates 
        rates = []
        for age in range(0, 81):
            rates.append(np.matrix(rates_dict[age]))
        rates = np.array(rates)
        
        #
        print('\nStarting Markov process simulation: \n' )
        
        # now we want to simulate the Markov process 
        dt = 0.067
        for t in range(0, 500):
            
            # variables/data to update after each year
            
            for n in range(1, 16):               
                
                
                # healthy state
                pop['ARF0']         += pop['DsFreeSus'] * ((rates[:, 0, 1]) * dt)
                pop['DsFreeSus']    -= pop['DsFreeSus'] * ((rates[:, 0, 1]) * dt)
                
                # ARF0
                pop['REM']      += pop['ARF0'] * (rates[:, 1, 2] * dt)
                pop['RHD0']     += pop['ARF0'] * (rates[:, 1, 4] * dt)
                pop['Deceased'] += pop['ARF0'] * (rates[:, 1, 7] * dt)
                pop['ARF0']     -= pop['ARF0'] * (np.sum(rates[:, 1, [2,4,7]], axis = 1) * dt)
                
                # REM
                pop['ARF1']     += pop['REM'] * (rates[:, 2, 3] * dt)
                pop['REM']      -= pop['REM'] * (rates[:, 2, 3] * dt)
                
                # ARF1
                pop['REM']      += pop['ARF1'] * (rates[:, 3, 2] * dt)
                pop['RHD0']     += pop['ARF1'] * (rates[:, 3, 4] * dt)
                pop['Deceased'] += pop['ARF1'] * (rates[:, 3, 7] * dt)
                pop['ARF1']     -= pop['ARF1'] * (np.sum(rates[:, 3, [2,4,7]], axis = 1) * dt)
                
                # RHD0
                pop['STK']      += pop['RHD0'] * (rates[:, 4, 5] * dt)
                pop['RHD1']     += pop['RHD0'] * (rates[:, 4, 6] * dt)
                pop['RHD0']     -= pop['RHD0'] * (np.sum(rates[:, 4, [5,6]], axis = 1) * dt)
                
                # STK
                pop['Deceased'] += pop['STK'] * (rates[:, 5, 7] * dt)
                pop['STK']      -= pop['STK'] * (rates[:, 5, 7] * dt)
                
                # RHD1
                pop['RHD0']     += pop['RHD1'] * (rates[:, 6, 4] * dt)
                pop['Deceased'] += pop['RHD1'] * (rates[:, 6, 7] * dt)
                pop['RHD1']     -= pop['RHD1'] * (np.sum(rates[:, 6, [4,7]], axis = 1) * dt)
                
                
                '''
                # make transitions according to the transition rates
                for state in self.states:

                    # indices of states to which a transition can happen from current state
                    to_states = self.possible_transitions[state]["Out"]
                    to_states_idx = []
                    for j in to_states:
                        to_states_idx.append(np.where(np.array(self.states) == j)[0][0])
                    
                    # indices of states from which transition to current state is possible
                    from_states = self.possible_transitions[state]["In"]
                    from_states_idx = []
                    for j in from_states:
                        from_states_idx.append(np.where(np.array(self.states) == j)[0][0])
                        
                    # index of current state
                    state_idx = np.where(np.array(self.states) == state)[0][0]
                    
                    # out flow
                    if to_states != []:
                        pop[state] = pop[state] * (np.ones(((age_dist.shape)[0],)) - (np.sum(rates[:, state_idx, to_states_idx] * dt, axis = 1)))
                    
                    # in flow
                    if from_states != []:
                        counter = 0
                        for j in from_states:
                            pop[state] += dt * pop[j] * rates[:, from_states_idx[counter], state_idx]
                            counter += 1
                            
                '''
    
            # perform aging
            pop             = pop.shift(1)
            pop.iloc[0, :]  = 0
                
            # births
            #pop.iloc[0, :]  = (pop["Deceased"].sum()) * np.matrix((self.get_stationary_dist(0)))
            #pop["Deceased"] = 0
            pop.iloc[0, :] = 0
            pop.iloc[0, 0] = cohort * (self.root.chain.get_age_dist())[0][0] #* (self.root.chain.get_stationary_dist(0))[0][0] 
            
            # print progress
            if (t%100) == 0:
                print(('Simulation is %d percent complete')%(t*100/500))
            
        #
        print('\n -- Markov process simulation has completed -- \n' )
        
        # prevalence of each state according to each age
        prevalence = pd.DataFrame(0, index = np.r_[0:81], columns = self.states)
        for age in range(0,81):
            for state in self.states:
                if pop[state].sum() != 0:
                    prevalence.loc[age, state] = ((pop[state][age]/pop.iloc[age, :].sum()) * age_dist.iloc[age])[0]

        return prevalence
    
    
    def set_cost(self):
        
        # import required data
        data = deepcopy((self.root.database.markov_chain_data)['Data for transition rewards'])
        
        # healthcare cost
        h_cost = pd.DataFrame(0, index = self.states, columns = self.states)
        
        # intervention cost
        i_cost = {}
        nature = self.action_set
        for i in range(0,len(nature)):
            i_cost[nature[i]] = pd.DataFrame(0, index = self.states, columns = self.states)
        
        # reset indices of data
        data = data.reset_index(drop = True)
        
        # healthcare cost
        h_cost['ARF0'][:]  -= data.iloc[2, 3]
        h_cost['ARF1'][:]  -= data.iloc[2, 3]
        h_cost['REM'][:]   -= data.iloc[3, 3]
        h_cost['RHD0'][:]  -= data.iloc[4, 3]
        h_cost['RHD1'][:]  -= data.iloc[5, 3]
        h_cost['STK'][:]   -= data.iloc[6, 3]
        
        # intervention cost (considering without discount values for now)
        
        # PP (PP includes community and provider education, surveillance, program
        #    administrative costs, and additional clinical expenses needed to manage all cases of streptococcal
        #    pharyngitis appropriately.)
        #i_cost['PP']               -= data.iloc[10, 3]
        #i_cost['PP']["Deceased"]   = 0
        
        # SP (case finding efforts,
        #   maintenance of a patient registry, provider education, program administrative costs, and additional
        #   clinical expenses needed to deliver monthly penicillin injections to all cases)
        #i_cost['SP']['ARF0'][:]   -= data.iloc[11, 3]
        #i_cost['SP']['REM'][:]    -= data.iloc[11, 3]
        #i_cost['SP']['ARF1'][:]   -= data.iloc[11, 3]
        
        # VS (either building infrastructure or refering abroad)
        #i_cost['VS']['RHD0']['RHD1'] -= data.iloc[12, 3]
        
        # point it to attribute
        cost = {}
        cost['Healthcare cost']     = h_cost
        cost['Intervention cost']   = i_cost
        
        # point to attribute
        self.cost = cost
        
        return
    
    def get_trm(self, action):
        
        # set get healthcare_cost and intervention cost
        cost = deepcopy(self.cost)
        h_cost = cost['Healthcare cost']
        i_cost = cost['Intervention cost']
        
        # initialize trm
        trm = h_cost
        if not action.empty:
            idx = action.index[action['Intervened'] > 0].tolist()
            for i in idx:
                trm += i_cost[i]
           
        return trm
    
    def set_disability_w(self):
        
        # import data
        data = deepcopy((self.root.database.markov_chain_data)["Data for transition rewards"])
        data = data.reset_index(drop = True)
        
        # create a mareix
        disability_w = np.matrix(np.ones((self.size, 1)))
        
        # fill the matrix
        disability_w[1,0] -= data.iloc[20,3] # ARF0
        disability_w[3,0] -= data.iloc[20,3] # ARF1
        disability_w[4,0] -= data.iloc[21,3] # RHD0
        disability_w[5,0] -= data.iloc[23,3] # STK
        disability_w[6,0] -= data.iloc[22,3] # RHD1
        disability_w[7,0] -= 1
        
        # point to attribute
        self.disability_w = disability_w
        
        return 
        
        
        
                        


'''
o1 = MarkovChain()
dt = get_data(filename_watkins)
o1.set_base_tpm(dt["Data for transition probabilities"])
'''

'''    
model = Model.Model()
rhd = MarkovChain()
model.chain = rhd
p = rhd.get_tpm(10)
pie = rhd.get_stationary_dist(10)
'''
