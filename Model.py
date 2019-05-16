#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:47:09 2019

@author: vijetadeshpande
"""

import numpy as np
import pandas as pd
import datetime
from database import database
from MarkovChain import MarkovChain
from simulation import simulation
from copy import deepcopy

class Model:
    
    def __init__(self, filenames = None):
        self.database               = database(self, filenames)
        self.chain                  = MarkovChain(self)
        self.simulation             = simulation(self)
        self.optimization           = None
        self.counter                = 0
        self.association_sheet_name = 'Eastern sub-Saharan Africa'
        
    
    def set_association_file(self, new_excel_file = False):
        if self.chain == None:
            print("Error in function 'set_association_file': MarkovChain object is either not define or Model.chain is not pointing to MarkovChain object")
            return
        
        # get template file
        template = deepcopy(self.database.asso_template)
        
        # collect starting indices of each block of state association     
        state_association = np.where(np.array(template.iloc[:, 0] == "<State Association>"))[0]
        treat_association = np.where(np.array(template.iloc[:, 0] == "<Treatment Association>"))[0]
        
        # import age distribution
        age_dist = (self.chain.get_age_dist())[:][:]

        # collect transition prob values for all ages
        tpm_dict = {}
        for i in range(0,81):
            tpm_dict[i] = (self.chain.get_tpm(i))[:][:]
            
        # take inverse exp of the transition probabilty values
        rates_dict = {}
        for i in range(0,81):
            rates_dict[i] = 1 * (-1 * np.log(np.ones(len(tpm_dict[i])) - tpm_dict[i]))
            rates_dict[i] = rates_dict[i].replace(float('inf'), 100)
            rates_dict[i] = rates_dict[i].replace(-0, 0)

            
        # calculate prevalence
        sim_out, icer = self.simulation.get_compartmental_sim()
        prev    = sim_out['Prevalence']
        sim_pop = sim_out['Population']
        prev2   = self.chain.get_prevance(rates_dict, age_dist)
        error   = np.square((prev - np.matrix(prev2))).sum()
        
        # compute expected values for rates
        # collect age group array
        age_groups      = template.iloc[7:19, 2].str.split("_", expand = True)
        age_groups      = age_groups.reset_index(drop = True)
        w_rates_dict    = {}
        prev_mat        = np.zeros((self.chain.size, age_groups.shape[0] - 1))
        prev_mat2       = np.zeros((self.chain.size, age_groups.shape[0] - 1))
        age_dist_grp    = np.zeros((1, len(age_groups) - 1))
        for i in range(0, len(age_groups) - 1):
            sum_float_rate  = np.zeros((self.chain.size, self.chain.size))
            sum_float_prev  = np.zeros((self.chain.size))
            sum_float_pop   = np.zeros((self.chain.size, 1))
            sum_float_age   = 0
            for j in range(int(age_groups[0][i]), int(age_groups[1][i]) + 1):
                sum_float_rate  += np.matrix(rates_dict[j]) #np.matrix(age_dist[0][j] * rates_dict[j])
                sum_float_prev  += prev.loc[j, :]
                sum_float_pop   += sim_pop[:,j]
                sum_float_age   += (age_dist.iloc[j])[0]
            w_rates_dict[i]     = pd.DataFrame(sum_float_rate, index = self.chain.states, columns = self.chain.states)
            prev_mat[:, i]      = sum_float_prev
            prev_mat2[:, i]     = (sum_float_pop/np.sum(sum_float_pop))[:,0]
            age_dist_grp[0, i]  = sum_float_age
        
        # start overwriting the template for state asssociations
        for i in state_association:
            
            # start and end of the current state association
            start   = i
            end     = i+20
            
            # current state
            current_state = (template.iloc[np.where(np.array(template.iloc[start:end, 1] == "StateID"))[0][0] + start, 2]).encode('ascii')
            if current_state == 'Disability':
                break
            current_state_idx = self.chain.states.index(current_state)
            
            # select the prevalence martix appropriately
            if current_state == 'DsFreeSus':
                prevalence = prev_mat
            else:
                prevalence = prev_mat2
                
            # find states to which transitions are possible
            to_state_col    = np.where(np.array(template.iloc[start + 5, :] == "<ToState>"))[0]
            to_state        = template.iloc[start + 7, to_state_col]
            
            # prevalence column number
            prev_col = np.where(np.array(template.iloc[start + 5, :] == "<Baseline Prevalence>"))[0]
            
            # print prevalence values
            template.iloc[start + 7:start+7+len(age_groups) - 1 , prev_col] = prevalence[current_state_idx, :].T

            # print rate values
            col = 0
            for state_i in to_state:
                for age_i in range(0, len(age_groups) - 1):
                    template.iloc[start + 7 + age_i, to_state_col[col] - 1] = w_rates_dict[age_i][state_i][current_state]
                col += 1
                
        
        # clean the final template
        template = template.replace('x', 0)
        # following line adjustment and should be removed once delphi code is capable of rewading treatment
        #template = template.iloc[0:175, :]
        
        if new_excel_file:
            # save as xlsx
            time_date = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
            file_name = '%s, Association File (ARF_RHD).xlsx'%(time_date)
            # template.to_excel(string, index = False, sheet_name = self.association_sheet_name)
            
            with pd.ExcelWriter(file_name) as writer:  # doctest: +SKIP
                template.to_excel(writer, index = False, sheet_name = self.association_sheet_name + '_F')
                template.to_excel(writer, index = False, sheet_name = self.association_sheet_name + '_M')
                

        return prev, prev2
                
