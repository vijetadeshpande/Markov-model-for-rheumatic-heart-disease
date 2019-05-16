#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:39:47 2019

@author: vijetadeshpande
"""

import pandas as pd
import numpy as np

class database:
    
    def __init__(self, model, filenames):
        self.country            = 'AFRE'
        self.root               = model
        self.markov_chain_data  = None
        self.asso_template      = None
        self.set_data_mc(filenames['Watkins_2016'], filenames['Pop_age_dist'], filenames['Association Template'])


    def set_data_mc(self, parameter_file, age_dist_file, association_file):
        
        # define empty dictionary
        d = {}
        
        # read the file
        input_parameters = pd.read_excel(parameter_file, sheet_name = "Model Inputs", header = None)
        
        # section of the data for calculation of TPM
        row_index   = np.r_[7:28]
        col_index   = np.r_[1:13, 17:len(input_parameters.columns)]
        tpm_data    = input_parameters.iloc[row_index, col_index]
        
        # section of data for costs and reward calculations
        row_index   = np.r_[35:input_parameters.shape[0]]
        col_index   = np.r_[1:13, 17:len(input_parameters.columns)]
        trm_data    = input_parameters.iloc[row_index, col_index]
        
        # data for rsik reduction
        row_index                       = np.r_[31:33+1]
        col_index                       = 3
        float_s                         = input_parameters.iloc[row_index, col_index]
        risk_reduction                  = pd.DataFrame(0, index = ['DsFreeSus_ARF0', 'REM_ARF1', 'RHD1_RHD0'], columns = ['Reduction'])
        risk_reduction['Reduction'][:]  = float_s
        
        # baseline intervention for ante
        other_parameters        = pd.read_excel(parameter_file, sheet_name = "Other parameters", header = None)
        row_index               = np.r_[46:51+1:2]
        col_index               = 3
        ante                    = other_parameters.iloc[row_index, col_index]
        row_index               = np.r_[47:52+1:2]
        post                    = other_parameters.iloc[row_index, col_index]
        coverage                = pd.DataFrame(0, index = ['Primary prevention', 'Secondary prevention', 'Surgery'], columns = ['Baseline','Scale-up'])
        coverage['Baseline'][:] = ante
        coverage['Scale-up'][:] = post
        
        # now set the age dist
        pop_data = pd.read_excel(age_dist_file)
        pop_data = pop_data.iloc[np.where(np.array(pop_data.iloc[:,1] == self.country))[0], 6]
        pop_data = pop_data.reset_index(drop = True)
        pop_data = pop_data.iloc[0:81, ]
        prob_age = pd.DataFrame(np.array(pop_data)/pop_data.sum())
        
                
        d['Data for transition probabilities']  = tpm_data
        d['Data for transition rewards']        = trm_data
        d['Data for risk reduction']            = risk_reduction
        d['Data for coverage']                  = coverage
        d['Data for age distribution']          = prob_age
        
        # place data on right pointer
        self.markov_chain_data = d
        
        # set association template
        template            = pd.read_excel(association_file)
        template            = template.reset_index(drop = True)
        self.asso_template  = template
        
        #
        print('''\n \n Data have been read and set to the database attribute. \n This msg should not print more than once \n\n''')
        
        return 