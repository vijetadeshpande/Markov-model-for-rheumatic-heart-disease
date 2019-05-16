#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:59:21 2019

@author: vijetadeshpande
"""
import pandas as pd
import numpy as np
#from Model import Model
#from MarkovChain import MarkovChain

class simulation:
    
    def __init__(self, model):
        self.root                       = model
        self.baseline_coverage_results  = {}
        
    def get_compartmental_sim(self, action = pd.DataFrame(), baseline = False):
        
        # define population matrix and compute the starting population
        cohort      = 7924897 #100000
        pop         = pd.DataFrame(0, index = np.r_[0:self.root.chain.size], columns = np.r_[0:81]) 
        pop         = np.matrix(pop)
        pop         = pop.astype(np.float64)
        age_dist    = self.root.chain.get_age_dist()
        pop[0,:]    = (age_dist.transpose()) * cohort
        
        # now get stationary distribution over state space for each age and compute starting 
        # population for each state in each age
        #for age in range(0, 81):
         #   pie         = np.matrix((self.root.chain.get_stationary_dist(age, action)).transpose())
          #  pop[:, age] = pop[0, age] * pie

        # now perform the transitions according to the tpm for a average life-time (70)
        # initialize
        surgery_ref_cost    = 0
        cost                = np.matrix(pd.DataFrame(np.zeros(pop.shape)))
        outcomes            = np.matrix(pd.DataFrame(np.zeros((pop.shape[0], 1))))
        incidence           = np.matrix(pd.DataFrame(np.zeros((pop.shape[0], 1))))
        DALY                = np.matrix(pd.DataFrame(np.zeros((pop.shape[0], 1))))
        QALY                = np.matrix(pd.DataFrame(np.zeros((pop.shape[0], 1))))
        RHD_death           = 0
        ARF_death           = 0
        
        #
        print('\nStarting compartmental simulation: \n' )
        
        for t in range(0, 70):
            
            # print progress
            if (t%14) == 0:
                print(('Simulation is %d percent complete')%(t*100/70))
            
            #print('Here t = %d'%(t))

            # initialize
            surgery_ref         = 0
            incidence_sample    = np.matrix(pd.DataFrame(np.zeros(pop.shape)))
            RHD_death_sample    = np.zeros((pop.shape[1], 1))
            ARF_death_sample    = 0
            
            # part of cohort getting dropped as we are only considering age from 0 to 80
            #dropped = pop[:, pop.shape[1] - 1].sum()
            
            for age in range(80, 0, -1):
                
                # first import tpm and trm for the current age
                trm = self.root.chain.get_trm(action)
                tpm = self.root.chain.get_tpm(age-1, action)
                
                # here we are doing two things
                # 1. Population transition
                # 2. Keeping track of the patients transitioning from RHD1 to RHD0
                
                # 2. lets do the 2nd task first, as it needs to be done before we update the population
                if not action.empty:
                    if 'VS' in action.index[action['Intervened'] > 0].tolist():
                        surgery_ref         = pop[6, age-1] * tpm.loc["RHD1", "RHD0"]
                        surgery_ref_cost    = surgery_ref * trm.loc['RHD1', 'RHD0']
                
                # 1. Population transition
                pop[:, age] = np.matmul(pop[:, age-1].T, np.matrix(tpm)).T
                
                # heart failures due to RHD
                RHD_death_sample[age, 0] += np.multiply(pop[:, age-1], np.matrix(tpm))[np.r_[6:self.root.chain.size-1],self.root.chain.size-1].sum()
                
                # heart failure due to ARF
                ARF_death_sample += np.multiply(pop[:, age-1], np.matrix(tpm))[np.r_[1:4],self.root.chain.size-1].sum()
                
                # calculate cost of transitions
                cost[:, age-1] += (np.multiply(np.multiply(pop[:, age-1], np.matrix(tpm)), np.matrix(trm))).sum(axis = 1)
                
                
            # outcomes
            outcomes += np.sum(pop, axis = 1)
            
            # incidence calculations
            incidence_sample    = pop[:, age] 
            incidence           = incidence * (1 - 1/(t+1)) + incidence_sample.sum(axis = 1) * (1/(t+1))
            
            # death rate due to RHD
            RHD_death = RHD_death * (1 - 1/(t+1)) + RHD_death_sample * (1/(t+1))
            
            # death rate due to ARF
            ARF_death = ARF_death * (1 - 1/(t+1)) + ARF_death_sample * (1/(t+1))
            
            # outcomes calculations
            DALY_sample = np.multiply((1 - self.root.chain.disability_w), pop.sum(axis = 1))
            QALY_sample = np.multiply(self.root.chain.disability_w, pop.sum(axis = 1))
            DALY        = DALY * (1 - 1/(t+1)) + DALY_sample * (1/(t+1))
            QALY        = QALY * (1 - 1/(t+1)) + QALY_sample * (1/(t+1))
            
            
            # make birts = deaths, changed to constant birth
            pop[:, 0] = 0 #(pop[pop.shape[0] - 1, :].sum() + dropped) * np.matrix((self.root.chain.get_stationary_dist(0)).transpose())
            pop[0, 0] = cohort * (self.root.chain.get_age_dist())[0][0] #* (self.root.chain.get_stationary_dist(0))[0][0] 
            
        #
        print('\n -- Compartmental simulation has completed -- \n' )
        
        # calculate prevalence
        prevalence = np.zeros((age_dist.shape[0], self.root.chain.size))
        for age in range(0,81):
            dist = pop[:, age]/pop[:, age].sum()
            prevalence[age, :] = (age_dist[0][age]*dist).T
            
        # adjust outcomes according to disability weights
        #QALY = np.multiply(outcomes, self.root.chain.disability_w)
        #DALY = np.multiply(outcomes, (1 - self.root.chain.disability_w))
        
        # adjustment
        cost                = pd.DataFrame(-1 * cost.sum(axis = 1), index = self.root.chain.states, columns = ['Cost'])
        surgery_ref_cost    = -1 * surgery_ref_cost
        outcomes            = pd.DataFrame(outcomes, index = self.root.chain.states, columns = ['Life years lived'])
        prevalence          = pd.DataFrame(prevalence, index = np.r_[0:81], columns = self.root.chain.states)

        # create dictionary for output
        output = {}
        output['Prevalence']        = prevalence
        output['Population']        = pop
        output['Incidence']         = incidence
        output['Mortality']         = {}
        output['Mortality']['RHD']  = RHD_death
        output['Mortality']['ARF']  = ARF_death
        output['Outcomes']          = {}
        output['Outcomes']['QALY']  = QALY
        output['Outcomes']['DALY']  = DALY
        output['Cost']              = {}
        output['Cost']['PP + SP']   = cost
        output['Cost']['VS']        = surgery_ref_cost
        
        # for calculation of inremental cost effectiveness analysis
        icer = pd.DataFrame()
        if self.baseline_coverage_results != {}:
            # calculate cost and outcomes for current policy
            c_c = cost
            o_c = outcomes
            
            # calculate cost and outcomes for baseline policy
            c_b = (self.baseline_coverage_results)['Cost']['PP + SP']
            o_b = (self.baseline_coverage_results)['Outcomes']
            
            # calculate the ICER
            delta_c     = c_c.subtract(c_b)
            delta_o     = o_c.subtract(o_b)
            icer        = np.divide(np.sum(np.matrix(delta_c)), np.sum(np.matrix(delta_o)))
            #icer        = np.sum(icer_state)
                
        else:
            if not baseline:
                print('\nWARNING in ''get_compartmental_results'' function of ''simulation'' class: \nICER calculations cannot be performed because there are no baseline results available \n')  
        
        # set the attribute value if baseline is true
        if baseline:
            self.baseline_coverage_results = output
            
        
        return output, icer
    
    '''
        prevalence = output['Prevalence']
        outcomes = output['Outcomes']          
        cost = output['Cost']['PP + SP']
        surgery_ref_cost = output['Cost']['VS']
        
    '''
    
    
    
#xyz = model.simulation.get_compartmental_sim(action)    
    
    