#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:16:51 2019

@author: vijetadeshpande
"""


'''
structure of the main file

1. import data and structure it in pieces required further for calculations
2. create model object
3. create markov chain object whithin
4. create a simulation model within
5. create decision model within
6. function for validation
7. function results and sensititvity analysis


PSEUDOCODE
filename = "RHD excel model_Watkins.XLSM"
data_dictionary = get_data("filename")

# create model
model           = markovmodel()

# append chain to the model
rhd             = MarkovChain_RHD()
model.chain     = rhd
rhd.set_transition_probabilities(data_dictionary["Data for transition probabilities"])



'''

from Model import Model
from database import database
import pandas as pd
import numpy as np
import scipy as sp
import sys

# dictionary for filenames
filenames = {}
filenames['Watkins_2016']           = "/Users/vijetadeshpande/Downloads/Avenir/RHD excel model_Watkins.XLSM"
filenames['Pop_age_dist']           = "/Users/vijetadeshpande/Downloads/Avenir/My codes/res4.xlsx"
filenames['Association Template']   = "/Users/vijetadeshpande/Downloads/Avenir/Association file/RheumaticHeartDisease_Template.xls"

# create model object
model = Model(filenames)

# create Markov chain object and link it to model
rhd = model.chain

# create simulation object and link it to model
sim = model.simulation

# set the association file
prev1, prev2 = model.set_association_file(True)

a1 = prev1 * 100000
a2 = prev2 * 100000
a_diff = a1-a2
a_diff_sq = np.square(a_diff)
a_sum_sq_err = a_diff_sq.sum(axis = 0)
RHD_prev1 = a1.loc[:, ["RHD0", "RHD1", "STK"]].sum().sum()
RHD_prev2 = a2.loc[:, ["RHD0", "RHD1", "STK"]].sum().sum()

x = rhd.get_age_dist()
# baseline
# set transition probabilities and check whether the row elements are summing to 1 for randomly selected 5 ages
a = [0.1, 0.1, 0.1]
action = pd.DataFrame(a, index = rhd.action_set, columns = ["Intervened"])
sim_output_baseline, icer1 = model.simulation.get_compartmental_sim(action, baseline = True)

# run the simulation model for scenario 1
a = [0.7, 0.1, 0.1]
action = pd.DataFrame(a, index = rhd.action_set, columns = ["Intervened"])
sim_output_ac2, icer2 = model.simulation.get_compartmental_sim(action)

# run the simulation model for scenario 2
a = [0.1, 0.92, 0.1]
action = pd.DataFrame(a, index = rhd.action_set, columns = ["Intervened"])
sim_output_ac3, icer3 = model.simulation.get_compartmental_sim(action)

'''
PSEUDOCODE: reinforcement learning for finding approximate optimal policy to prevent and treat RHD

Initial notes:
The paper Watkins_2016 basically is defining the transition transition probabilities for us.
But the data, from which the transition probabilities are defined, does have lot of regional
variability, e.g. the different countries in eastern sub-sharan Africa have very different
prevalence and incidence. Some of the countries don't even have data. Therefore, in such
case it will be interesting to compute an aggregate policy for prevention and treatment.
Hence, good to have a general guidlines for investment in primary prevention, secondary
prevention and surgery infrastructure (or surgery referrals to other countries).

First of all, if we have transition probability matrix, why to use RL in that case. Because,
the data from which the transition rates/probabilities are defined is itself with huge
regional variability. Therefore we can say that the environment is variable and does not
respond or make transitions exactly as defined by TPM. Therefore, by defining a Dirichlet
distribution ober the transition probabilities/rates and randomly choosing samples from
distribution, we can inject the variability in environment. When an agent is reaches the
approximate optimal decisions in such environment we can have a good idea about the
aggreate intervention/treatment policy or investment in PP, SP or VS


'''

'''

# set transition probabilities and check whether the row elements are summing to 1 for randomly selected 5 ages
a = [0, 0, 0]
action = pd.DataFrame(a, index = rhd.action_set, columns = ["Intervened"])
sim_output_natural, icer_natural = model.simulation.get_compartmental_sim(action, baseline = True)


# baseline
# set transition probabilities and check whether the row elements are summing to 1 for randomly selected 5 ages
a = [0.1, 0.1, 0.1]
action = pd.DataFrame(a, index = rhd.action_set, columns = ["Intervened"])
sim_output_baseline, icer1 = model.simulation.get_compartmental_sim(action, baseline = True)

# run the simulation model
a = [0.7, 0.1, 0.1]
action = pd.DataFrame(a, index = rhd.action_set, columns = ["Intervened"])
sim_output_ac1, icer2 = model.simulation.get_compartmental_sim(action)


# testing results
prev1 = sim_out

'''

'''
Once we set the base tpm, then all information required for writing the aasociation
file (except the treatment states in association file) is with us and we can
call the set_association_file function in the object of class Model

-----------

# call the set association file function
prev1, prev2, e = model.set_association_file()

a1 = prev1 * 100000
a2 = prev2 * 100000

a_diff = a1-a2

a_diff_sq = np.square(a_diff)

a_sum_sq_err = a_diff_sq.sum(axis = 0)

RHD_prev1 = a1.loc[:, ["RHD0", "RHD1"]].sum().sum()
RHD_prev2 = a2.loc[:, ["RHD0", "RHD1"]].sum().sum()

'''
