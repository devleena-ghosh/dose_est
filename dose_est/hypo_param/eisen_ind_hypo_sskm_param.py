'''code for estimating parameters using genetic algorithms

author: Devleena Ghosh
#################################################################
'''
'''
ToDO:
Increase population variation

'''

#importing calculate_data function from parameter_estimation.py file
import array
from binascii import crc_hqx
import multiprocessing
import random as rnd
import sys, os
import numpy as np
# import trhParameters as trp
#from trhODE_1 import *	
import math	
import csv
from scipy.interpolate import interp1d
#from itertools import repeat
from operator import attrgetter  
# from parseODE import *
# from ode_util_nm import *
from scipy.stats import entropy
from datetime import datetime

import multiprocessing
# from multiprocessing import Pool#, Queue, Process
from pathos.multiprocessing import ProcessingPool

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from odeUtil.parseODE import *

from sagepy.sageCal import *
from sagepy.optimizeOffset import *
from sagepy.drugFourier import *

LOOP =  True 
# LOOP = False
# TEST = True
IFPOOL = False #True
TEST = False
PLOT = True
LOG = False
# LOG = True
ENTROPY = True
GPC = False
OPT = True #False
#from fim_morris_4 import *
if PLOT:
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_pdf import PdfPages
#import fim
#import pretty_print as pprint
import time

import warnings
warnings.filterwarnings("ignore")
#import warnings

st = time.time()
print('Start time : ', st)

seed = 64 #datetime.now()
print('### Seed used : ', seed)
rnd.seed(seed) #datetime.now())

import multiprocessing
# from multiprocessing import Pool
#, Queue, Process
# from pathos.multiprocessing import ProcessingPool

import odeUtil.ode_util_eisen as trh_ode_util
import odeUtil.parseODE as trh_parse
# import FIM_util.sense_util as su
import eisen_dose_params as ep

from plotODE_eisen import *	
import opt_func as opt

# import odeUtil.gp_classifier as gpc

# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

COMBINED = False


# eu_csv = 'eu_clustered_3obs_updated_new.csv'
hypo_csv = 'sskm/sskm_serialised_dist.csv' #'hypo_clustered_3obs.csv'
# result_file = 'sskm/eisen_sskm_opt_22323.csv'.format(index)
# top_sol_csv = 'sim_new/top_solutions_02_12.csv' #top_solutions_04_01_dist.csv'
# simulated_csv = 'sim_new/simulated_data_dist_142.csv'
simulated_new_csv = 'sim_new/simulated_data_new_dist.csv'
plotFileName = 'plot_dosage.pdf'
sskm_hypo_test_params = 'sskm/eisen_sskm_opt_data_070723.csv'
data_file = 'sskm/eisen_sskm_hypo_070723.csv'


l_var = 14 #20
NK = 100
var_output = {}

t_unit1 = 3600.0
t_unit2 = 60.0
t_units = [t_unit1] #, t_unit2]

all_indices = ep.getParam_indices()
ParamNames_all = ep.getParam_all()
# Default_values = ep.getdefaultParams()
range_all = ep.getRanges()
Default_Params = ep.getdefaultParams()
# dose_all_indices = ep.getParam_indices()

selectedIndices =  [ep.ind_d_1,  ep.ind_p_1, ep.ind_p_19, ep.ind_p_7, ep.ind_p_40, ep.ind_p_10, ep.ind_p_37, ep.ind_p_13]
#[ep.ind_d_1, ep.ind_p_38, ep.ind_p_39, ep.ind_p_7,  ep.ind_p_40] # ep.ind_p_1, ep.ind_p_19, ep.ind_p_37, #ep.ind_p_40, ep.ind_p_37, ep.ind_p_38, ep.ind_p_39, ep.ind_p_23, ep.ind_p_20, ep.ind_p_29, ep.ind_p_6, ep.ind_p_3, ep.ind_p_1, ep.ind_p_19] #trp.ind_k4, trp.ind_kdelay_4] #, trp.ind_k3, trp.ind_k0, trp.ind_k1, trp.ind_k5, ]
#selectedIndices = [trp.ind_p_40, trp.ind_p_37, trp.ind_p_38, trp.ind_p_39, trp.ind_p_23, trp.ind_p_20, trp.ind_p_29, trp.ind_p_6, trp.ind_p_3, trp.ind_p_1, trp.ind_p_19, trp.ind_p_7, trp.ind_p_24]
hypo_Indices =  selectedIndices+ [ep.ind_p_47, ep.ind_p_48]
hashmap = ep.findIndices(hypo_Indices)

selected_Names = [ParamNames_all[i] for i in hypo_Indices]

dose_selectedIndices = [ep.ind_dose_t4, ep.ind_et4_init,  ep.ind_eps_t4, ep.ind_delta_t4]
if COMBINED:
	dose_selectedIndices += [ep.ind_dose_t3,  ep.ind_et3_init,  ep.ind_eps_t3, ep.ind_delta_t3]
	
KLow = []
KUpp = []
ParamNames = []

for index in selectedIndices:
	#print(index, Param)
	KLow.append(range_all[index][0])
	KUpp.append(range_all[index][1])
	ParamNames.append(ParamNames_all[index])

# print('#############')
# for key in hashmap.keys():
# 	print(key, ParamNames_all[key], hashmap[key])
# print('########################')
# for i in range(len(ParamNames)):
# 	print(ParamNames[i], KLow[i], KUpp[i])
# print('########################')

eq = ep.equation #getEquations(t_units)
# ode_models = {}
# for item in equations.keys():
# 	t_unit = item
# 	eq = equations[item]
# 	#print('trp', eq)
# 	ode_model = trh_parse.getModel(eq)
# 	ode_model.setParamIndices(all_indices)
# 	ode_models.update({t_unit:ode_model})

# print(eq)

ode_models = {}
ode_model = trh_parse.getModel(eq)
ode_model.setParamIndices(all_indices)
ode_models.update({t_unit1:ode_model})

#print(str(ode_model1))
var_output = trh_ode_util.generate_ODE_file(ode_models, hypo_Indices, hypo_Indices, importFile = 'eisen_dose_params', trp=ep, diffFile = "diffEqn_eisen.py")
#trh_ode_util.generate_ODE_file(ode_models, selectedIndices, selectedIndices)
#var_output2 = generate_ODE_file(ode_model1, selectedIndices, sensitiveIndices)
print('Generated equations again')
import diffEqn_eisen as de

F1 = 0
#exit()

SPECIFIED_ERR_BOUND = 0.10
BOUND = 0.008
FLAG_RANGE = 0

observable_names = ['TSH', 'FT4', 'T3', 'T4', 'FT3'] #'T4', 'T3', 'FT4', 'FT3']
observable_indices = [var_output[nm] for nm in observable_names]
observable_index = [var_output[nm] for nm in observable_names]
# target_obs = [ep.AVG_tsh, ep.AVG_FT4, 'T3', 'T4', 'FT3']

dose_var_names = ['TSH', 'FT4'] #, 'T3'] #, 'T4', 'FT3'] #['TSH', 'FT4'] #, 'T3', 'FT4', 'FT3']
dose_observable_index = [var_output[nm] for nm in dose_var_names]

# observed_indices =  [var_output['t4'], var_output['t3p'], var_output['tsh']]
# observable_indices = (var_output['tsh'], var_output['t3p'], var_output['t4'], var_output['ft3'], var_output['ft4'])
# observable_index = (var_output['tsh'], var_output['t3p'], var_output['t4'], var_output['ft3'], var_output['ft4'])

steady_indices = [var_output['TSH'], var_output['T4'], var_output['T3']]

t_d = 24
total_time_to_siimulate = int(2*24)
sample_per_hr = 2.0
sample_scaling_factor = 2.0
time_unit_chosen = 3600.0
total_samples = int(total_time_to_siimulate*sample_per_hr*sample_scaling_factor)
sample_after_30_mins = int(0.5*sample_per_hr*sample_scaling_factor)
samples_ph = 10

transiet_time_tosimulate = int(100*24)
transient_sampHr = 0.1
transient_samples = int(transiet_time_tosimulate*transient_sampHr)

intermediate_time_tosimulate = int(5*24)
intermediate_sampHr = 2.0
intermediate_samples = int(intermediate_time_tosimulate*intermediate_sampHr)


def getAmp(X): 
	amp = (np.max(X) - np.mean(X))/np.mean(X)
	# amp = (np.max(X) - np.mean(X))
	return amp

def getValue_scale_obs(matrix, i, log = False):  
	T = matrix[i]
	amp = getAmp(T)
	f = 1.0
	# if i in (var_output['TSH'], var_output['T4'], var_output['T3'], var_output['FT3'], var_output['T4']):
	# 	f = 1.0
	T1 = T/f
	if log:
		print('getValue_scale_obs', i, observable_names[i], np.mean(T), np.mean(T1))
	return T1, f, amp

def ifSteady(values):   
	j = 0
	amp1 = 0.0
	for i in steady_indices:
		T, f, amp = getValue_scale_obs(values, i)
		ic = T[0] 
		end = T[-1] 
		avg = (ic+end)/2
		amp1 = (max(end,ic)/avg -1) 
		slp = abs(amp1 - amp)
		# print(plot_names[j], 'ifSteady', slp, amp1, amp, ic, end)
		j += 1
		if amp1 > 4e-6: 
			#print('############### not steady #############################')
			return False, amp1
		# elif slp > 0.001:
		#   print('############### not steady #############################')
		#   return False
	# print('############### steady #############################')
	return True, amp1

def simulate(tspan, inputs, hashmap, ic_in = []):   
	# print('-------- tm, t4, t3_th, t3_t4c, t3p, t3z, tsh, tshz, trh_test, trhk, t4_pill, t4_gut, t3_pill, t3_gut, q6, ft4d, t3rd, tshzd, t3pd, q1, q2, q3, q4, q5, tshd, t3p_0, t4_0 -------') 
	# print('simulate initial condition : ', list(ic_in))#, '\n parameters', inputs)
	# print('---------------')
	values, constraints, err = de.calculate_data_3600(inputs, tspan, hashmap, ic_in)
	return values, constraints, False #err

def run(tspan, inputs, hashmap, recalculate = True):    
	# print('-------- run: ', recalculate)
	st = time.time()    

	values, constraints, err = simulate(tspan, inputs, hashmap)
	#steady_values = values
	steady, amp1 = ifSteady(values)
	if recalculate: 
		if err or len(values) == 0:
			ic1 = []
		else:
			ic1 = [values[i][-1] for i in range(0, l_var)]
			ic1[var_output['tm']] = 0.0
			ic1[var_output['q_10']] = 0.0
			ic1[var_output['q_12']] = 0.0
		k = 0
		while not steady:
			tspan1 = np.linspace(0, intermediate_time_tosimulate, intermediate_samples)
			if err or len(values) == 0:
				ic1 = []
			else:
				ic1 = [values[i][-1] for i in range(0, l_var)]
				ic1[var_output['tm']] = 0.0
				ic1[var_output['q_10']] = 0.0
				ic1[var_output['q_12']] = 0.0

			# print(k,steady,'############### run again #############################')
			values, constraints, err = simulate(tspan1, inputs, hashmap, ic1)
			steady, amp1 = ifSteady(values)
			k += 1
			if k > NK:
				break			
		values, constraints, err = simulate(tspan, inputs, hashmap, ic1)
		steady_values = values
		steady, amp1 = ifSteady(values)

	ic1 = [values[i][-1] for i in range(0, l_var)]
	ic1[var_output['tm']] = 0.0
	ic1[var_output['q_10']] = 0.0
	ic1[var_output['q_12']] = 0.0

	# print(steady, "--- %s seconds ---" % (time.time() - st), ic1)
	values, constraints, err = simulate(tspan, inputs, hashmap, ic1)
	steady, amp1 = ifSteady(values)
	# print(' -------- ')
	return values, constraints, steady #, amp1, steady_values

def getODEResult(top, timespace, recalculate = True, log = False): #, transform = TRANSFORM):
	#ind = param_values
	# t = int(t_d*24)
	# tsamples = t*samples_ph
	# timespace=np.linspace(0, t, tsamples)
	# timespace=np.linspace(0, total_time_to_siimulate, total_samples)
	matrix, cnsts, steady  = run(timespace, top, hashmap, recalculate)
	values = [getValue_scale_obs(matrix, i)[0] for i in dose_observable_index]
	# steady_values = [getValue_scale_obs(steady_matrix, i)[0] for i in observed_indices]
	if log:
		print('getODEResult: cap = {0}, steady = {1}'.format(top[-2], steady))
	return matrix, timespace, values  #, steady, steady_matrix, steady_values

def getSIMData(top):
	t = int(2*24)
	tsamples = t*samples_ph
	timespace=np.linspace(0, t, tsamples)
	matrix, timespace, values = getODEResult(top, timespace)
	data = [v[-1] for v in values]
	ic1 = [matrix[i][-1] for i in range(0, l_var)]
	ic1[var_output['tm']] = 0.0
	ic1[var_output['q_10']] = 0.0
	ic1[var_output['q_12']] = 0.0

	# print('getSIMData', data)
	return data, ic1

index = 1
rowCount = 520
st_row = (index-1)*rowCount
end_row = st_row+rowCount
param_index = -1

EPS = 0.1
t1 = []
all_data = []
clusters = []

with open(data_file, mode= 'r') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	header = next(csv_reader)
	print(header, type(header))
	for row in csv_reader:
		if row[0].startswith("c"):
			continue
		else:
			data = {} #int(row[0])]
			for i in range(0, len(row)):
				if row[i] == '':
					continue
				# print(header[i], row[i])
				data.update({header[i]:float(row[i])})
			all_data.append(data)
print(len(all_data), all_data[0])

# import eisen_dose_params as ep
# from ode_util_eisen import *

# hypo_indices = [ep.ind_d_1] #, ep.ind_p_47] #, ep.ind_p_38, ep.ind_p_39, ep.ind_p_23, ep.ind_p_20, ep.ind_p_29, ep.ind_p_6, ep.ind_p_3, ep.ind_p_1, ep.ind_p_19, ep.ind_d_1, ep.ind_p_47] 
#trp.ind_k4, trp.ind_kdelay_4] #, trp.ind_k3, trp.ind_k0, trp.ind_k1, trp.ind_k5, ]
# Default_Params = ep.getdefaultParams()
# dose_all_indices = ep.getParam_indices()
# dose_selectedIndices = [ep.ind_dose_t4, ep.ind_dose_t3, ep.ind_off_t4, ep.ind_off_t3, ep.ind_et4_init, ep.ind_et3_init]
# dose_selectedIndices = [ep.ind_tg_t4, ep.ind_tg_t3, ep.ind_dose_t4, ep.ind_dose_t3, ep.ind_et4_init, ep.ind_et3_init, ep.ind_eps_t4, ep.ind_delta_t4, ep.ind_eps_t3, ep.ind_delta_t3]
#dose_selectedIndices = [ep.ind_dose_t4, ep.ind_dose_t3, ep.ind_et4_init, ep.ind_et3_init, ep.ind_eps_t4, ep.ind_delta_t4, ep.ind_eps_t3, ep.ind_delta_t3]

sensitiveIndices = dose_selectedIndices 
dose_All_indices = dose_selectedIndices+hypo_Indices

hashmap_All = ep.findIndices(dose_All_indices)
lp = len(dose_selectedIndices)
Dose_ParamNames_all = ep.getParam_all()

IndexBound = lp - 3
t_unit1 = 3600.0
t_unit2 = 60.0
t_units = [t_unit1] #, t_unit2]

# print(dose_observable_index)

transiet_time_tosimulate = ep.TT*ep.TU
transient_sampHr = ep.sampHr
transient_samples = int(transiet_time_tosimulate*transient_sampHr)

stable_time_tosimulate = ep.TD*ep.TU
stable_sampHr = ep.sampHr #24/5.0
stable_samples = int(stable_time_tosimulate*stable_sampHr)

penalty_range1 = 00*stable_sampHr
penalty_range2 = (3600/t_unit1)*stable_sampHr
interval_range = (3600/t_unit1)*stable_sampHr
# interval_range = (3600/t_unit1)*stable_sampHr

def run_dose(tspan, inputs, tunit, ic_in = [], recalculate = True):	
	# print('In run, ', recalculate)
	inputs = ep.setOffsetParams(inputs, dose_selectedIndices, dose_All_indices, hashmap_All, LOOP)
	top = []
	for i in range(len(dose_selectedIndices), len(inputs)):
	# for i in range(0, len(hypo_Indices)):
		top.append(inputs[i])
	# print('run_dose', len(top), hashmap, len(inputs))
	values, constraints, err = simulate(tspan, top, hashmap, ic_in) #simulate(tspan, top, hashmap)
	steady_values = values
	if recalculate: 
		steady, amp1 = ifSteady(values)
		if err or len(values) == 0:
			ic1 = []
		else:
			ic1 = [values[i][-1] for i in range(0, l_var)]
			ic1[var_output['tm']] = 0.0
			ic1[var_output['q_10']] = 0.0
			ic1[var_output['q_12']] = 0.0

		k = 0
		tspan1 = np.linspace(0, intermediate_time_tosimulate, intermediate_samples)
		while not steady:
			if err or len(values) == 0:
				ic1 = []
			else:
				ic1 = [values[i][-1] for i in range(0, l_var)]
				ic1[var_output['tm']] = 0.0
				ic1[var_output['q_10']] = 0.0
				ic1[var_output['q_12']] = 0.0

			# print(k,steady,'############### run again #############################')
			# print('ic1', ic1, len(values))
			values, constraints, err = simulate(tspan1, top, hashmap, ic1) 
			#simulate_dose(tspan1, inputs, hashmap_All, ic1) #simulate(tspan1, top, hashmap, ic1)
			steady, amp1 = ifSteady(values)
			k += 1

			if k > NK:
				break			
		values, constraints, err = simulate(tspan, top, hashmap, ic1) 
		#simulate_dose(tspan1, inputs, hashmap_All, ic1) #simulate(tspan1, top, hashmap, ic1)
		steady_values = values
		steady, amp1 = ifSteady(values)


	ic1 = [values[i][-1] for i in range(0, l_var)]
	ic1[var_output['tm']] = 0.0
	#print('steay ic', ic1)
	ic1[var_output['q_10']] = 0.0
	ic1[var_output['q_12']] = 0.0

	if LOOP:
		ic1[var_output['q_10']] += inputs[hashmap_All[ep.ind_dose_t4]]
		ic1[var_output['q_12']] += inputs[hashmap_All[ep.ind_dose_t3]] if ep.ind_dose_t3 in hashmap_All else 0.0

		all_values = {}
		dose_time = 24
		tspan_small = np.linspace(0, dose_time, int(dose_time*stable_sampHr))

		t = 0
		while t < stable_time_tosimulate:
		#top1 = ep.getParams(dose_All_indices, hashmap_All)
			values, constraints, err = simulate_dose(tspan_small, inputs, hashmap_All, ic1)

			for i in range(len(values)):
				if t == 0 or isinstance(values[i], int) or isinstance(values[i], float):
					all_values.update({i:values[i]})
				else:
					val = list(all_values[i])
					for x in values[i]:
						val.append(x)
					all_values.update({i:np.array(val)})

			ic1 = [values[i][-1] for i in range(0, l_var)]
			ic1[var_output['tm']] = 0.0
			
			ic1[var_output['q_10']] += inputs[hashmap_All[ep.ind_dose_t4]]
			ic1[var_output['q_12']] += inputs[hashmap_All[ep.ind_dose_t3]] if ep.ind_dose_t3 in hashmap_All else 0.0

			t += dose_time
		a_values, constraints, err = all_values, constraints, err
	else:

		ic1[var_output['q_10']] = inputs[hashmap_All[ep.ind_et4_init]]
		ic1[var_output['q_12']] = inputs[hashmap_All[ep.ind_et3_init]] if ep.ind_et3_init in hashmap_All else 0.0
		a_values, constraints, err = simulate_dose(tspan, inputs, hashmap_All, ic1)

	return a_values, constraints, err

def simulate_dose(tspan, inputs, hashmap_All, ic_in = []):
	# if tunit == 60.0:		
	# 	values, constraints, err = de2.calculate_data_60(inputs, tspan, hashmap_All, ic_in)
	# else:
		# print(inputs)
	values, constraints, err = de.calculate_data_3600(inputs, tspan, hashmap_All, ic_in)
	return values, constraints, err

def getData(top, hypo_param):
	top_param, eu_obs, cl_cap_id, hypo_obs, eu_sp_obs, hypo_ic = hypo_param
	if len(hypo_param) > 4:
		ic_in = hypo_ic #hypo_param[5]
	else:
		ic_in = []
	inputs = [v for v in top] + [v for v in top_param ] #hypo_param[0]]
	# inputs = ep.setOffsetParams(inputs, dose_selectedIndices, dose_All_indices, hashmap_All)
	timespace = np.linspace(0, stable_time_tosimulate, stable_samples)
	# print('getData - top', inputs)
	values, _, _ = run_dose(timespace, inputs, t_unit1, ic_in=ic_in)
	# print('getData', inputs, len(values), observable_index)
	ob = [values[i][-1] for i in dose_observable_index]
	# T4_c = values[var_output['T4']][-1]
	# T3_c = values[var_output['T3']][-1]
	# TSH_c = values[var_output['TSH']][-1]	
	# FT4_c = values[var_output['FT4']][-1]
	# FT3_c = values[var_output['FT3']][-1]
	# ob = (TSH_c, FT4_c, T3_c, T4_c, FT3_c) #(T4_c, T3_c, TSH_c, FT4_c, FT3_c)
	return ob

def getValue_error(matrix, k, stm, eu_set_Point):
	# w_tsh = 1.0 #0.5
	# w_ft3 = 1.0 #0.3
	# w_ft4 = 1.0 #0.1
	# w_tt3 = 1.0 #0.04
	# w_tt4 = 1.0 #0.06
	w_tsh = 10 #0.9
	w_ft4 = 2
	w_tt3 = 0.1
	w_ft3 = 0.01
	w_tt4 = 0.01
	NORM = False

	def tsh_err(v, a = -1):
		MAX = ep.MAX_tsh
		MIN = ep.MIN_tsh
		l, h = MIN, MAX
		m = l #(l+h)/2
		if a < 0:
			a = (l+h)/2
		# print('TSH expected {0} observed {1}'.format(a, v))
		#e =  w_tsh*(max(h - MAX, 0)/MAX + max(MIN - l, 0)/MIN)
		e =  w_tsh*((v-a)/m)**2 if NORM else w_tsh*((v-a))**2
		return e 

	def tt3_err(v, a = -1):
		MAX = ep.MAX_tt3
		MIN = ep.MIN_tt3
		l, h = MIN, MAX
		m = l #(l+h)/2
		if a < 0:
			a = (l+h)/2
		# print('TT3 expected {0} observed {1}'.format(a, v))
		#e =  w_tt3*(max(h - MAX, 0)/MAX + max(MIN - l, 0)/MIN)
		e =  w_tt3*((v-a)/m)**2 if NORM else w_tt3*((v-a))**2
		return e
		
	def tt4_err(v, a = -1):
		MAX = ep.MAX_tt4
		MIN = ep.MIN_tt4
		l, h = MIN, MAX
		m = l #(l+h)/2
		if a < 0:
			a = (l+h)/2
		# print('TT4 expected {0} observed {1}'.format(a, v))
		#e = w_tt4*(max(h - MAX, 0)/MAX + max(MIN - l, 0)/MIN)
		e =  w_tt4*((v-a)/m)**2 if NORM else w_tt4*((v-a))**2
		return e

	def ft3_err(v,a = -1):
		MAX = ep.MAX_t3
		MIN = ep.MIN_t3
		l, h = MIN, MAX
		m = l #(l+h)/2
		if a < 0:
			a = (l+h)/2
		# print('FT3 expected {0} observed {1}'.format(a, v))
		#e = w_ft3*(max(h - MAX, 0)/MAX + max(MIN - l, 0)/MIN)
		e =  w_ft3*((v-a)/m)**2 if NORM else w_ft3*((v-a))**2
		return e
	
	def ft4_err(v, a = -1):
		MAX = ep.MAX_t4
		MIN = ep.MIN_t4
		l, h = MIN, MAX
		m = l #(l+h)/2
		if a < 0:
			a = (l+h)/2
		m = 0.001
		#return w_ft4*(max(h - MAX, 0)/MAX + max(MIN - l, 0)/MIN)
		e =  w_ft4*((v-a)/m)**2 if NORM else w_ft4*((v-a))**2
		return e
		
	i = dose_observable_index[k]
	T = matrix[i]
	# print(k, i, dose_var_names[k], T)
	#t_max = np.max(T[int(stm):int(stm+4)])
	#t_min = np.min(T[int(stm):int(stm+4)])
	#e =  w_tsh*(max(h - MAX, 0)/MAX + max(MIN - l, 0)/MIN)
	t_cur = T[int(stm)]
	#print(t_min, t_max)
	err = 0
	v = t_cur
	# print(k, len(eu_set_Point))
	a = eu_set_Point[k]
	w = 1.0
	if i == var_output['TSH']:
		err = tsh_err(v, a)
	elif i == var_output['T3']:
		err = tt3_err(v, a)
	elif i == var_output['T4']:		
		err = tt4_err(v, a)
	elif i == var_output['FT3']:		
		err = ft3_err(v, a)
	elif i == var_output['FT4']:
		err = ft4_err(v, a)

	#print('{0} expected {1} calculated {2}'.format(dose_var_names[k], a, v))

	d_err = w*((v-a)/a)
	return err, d_err

range_all = ep.getRanges()
KLow = [] #[0.00001, 0.00001, 0.00001, 0.00001]
KUpp = [] #[1.0, 0.88, 1.0, 0.88]
i = 0
for index in dose_selectedIndices:
	#if i < IndexBound: 
	KLow.append(range_all[index][0])
	KUpp.append(range_all[index][1])
	'''else:
		KLow.append(0.0)
		KUpp.append(0.0)
	i += 1
	'''
print(KLow, KUpp)
   
def getYticks(Y):
	min_Y = min(Y)*0.99
	max_Y = max(Y)*1.01
	# print('getYticks', min_Y, max_Y)
	div = (max_Y-min_Y)/3
	if max_Y-min_Y > 0.0:
		ylocs = np.arange(min_Y, max_Y, div)
		ydivs = []
		for l in ylocs:
			if l < 0.001:
				fm = '{0:1.1e}'
			else:
				fm = '{0:0.4f}'
			ydivs.append(fm.format(l))
		
	else:
		ylocs = [0.0]
		ydivs = ['0.0']
	return ylocs, ydivs

def plot(tspan, values):
	plot_names = ['T4', 'T3', 'TSH', 'FT3', 'FT4', 'eT4', 'eT3', 'edT4', 'edT3'] 

	figs = []
	i = 0
	l = len(plot_names)
	fig = plt.figure()
	for name in plot_names:
		if i % 3 == 0:
			fig = plt.figure()
		plt.subplot(3, 1, (i%3+1))	
		index = var_output[name]
		val = values[index]

		plt.plot(tspan, val, 'b-')
		plt.ylabel(name)
		# print(name)
		ylocs,ylabels = getYticks(val)
		plt.yticks(ylocs,ylabels)
		i += 1

		if i% 3 == 0:
			figs.append(fig)
		#i = 0
	if i%3 != 0:
		figs.append(fig)

	return figs	
	
def plot_sol(inputs, hypo_param, filen='plot_test.pdf'):	
	
	top_param, eu_obs, cl_cap_id, hypo_obs, eu_sp_obs, hypo_ic = hypo_param
	'''
	print('before:', inputs)	
	et4_init, et3_init, e1, d1, e2, d2 = ep.getOffsetParams(inputs, hashmap)
	#inputs1 = [i for i in inputs]
	inputs[4] = et4_init
	inputs[5] = et3_init
	inputs[6] = e1
	inputs[7] = d1
	inputs[8] = e2
	inputs[9] = d2 '''
	#values = setParams(inputs, hashmap)
	#top = ep.getParams(selectedIndices, hashmap, inputs1)
	if len(hypo_param) > 4:
		ic_in = hypo_ic #_param[5]
	else:
		ic_in = []
	top = [v for v in inputs] + [v for v in top_param] #hypo_param[0]]
	#inputs + hypo_param[0] #[x for x in inputs] + [x for x in hypo_param]
	# print('after:', inputs, hashmap)
	timespace = np.linspace(0, stable_time_tosimulate, stable_samples)
	values, _, _ = run_dose(timespace, top, t_unit1, ic_in)
	figs = plot(timespace, values)
	
	'''
	pp = PdfPages(filen)
	for fig in figs:
		pp.savefig(fig)

	pp.close()'''

	return figs

def plot1(tspan, values):
	plot_names = ['TSH', 'T4', 'T3', 'FT4', 'FT3'] #, 'eT4', 'eT3'] 

	figs = []
	i = 0
	l = len(plot_names)
	fig = plt.figure()
	for name in plot_names:
		if i % 3 == 0:
			fig = plt.figure()
		plt.subplot(3, 1, (i%3+1))	
		index = var_output[name]
		val = values[index]
		plt.plot(tspan, val, 'b-')
		plt.ylabel(name)
		# print('plot1', name, np.min(val), np.max(val), np.mean(val))
		ylocs,ylabels = getYticks(val)
		plt.yticks(ylocs,ylabels)
		i += 1

		if i% 3 == 0:
			figs.append(fig)
		#i = 0
	if i%3 != 0:
		figs.append(fig)

	return figs	
	
def plot_sol1(inputs):	
	# print('after:', inputs, hashmap)
	timespace = np.linspace(0, stable_time_tosimulate, stable_samples)
	values, _, _ = getODEResult(inputs, timespace)
	figs = plot1(timespace, values)
	return figs

def run_eisenOpt(hypo_param, option):
	top_param, eu_obs, cl_cap_id, hypo_obs, eu_sp_obs, hypo_ic = hypo_param
	def get_error(inputs, hypo_param):
		top_param, eu_obs, cl_cap_id, hypo_obs, eu_sp_obs, hypo_ic = hypo_param
		# print('before:', inputs, hypo_param[0])
		'''
		et4_init, et3_init, p1, f1, p2, f2 = ep.getOffsetParams(inputs, hashmap)
		#inputs1 = [i for i in inputs]
		inputs[4] = et4_init
		inputs[5] = et3_init
		inputs[6] = p1
		inputs[7] = f1
		inputs[8] = p2
		inputs[9] = f2 '''
		prange1, prange2 = ep.PR1*stable_sampHr, ep.PR2*stable_sampHr
		#sys.stdout.flush()
		if len(hypo_param) > 4:
			ic_in = hypo_ic #hypo_param[5]
		else:
			ic_in = []
		top = [v for v in inputs] + [v for v in top_param] #hypo_param[0]]
		# print('get_error', len(top), top)
		#inputs + hypo_param[0]
		#values = setParams(inputs, hashmap)
		#inputs = ep.getParams(selectedIndices, hashmap, inputs1)
		# print('get_error--after:', inputs, hypo_param[1])
		#(T4_c, T3_c, TSH_c) = getData(inputs)
		#print('@@@@@@@@@@@@', T4_c, T3_c, TSH_c)
		timespace = np.linspace(0, stable_time_tosimulate, stable_samples)
		values, _, _ = run_dose(timespace, top, t_unit1, ic_in=ic_in)
		t_error = 0
		stm = -1 #prange2-1
		j = 0
		#while stm < prange2: #stable_samples - stable_sampHr:
			# p = 0
		stm = -1
		for k in range(len(dose_observable_index)):
			err,_ = getValue_error(values, k, stm, eu_sp_obs) #hypo_param[1])
			# print(stm, 'err for {0} {1}'.format(dose_var_names[k], err))
			t_error += err
			# p +=1 
			j += 1
		#stm += 5
		error = t_error #np.sqrt(t_error/j)
		
		return error

	def eval_fit(inputs):
		error = get_error(inputs, hypo_param)
		return error

	def runOpt(x0, hypo_param):
		top_param, eu_obs, cl_cap_id, hypo_obs, eu_sp_obs, hypo_ic = hypo_param

		# if len(hypo_param) > 4:
		# 	ic_in = hypo_param[5]
		# else:
		# 	ic_in = []
		d_bounds = []
		for i in x0:
			d_bounds.append([i*0.1, i*5.0])
		top, min_err, success, par_errs = opt.fit_Params(eval_fit, init_params = x0, bounds = d_bounds, NGEN = 200, NPOP = 300, manual = option, subop=opt.NELDER)
		gd = getData(top, hypo_param)
		print('getData calculated after: ', [gd[i] for i in range(len(dose_var_names))], top)
		# i = 0
		# for nm in dose_var_names:
		# 	print(nm, gd[i])
		# 	i += 1
		# print('getData calculated after: ', 'TSH_c {0}, FT4_c {1}, T4_c {2}, T3_c {3}, FT3_c {4}'.format(gd[0], gd[1], gd[2], gd[3], gd[4]))
		print('Eu set point: ', eu_sp_obs, eu_obs, 'min_err', min_err) #hypo_param[1])
		figs = [] #plot_sol(top, hypo_param)
		return top, min_err, figs, [top]

	if TEST:
		hypo_wt = 70.0
		###########################
		# print('#############')
		print('TEST - hypo_param', hypo_param)
		sys.stdout.flush()
		#top = ep.getParams(selectedIndices, hashmap)
		#[ep.tg_t4, ep.tg_t3, ep.dose_t4, ep.dose_t3, ep.off_t4, ep.off_t3] #, ep.eps_t4, ep.delta_t4, ep.eps_t3, ep.delta_t3]
		figs0 = plot_sol1(hypo_param[0])
		# top1 = [24, 24, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		top1 = [0, 0, 0.0, 0.0] #0.0, 0.0, 0.0, 0.0]
		gd = getData(top1, hypo_param, hypo_ic)
		figs1 = plot_sol(top1, hypo_param)
		print('before dose', gd)
		d1 = 33/777.0
		d2 = 5/651.0
		# top = [24, 24, d1, d2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #+ hypo_param
		#top = [d1, d2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		top = [d1, 0.0, 0.0, 0.0] #, 0.0, 0.0, 0.0]
		# print(top)
		st = time.time()
		e = get_error(top, hypo_param)
		print('error:', e, top, getData(top, hypo_param))
		print('time taken ', time.time()-st, e)
		# print('#############')
		# fig = plot_sol(top, hypo_param)
		figs2 = plot_sol(top, hypo_param)
		# figs.append(fig1)
		figs = figs0+figs1 + figs2
		return top, e, figs, [top]

	hypo_wt = 70.0

	init_params = [Default_Params[i] for i in dose_selectedIndices]
	init_params[0] = hypo_wt*1.8/777.0
	# print(dose_selectedIndices, init_params)
	top, min_err, fig1, best_sols = runOpt(init_params, hypo_param) #, ic_in)

	# print('-----------------------------')
	#min_err = top.fitness.values[0] #get_error(top)
	# print('solution: {0}, Error: {1}'.format(top, min_err)) #, obs))#, Cf, det))
	#for i in range(len(selectedIndices)):		
	#	pn = ParamNames_all[selectedIndices[i]]
	#	print ("{0} {1:1.6E} : ({2:1.6E} {3:1.6E}))".format(pn, top[i], KLow[i], KUpp[i]))
	return top, min_err, fig1, best_sols

# def getSimulatedValues(top, tspan, tunit):
# 	values, _, _ = run(tspan, top, tunit)
# 	T4 = values[var_output['T4']]
# 	T3 = values[var_output['T3']]
# 	TSH = values[var_output['TSH']]
# 	FT3 = values[var_output['FT3']]
# 	FT4 = values[var_output['FT4']]
# 	ob = (T4, T3, TSH, FT3, FT4)
# 	print('calculated', ob)
# 	print('ft3_scale {0}, ft4_scale {1}'.format(T3[0]/FT3[0], T4[0]/FT4[0]))
# 	print('tsh amplitude {0}, t3 amplitude {1} t4 amplitude {2}'.format(getAmp(TSH), getAmp(T3), getAmp(T4)))
# 	return ob


def getTestData(rowNo = 4):
	print('----- getTestData -----', rowNo)
	VALIDATE = False #rue #eu

	with open(hypo_csv, mode= 'r') as csv_file:
	# with open(mycsv1, mode= 'r') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if row[0].startswith("s"):# or not(row[22] == str(classId)):
				continue
			else:
				line_count += 1
				
			if line_count == rowNo:
				break

	hypo_tsh = float(row[5])
	hypo_ft4 = float(row[6])*0.01 # ng/dL to ug/L
	ht = float(row[2])/100
	wt = float(row[3])

	# nadel's formula
	if 'M' in row[4]:
		BV = 0.3669 * ht**3 + 0.03219 * wt + 0.6041
	else:
		BV = 0.3561 * ht**3 + 0.03308 * wt + 0.1833

	# hypo_PV = BV*0.55 #/1000 
	hypo_PV = BV*0.6 #/1000 
	# hypo_PV = ep.p_47
	# hypo_tt4 = float(row[18])

	fixed_tsh = float(row[8])
	fixed_ft4 = float(row[9])*0.01 # ng/dL to ug/L
	fixed_t3 = 1.4
	hypo_wt = wt
	hypo_ht = ht
	given_dose = float(row[7])

	hypo_obs = [hypo_tsh, hypo_ft4] #, hypo_tt3]
	wt_obs = [1.0, 1.0] #50]
	mean_obs = [1.0, 0.005]
	# x_mean = np.mean(t1, axis = 0)
	# print('x_mean', x_mean)

	with open(sskm_hypo_test_params, 'a+') as wr_csv:
		res_writer = csv.writer(wr_csv, delimiter=',')
		hypo_match_min_err = {}
		# for i in range(len(all_data)):
		row_data = all_data[rowNo-1]
		sub = int(row_data['item'])
		cl_cap_id = float(row_data['d_1'])
		cl_PV = float(row_data['p_47'])
		cl_PV1 = float(row_data['p_48'])
		eu_obs = [fixed_tsh, fixed_ft4] # for given post-treatment  #[row_data[i] for i in ['TSH_eu','FT4_eu']] #eu_data
		eu_sp_obs = [row_data[i] for i in ['TSH_es','FT4_es']] #eu_sp_data -- from goede model
		hypo_obs = [row_data[i] for i in ['TSH_o','FT4_o']] # hypo data
		hypo_c = [row_data[i] for i in ['TSH_c','FT4_c']] # hypo_c data
		top_param =  [float(row_data[selected_Names[i]]) for i in range(len(hypo_Indices))]#[cl_cap_id, cl_PV, cl_PV1] #params : [] + [6, ep1.47], 5 : cluster
		# print('Eu', eu_obs, 'Hypo', hypo_obs, 'cap', cl_cap_id)
		hypo_ic = [row_data[i] for i in ['ic_0','ic_1','ic_2','ic_3','ic_4','ic_5','ic_6','ic_7','ic_8','ic_9','ic_10','ic_11','ic_12','ic_13']]
		 #,'ic_14','ic_15','ic_16','ic_17','ic_18','ic_19']]
		# hypo_param = (param, eu_obs, cl_cap_id, hypo_obs, eu_sp_obs, hypo_ic)
		# hypo_params_all.append(hypo_param)

		hypo_sim_data, hypo_ic = getSIMData(top_param)

		hypo_param = (top_param, eu_obs, cl_cap_id, hypo_obs, eu_sp_obs, hypo_ic)

		chosen = [cl_cap_id, hypo_sim_data, eu_obs, top_param, hypo_ic, eu_sp_obs, hypo_param]
		# l2_norm = np.linalg.norm(np.array(sim_data) - np.array(hypo_obs))
		print(' -- [Sub] {0}, cap {1}, actual {2}, sim {3}, eu setpoint {4}, eu_base {5}'.format(sub, (cl_cap_id, cl_PV, cl_PV1), hypo_obs, hypo_sim_data, eu_sp_obs,  eu_obs))
				
		opt_top, opt_min_err, opt_figs, opt_best_sols = run_eisenOpt(hypo_param, opt.CF)
	
		top1 = [ti for ti in opt_top] + [x for x in top_param] #hypo_param[0]]
		# if not LOOP:
		(res_t4, res_t4_1, res_t3, res_t3_1) = ep.getActualDosages(top1, hashmap_All, LOOP)
		opt_dose = res_t4*777.0


		hypo_param = (top_param, eu_sp_obs, cl_cap_id, hypo_obs, eu_obs, hypo_ic)
		opt_top1, opt_min_err, opt_figs, opt_best_sols = run_eisenOpt(hypo_param, opt.CF)
	
		top1 = [ti for ti in opt_top1] + [x for x in top_param] #hypo_param[0]]
		# if not LOOP:
		(res_t41, res_t4_1, res_t3, res_t3_1) = ep.getActualDosages(top1, hashmap_All, LOOP)
		opt_dose1 = res_t41*777.0
		# err, cl_id, cap_id, sim_data, eu_obs, top_c = chosen
		# cr = [rowNo, cl_id, cap_id, err] + [v for v in sim_data] + [v for v in eu_obs] + [v for v in top_c]
		cr = [rowNo] + [v for v in top_param] + [v for v in hypo_ic] + [given_dose, opt_dose, opt_dose1] + [v for v in hypo_obs] + [v for v in hypo_sim_data] + [v for v in eu_sp_obs] + [v for v in eu_obs] #, cl_id, cap_id, err] + [v for v in sim_data] +  + 
		res_writer.writerow(cr)
		# continue
		print(cr)
		s_tr = '################ row {0} ----\n'.format(rowNo)
		s_tr += '--- top_c' + str(top_param) + '\n'
		s_tr += '--- opt_dose - given post-treatment obs ' + str(opt_top)  +' Actual dose: ' + str(opt_dose) + '\n'
		s_tr += '--- opt_dose - eu setpoint obs ' + str(opt_top1)  +' Actual dose: ' + str(opt_dose1) + '\n'
		print(s_tr)
		# k = 0
		# for k in range(len(hypo_Indices)):
		# 	print('#define {0} {1}'.format(ParamNames_all[hypo_Indices[k]], top_c[k]))
			# k += 1
		print('################ row ended ###############')
	


	# if len(hypo_params_all) <= 0:
	# 	print('---- hypo cluster param did not match ----- exiting ...')	
		# exit()

	# print([hypo_params_all[0]])
	# print('---- hypo cluster param detected -----')
	# print('Part 1', (fixed_tsh, fixed_ft4, fixed_t3), 'Part 2', sim_data[0], sim_data[1], sim_data_1[-1])
	# return hypo_params_all, hypo_match_min_err

def getTrainingData():
	hypo_params_all = []
	for i_all in range(len(all_data)):
		row_data = all_data[i_all]
		# print('getTrainingData', len(row_data), row_data)
		sub = int(row_data['item'])
		cl_cap_id = float(row_data['d_1'])
		cl_PV = float(row_data['p_47'])
		cl_PV1 = float(row_data['p_48'])
		eu_obs = [row_data[i] for i in ['TSH_eu','FT4_eu']] #eu_data
		eu_sp_obs = [row_data[i] for i in ['TSH_es','FT4_es']] #eu_sp_data
		hypo_obs = [row_data[i] for i in ['TSH_o','FT4_o']] # hypo data
		hypo_c = [row_data[i] for i in ['TSH_c','FT4_c']] # hypo_c data
		# param =  [cl_cap_id, cl_PV, cl_PV1] #params : [] + [6, ep1.47], 5 : cluster
		param =  [float(row_data[selected_Names[i]]) for i in range(len(hypo_Indices))]
		# print('Eu', eu_obs, 'Hypo', hypo_obs, 'cap', cl_cap_id)
		hypo_ic = [row_data[i] for i in ['ic_0','ic_1','ic_2','ic_3','ic_4','ic_5','ic_6','ic_7','ic_8','ic_9','ic_10','ic_11','ic_12','ic_13']]#,'ic_14','ic_15','ic_16','ic_17','ic_18','ic_19']]
		hypo_param = (param, eu_obs, cl_cap_id, hypo_obs, eu_sp_obs, hypo_ic)
		hypo_params_all.append(hypo_param)
	return hypo_params_all

not_considered = []
# for i in range(0, 64):
# 	not_considered.append(i)
# for i in range(1000, 1064):
# 	not_considered.append(i)
# for i in range(2000, 2096):
# 	not_considered.append(i)
# for i in range(3000, 3064):
# 	not_considered.append(i)
# for i in range(4000, 4064):
# 	not_considered.append(i)
# for i in range(5000, 5064):
# 	not_considered.append(i)

def getResults(inputs):
	hypo_param, ij = inputs
	print('-------', ij, '-----------')

	top_param, eu_obs, cl_cap_id, hypo_obs, eu_sp_obs, hypo_ic = hypo_param
	# cl_id, cl_cap_id = hypo_param[2], hypo_param[3]
	# cl_sim_data = hypo_param[1]
	# hypo_obs = hypo_param[4]
	# hypo_ic = hypo_param[5]
	if ij in not_considered:
		return []
	else:
		top, min_err, figs, best_sols = run_eisenOpt(hypo_param, opt.CF)
		# all_results.append((top, min_err, figs, best_sols))
		#min_err = top.fitness.values[0] #get_error(top)
		print('solution: {0}, Error: {1}'.format(top, min_err)) #, obs))#, Cf, det))
		
		gd = getData(top, hypo_param)
		print('hypo obs: ',hypo_obs)
		print('getData after: ', gd)
		print('Eu set point: ', eu_sp_obs, eu_obs)
		results = top, min_err, hypo_param, gd
		
		print('------------------')
		sys.stdout.flush()
		return results #_for_row


# main(st_row, end_row)
with open(sskm_hypo_test_params, 'w+') as wr_csv:
	res_writer = csv.writer(wr_csv, delimiter=',')
	# cr = ['rn', 'cl_id', 'cap_id', 'err'] + [v for v in dose_var_names] + [v for v in dose_var_names] + [ParamNames_all[v] for v in hypo_Indices]
	cr = ['row'] + selected_Names + ['ic_{0}'.format(i) for i in range(0, l_var)]+ ['LT4', 'opt_dose', 'opt_dose1'] + ['hypo_obs_'+v for v in dose_var_names] +  ['hypo_c_'+v for v in dose_var_names] + ['eu_sp_'+v for v in dose_var_names] +  ['eu_base_'+v for v in dose_var_names]
	#[rowNo] + [v for v in top_c] + [v for v in hypo_ic] + [given_dose, opt_dose] + [v for v in hypo_obs] + [v for v in hypo_sim_data] + [v for v in eu_sp_obs] + [v for v in eu_obs]
	res_writer.writerow(cr)

for i in range(1, 20+1):
	getTestData(i)
	
exit()
