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
TEST = True
IFPOOL = False #True
TEST = False
PLOT = True
LOG = False
# LOG = True
ENTROPY = True
GPC = False
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

result_file = 'sskm/eisen_sskm_hypo_070723.csv' #.format(index)
plotFileName = 'plot_dosage.pdf'
sskm_data_csv = 'sskm/sskm_serialised_dist.csv'

l_var = 14
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

# selectedIndices = [ep.ind_p_40, ep.ind_p_37, ep.ind_p_38, ep.ind_p_39, ep.ind_p_23, ep.ind_p_20, ep.ind_p_29, ep.ind_p_6, ep.ind_p_3, ep.ind_p_1, ep.ind_p_19] #trp.ind_k4, trp.ind_kdelay_4] #, trp.ind_k3, trp.ind_k0, trp.ind_k1, trp.ind_k5, ]
#selectedIndices = [trp.ind_p_40, trp.ind_p_37, trp.ind_p_38, trp.ind_p_39, trp.ind_p_23, trp.ind_p_20, trp.ind_p_29, trp.ind_p_6, trp.ind_p_3, trp.ind_p_1, trp.ind_p_19, trp.ind_p_7, trp.ind_p_24]
selectedIndices = [ep.ind_d_1,  ep.ind_p_1, ep.ind_p_19, ep.ind_p_7, ep.ind_p_40, ep.ind_p_10, ep.ind_p_37, ep.ind_p_13] #ep.ind_p_38, ep.ind_p_39, ep.ind_p_7, ep.ind_p_40, ep.ind_p_10, ep.ind_p_37, ep.ind_] # ep.ind_p_1, ep.ind_p_19, ep.ind_p_37]
hypo_Indices = selectedIndices + [ep.ind_p_47, ep.ind_p_48] #, ep.ind_p_38, ep.ind_p_39]
hashmap = ep.findIndices(hypo_Indices)
selected_hashMap = ep.findIndices(selectedIndices)
selected_Names = [ParamNames_all[i] for i in hypo_Indices]
len_selected = len(selected_Names)

estimated_Names = [ParamNames_all[i] for i in selectedIndices]
len_estimated = len(estimated_Names)

KLow = []
KUpp = []
ParamNames = []
def_Params = []
for index in selectedIndices:
	#print(index, Param)
	KLow.append(range_all[index][0])
	KUpp.append(range_all[index][1])
	ParamNames.append(ParamNames_all[index])
	def_Params.append(Default_Params[index])

print('#### hashmap #########')
for key in hashmap.keys():
	print(key, ParamNames_all[key], hashmap[key])
print('########################')
for i in range(len(ParamNames)):
	print(ParamNames[i], KLow[i], KUpp[i])
print('########################')

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
var_output = trh_ode_util.generate_ODE_file(ode_models, selectedIndices, selectedIndices, importFile = 'eisen_dose_params', trp=ep, diffFile = "diffEqn_eisen.py")
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
sample_scaling_factor = 5.0
time_unit_chosen = 3600.0
total_samples = int(total_time_to_siimulate*sample_per_hr*sample_scaling_factor)
sample_after_30_mins = int(0.5*sample_per_hr*sample_scaling_factor)
samples_ph = 10

transiet_time_tosimulate = int(100*24)
transient_sampHr = 0.1
transient_samples = int(transiet_time_tosimulate*transient_sampHr)

intermediate_time_tosimulate = int(5*24)
intermediate_sampHr = 10.0 #2
intermediate_samples = int(intermediate_time_tosimulate*intermediate_sampHr)


# IndexBound = lp - 3
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

def ifSteady(values, sph = samples_ph):   
	j = 0
	amp1 = 0.0
	for i in steady_indices:
		T, f, amp = getValue_scale_obs(values, i)
		ic = T[0] 
		end = T[-1] 
		avg = (ic+end)/2
		amp1 = (max(end,ic)/avg -1) 
		slp = abs(amp1 - amp)
		slp = abs(T[int(-1-24*sph)] - T[-1])
		# print(i, 'ifSteady', slp, amp1, amp, ic, end)
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

def run(tspan, inputs, hashmap, recalculate = False):    
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
			steady, amp1 = ifSteady(values, intermediate_sampHr)
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

def run_dose(tspan, inputs, hashmap, dose = 0, recalculate = False):    
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
	#print('steay ic', ic1)
	ic1[var_output['q_10']] = 0.0
	ic1[var_output['q_12']] = 0.0

	if dose > 0:
		ic1[var_output['q_10']] += dose
		ic1[var_output['q_12']] +=  0.0

		all_values = {}
		dose_time = 24
		tspan_small = np.linspace(0, dose_time, int(dose_time*stable_sampHr))

		t = 0
		while t < stable_time_tosimulate:
			values, constraints, err = simulate(tspan_small, inputs, hashmap, ic1)

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
			
			ic1[var_output['q_10']] += dose
			ic1[var_output['q_12']] +=  0.0

			t += dose_time
		a_values, constraints, err = all_values, constraints, err
	else:

		# print(steady, "--- %s seconds ---" % (time.time() - st), ic1)
		a_values, constraints, err = simulate(tspan, inputs, hashmap, ic1)
		steady, amp1 = ifSteady(values)
		# print(' -------- ')
	return a_values, constraints, steady #, amp1, steady_values


def getSIMData(top, timespace = []):
	if len(timespace) == 0:
		t = int(2*24)
		tsamples = t*samples_ph
		timespace=np.linspace(0, t, tsamples)
	matrix, cnsts, steady  = run(timespace, top, hashmap)
	# data = [v[-1] for v in values]
	ic1 = [matrix[i][-1] for i in range(0, l_var)]
	ic1[var_output['tm']] = 0.0
	ic1[var_output['q_10']] = 0.0
	ic1[var_output['q_12']] = 0.0

	# print('getSIMData', data)
	return matrix, ic1

def getData(top, hypo_param):
	# ic_in = []
	inputs = [v for v in top] + [v for v in hypo_param[0]]
	# print(inputs)
	timespace = np.linspace(0, stable_time_tosimulate, stable_samples)
	# print('getData - top', inputs)
	#values, _, _ = run(timespace, inputs, hashmap) #, ic_in=ic_in)
	matrix, ic1 = getSIMData(inputs, timespace)
	# print('getData', inputs, len(values), observable_index)
	ob = [matrix[i][-1] for i in dose_observable_index]
	# T4_c = values[var_output['T4']][-1]
	# T3_c = values[var_output['T3']][-1]
	# TSH_c = values[var_output['TSH']][-1]	
	# FT4_c = values[var_output['FT4']][-1]
	# FT3_c = values[var_output['FT3']][-1]
	# ob = (TSH_c, FT4_c, T3_c, T4_c, FT3_c) #(T4_c, T3_c, TSH_c, FT4_c, FT3_c)
	return ob, ic1

def getDoseData(top, hypo_param, d1):
	inputs = [v for v in top] + [v for v in hypo_param[0]]
	# if len(hypo_param) > 4:
	# 	ic_in = hypo_ic #hypo_param[5]
	# else:
	ic_in = []
	timespace = np.linspace(0, stable_time_tosimulate, stable_samples)
	# print('getData - top', inputs)
	matrix, _, _ = run_dose(timespace, inputs, hashmap, dose = d1)# ic_in=ic_in)
	# values = [getValue_scale_obs(matrix, i)[0] for i in observable_indices]
	# print('getData', inputs, len(values), observable_index)
	ob = [matrix[i][-1] for i in dose_observable_index]
	# T4_c = values[var_output['T4']][-1]
	# T3_c = values[var_output['T3']][-1]
	# TSH_c = values[var_output['TSH']][-1]	
	# FT4_c = values[var_output['FT4']][-1]
	# FT3_c = values[var_output['FT3']][-1]
	# ob = (TSH_c, FT4_c, T3_c, T4_c, FT3_c) #(T4_c, T3_c, TSH_c, FT4_c, FT3_c)
	return ob

def getValue_error(matrix, k, stm, target):
	# w_tsh = 1.0 #0.5
	# w_ft3 = 1.0 #0.3
	# w_ft4 = 1.0 #0.1
	# w_tt3 = 1.0 #0.04
	# w_tt4 = 1.0 #0.06
	w_tsh = 10 #0.9
	w_ft4 = 2 #for paper #2 before thesis 
	w_tt3 = 0.1
	w_ft3 = 0.01
	w_tt4 = 0.01
	NORM = True #False

	def tsh_err(v, a = -1):
		MAX = ep.MAX_tsh
		MIN = ep.MIN_tsh
		l, h = MIN, MAX
		m = l #(l+h)/2
		if a < 0:
			a = (l+h)/2
		# print('TSH expected {0} observed {1}'.format(a, v))
		# e =  w_tsh*(max(h - MAX, 0)/MAX + max(MIN - l, 0)/MIN)
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
	a = -1 #eu_set_Point[k]
	if len(target) > k:
		a = target[k]
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
	
def plot_sol(inputs, hypo_param):	
	# print('after:', inputs, hashmap)
	timespace = np.linspace(0, stable_time_tosimulate, stable_samples)
	
	values, cnsts, steady  = run(timespace, inputs, hashmap)
	figs1 = plot(timespace, values)

	values1, cnsts, steady  = run_dose(timespace, inputs, hashmap, hypo_param[2]/777.0)
	figs2 = plot(timespace, values1)
	return figs1+figs2

def run_eisenOpt(rowNo, hypo_param, option):
	
	def get_error(inputs, hypo_param):
		# print('before:', inputs, hypo_param[0])

		timespace = np.linspace(0, stable_time_tosimulate, stable_samples)
		prange1, prange2 = ep.PR1*stable_sampHr, ep.PR2*stable_sampHr
		step = ep.TU*stable_sampHr/2

		#sys.stdout.flush()
		# ic_in = []
		top1 = [v for v in inputs] + [v for v in hypo_param[0]]
		top1[0] = 1.0
		# print('get_error', len(top), top)
		# print('get_error--after:', inputs, hypo_param[1])
		#(T4_c, T3_c, TSH_c) = getData(inputs)
		#print('@@@@@@@@@@@@', T4_c, T3_c, TSH_c)
		values, _, _ = run(timespace, top1, hashmap) #, ic_in=ic_in)
		t_error = 0
		stm = prange1
		j = 0
		while stm < prange2: #stable_samples - stable_sampHr:
			# p = 0
			for k in range(len(dose_observable_index)):
				err,_ = getValue_error(values, k, stm, hypo_param[-1])
				# print(stm, 'err for {0} {1}'.format(dose_var_names[k], err))
				t_error += err
				# p +=1 
			j += 1
			stm += step
		error1 = 0.2*t_error/j #np.sqrt(t_error/j)

		top = [v for v in inputs] + [v for v in hypo_param[0]]
		# print('get_error', len(top), top)
		# print('get_error--after:', inputs, hypo_param[1])
		#(T4_c, T3_c, TSH_c) = getData(inputs)
		#print('@@@@@@@@@@@@', T4_c, T3_c, TSH_c)
		values, _, _ = run(timespace, top, hashmap) #, ic_in=ic_in)
		t_error = 0
		stm = prange1
		j = 0
		while stm < prange2: #stable_samples - stable_sampHr:
			# p = 0
			for k in range(len(dose_observable_index)):
				err,_ = getValue_error(values, k, stm, hypo_param[1])
				# print(stm, 'err for {0} {1}'.format(dose_var_names[k], err))
				t_error += err
				# p +=1 
			j += 1
			stm += step
		error2 = 0.4*t_error/j #np.sqrt(t_error/j)

		dose = hypo_param[2]/777.0 #, 0.0, 0.0, 0.0]
		# top = [v for v in inputs] + [v for v in hypo_param[0]]

		values, _, _ = run_dose(timespace, top, hashmap, dose)
		t_error = 0
		stm = prange1
		j = 0
		while stm < prange2: #stable_samples - stable_sampHr:
			# p = 0
			for k in range(len(dose_observable_index)):
				err,_ = getValue_error(values, k, stm, hypo_param[-2])
				# print(stm, 'err for {0} {1}'.format(dose_var_names[k], err))
				t_error += err
				# p +=1 
			j += 1
			stm += step
		error3 = 0.4*t_error/j #np.sqrt(t_error/j)

		const_error = 0
		for i in range(len(selectedIndices)):
			const_error += ((inputs[i] - def_Params[i])/def_Params[i])**2

		error = error1 + error2 + error3 + const_error
		return error

	def eval_fit(inputs):
		error = get_error(inputs, hypo_param)
		return error

	def runOpt(x0, hypo_param):

		# if len(hypo_param) > 4:
		# 	ic_in = hypo_param[5]
		# else:
		# ic_in = []
		d_bounds = []
		for i in x0:
			if i > 0:
				d_bounds.append([i*0.5, i*1.5])
			else:
				d_bounds.append([i*1.5, i*0.5])

		d_bounds[0] = [0.00001, 0.1]
		# d_bounds[1] = [x0[1]*0.75, x0[1]*1.25]
		# d_bounds[2] = [x0[2]*0.75, x0[2]*1.25]

		print('bounds', d_bounds)
		top, min_err, success, par_errs = opt.fit_Params(eval_fit, init_params = x0, bounds = d_bounds, NGEN = 200, NPOP = 300, manual = option, subop=opt.NELDER)
		# top, min_err, success, par_errs = opt.fit_Params(eval_fit, init_params = x0, bounds = d_bounds, NGEN = 200, NPOP = 300, manual = opt.LS, subop=opt.LS)
		gd, ic1 = getData(top, hypo_param)
		print('getData estimated param', top , 'hypo ', gd)
		eu_top = [tt for tt in top]
		eu_top[0] = 1.0
		eugd, eu_ic1 = getData(eu_top, hypo_param)

		dose_gd = getDoseData(top, hypo_param, hypo_param[2]/777.0)
		# i = 0
		# for nm in dose_var_names:
		# 	print(nm, gd[i])
		# 	i += 1
		# print('getData calculated after: ', 'TSH_c {0}, FT4_c {1}, T4_c {2}, T3_c {3}, FT3_c {4}'.format(gd[0], gd[1], gd[2], gd[3], gd[4]))
		print('hypo obs: ', hypo_param[1], 'simulated', gd)
		print('Eu obs', hypo_param[-1], 'simulated', eugd)
		print('After dose', hypo_param[2], hypo_param[2]/777.0, 'obs', hypo_param[-2], 'simulated', dose_gd)
		figs = plot_sol(top, hypo_param)
		return top, min_err, gd, ic1, figs

	init_params = [Default_Params[i] for i in selectedIndices]
	init_params[0] = 0.1
	# print(dose_selectedIndices, init_params)
	top, min_err, gd, ic1, figs = runOpt(init_params, hypo_param) #, ic_in)
	sys.stdout.flush()
	pp = PdfPages("eisen_hypo_param_{0}.pdf".format(rowNo))
	for ff in figs:
		pp.savefig(ff)
	pp.close()
	# print('-----------------------------')
	#min_err = top.fitness.values[0] #get_error(top)
	# print('solution: {0}, Error: {1}'.format(top, min_err)) #, obs))#, Cf, det))
	#for i in range(len(selectedIndices)):		
	#	pn = ParamNames_all[selectedIndices[i]]
	#	print ("{0} {1:1.6E} : ({2:1.6E} {3:1.6E}))".format(pn, top[i], KLow[i], KUpp[i]))
	return top, min_err, gd, ic1

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


def getTestData(rowNo=4):
	print('----- For test data  ---- ', rowNo)
	with open(sskm_data_csv, mode= 'r') as csv_file:
	# with open(mycsv1, mode= 'r') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = -1
		for row in csv_reader:
			if row[0].startswith("s"):# or not(row[22] == str(classId)):
				continue
			else:
				line_count += 1
				
			if line_count == rowNo:
				break

	hypo_tsh = float(row[5])
	# ng/dL -- pmol/L * 12.87
	hypo_ft4 = ep.FT4conv2pmol(float(row[6])) #*12.871 #*100
	ht = float(row[2])/100
	wt = float(row[3])

	# nadel's formula
	if 'M' in row[4]:
		BV = 0.3669 * ht**3 + 0.03219 * wt + 0.6041
	else:
		BV = 0.3561 * ht**3 + 0.03308 * wt + 0.1833

	# hypo_PV = BV*0.55 #/1000 
	hypo_PV = BV*0.6 #/1000 
	#hypo_PV = ep.p_47
	p48 = 5.2 + (hypo_PV - 3.2) #ep.p_48 #wt*ep.p_48/70
	# hypo_tt4 = float(row[18])

	fixed_tsh = float(row[8])
	# ng/dL -- pmol/L * 12.87
	fixed_ft4 = ep.FT4conv2pmol(float(row[9]))#*12.871*100
	fixed_t3 = 1.4
	hypo_wt = wt
	hypo_ht = ht
	given_dose = float(row[7])
	
	# fixed_tsh = 1.5
	# fixed_ft4 = ep.FT4conv2pmol(100*(ep.MAX_t4+ep.MIN_t4)/2)#1.2*0.01

	print('getTestData', 'PV', hypo_PV, 'VTSH', p48, 'hypo data', hypo_tsh, hypo_ft4,  'fixed data', fixed_tsh, fixed_ft4 )

	phi = np.log(fixed_tsh/hypo_tsh)/((hypo_ft4 - fixed_ft4))
	s = hypo_tsh * np.exp(phi *hypo_ft4)

	print('getTestData', 'phi', phi, 's', s)

	eu_sp_tsh = 1/(phi * np.sqrt(2))
	eu_sp_ft4 = np.log(phi * s * np.sqrt(2))/phi

	hypo_obs = [hypo_tsh, ep.FT4conv2ug(hypo_ft4)] #, hypo_tt3]
	fixed_obs = [fixed_tsh, ep.FT4conv2ug(fixed_ft4)]
	eu_sp_obs = [eu_sp_tsh, ep.FT4conv2ug(eu_sp_ft4)]
	# x_mean = np.mean(t1, axis = 0)
	print('eu_sp_obs', eu_sp_obs, 'fixed_obs', fixed_obs, 'hypo_obs', hypo_obs)
	sys.stdout.flush()

	hypo_param = [[hypo_PV, p48], hypo_obs, given_dose, fixed_obs, eu_sp_obs]
	top, min_err, gd, ic1 = run_eisenOpt(rowNo, hypo_param, opt.CF)

	return top, min_err, hypo_param, gd, ic1


def main(st_r, en_r):
	NGEN, NPOP = 200, 300
	#pp = PdfPages(plotFileName)
	#population = toolbox.population(n=population_size)
	# result_dosages = []
	# all_results = []
	# hypo_min_max_range = {}
	hypo_params_all = list(range(st_r, en_r+1))
	mcpu = int(multiprocessing.cpu_count()*5.0/6)
	ik = st_r
	data_row = ['item']+selected_Names+['LT4', 'min_err', 'TSH_o', 'FT4_o', 'TSH_c', 'FT4_c', 'TSH_eu', 'FT4_eu', 'TSH_es', 'FT4_es'] + ['ic_{0}'.format(i) for i in range(l_var)]	
	with open(result_file, 'w+') as fp:
		for dr in data_row:
			fp.write(str(dr)+',')
		fp.write('\n')

	while ik < en_r:
		print('######### {0}/{1} ################## OPT for rows ################################'.format(ik, en_r))
		sys.stdout.flush()
		# hypo_param = hypo_params_all[ik]
		num_proc = min(len(hypo_params_all[ik:en_r]), mcpu) # last remainder may be less than batch size
		POOL = True if num_proc > 1 else False
		#num_proc = 1 # min(np1, stack.size())
		POOL = False #True if num_proc > 1 else False
		print('Using multiprocessing : '+ str(POOL)+ ', '+ str(num_proc) + ' Boxes left --' + str(en_r-ik), ik, ik+num_proc)

		all_params = [(hypo_params_all[j]) for j in range(ik, ik+num_proc)]       
		# print(allboxes)
		if POOL:
			pool = ProcessingPool(num_proc)
			inputs = [all_params[j] for j in range(num_proc)]
			#print(inputs)
			results = pool.map(getTestData, inputs)
		else:
			results = [getTestData(all_params[j]) for j in range(num_proc)]

		if POOL:
			pool.close()
			pool.join()
			pool.clear()

		# print('results', results, num_proc)
		for j in range(num_proc):
			ij = all_params[j]
			top, min_err, hypo_param, gd, ic1 = results[j]
			hypo_PV, hypo_obs, given_dose, fixed_obs, eu_sp_obs  = hypo_param

			print('hypo_PV', hypo_PV,  'hypo_obs', hypo_obs, 'given_dose', given_dose, 'fixed_obs', fixed_obs, 'eu_sp_obs', eu_sp_obs, 'min_err', min_err )
			cap = top[0]
			err = 0
			if isinstance(min_err, list):
				err = min_err[0]
			elif isinstance(min_err, int) or isinstance(min_err, float):
				err = min_err
			result_for_row = [ij]+ [v for v in top] + [hypo_PV[0], hypo_PV[1], given_dose, err] + [hp for hp in hypo_obs]+ [g for g in gd] +\
			 [fo for fo in fixed_obs] + [fo for fo in eu_sp_obs] + [c for c in ic1]
			print(ij, 'result_for_row', result_for_row)
			
			with open(result_file, 'a+') as fp:
				for rd in result_for_row:
					fp.write(str(rd)+',')
				fp.write('\n')
		ik = ik + num_proc
		print('########################### OPT ended ################################')

	
main(0, 20)

