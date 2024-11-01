
from __future__ import print_function
PLOT = False 

if PLOT:
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_pdf import PdfPages
from math import *
import numpy as np
from scipy.integrate import ode, odeint
import os
import subprocess
import re
import sys, getopt
import csv


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.haModel import *
from model.phaModel import *
from model.ha_factory import *
from model.node import *
from parser.parseSTL import *
from parser.parseParameters import *
from parser.parseEquations import *

# def getBox(params):
# 	edges = {}
# 	for par in params:
# 		rng = params[par]
# 		left = rng.leftVal()
# 		right = rng.rightVal()
# 		it = PyInterval(left, right)
# 		#it.mark()
# 		edges.update({par: it})	
		
# 	sbox = Box(edges)
# 	return sbox

# deterministic simulation of HA
def main(argv):
	try:
		opts, args = getopt.getopt(argv,"hi:d:o:",["ifile=","delta=","outputs="])
	except getopt.GetoptError:
			print("HA2ode.py -i <inputfile> -d <deltaprecision> -o <outputs>")
			sys.exit(2)
			
	for opt, arg in opts:
		if opt == '-h':
			print("HA2ode.py -i <inputfile> -d <deltaprecision> -o <outputs>")
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
			print("Input file is :" + inputfile)
		elif opt in ("-d", "--delta"):
			delta = arg
			print("precision is :" + delta)
		elif opt in ("-o", "--outputs"):
			outputfile = arg
			print("outputs file is :" + outputfile)

	datafile = inputfile.split('.')[0]+'.csv'
	input_name = inputfile.split('/')[-1]
	# simulateFile = 'simulateHA_'+input_name.split('.')[0]+'.py'


	simulateFile = 'simulateHA.py'
	createSimulator(datafile, inputfile, outputfile, simulateFile, delta)

def createSimulator(indatafile, inputfile, outputfile, simulateFile, delta):

	datafile = inputfile.split('.')[0]+'.csv'

	outputEqns = getEquationsFile(outputfile)
	'''detect parameters'''
	for var in outputEqns:
		print(var + ' : '+ str(outputEqns[var]))

	# read a HA
	ha1 = getModel(inputfile)
	ha = ha1.simplify(ONLY = False)
	print('model parsed')
	print('############ simplified HA ############')
	print(ha)
	print('############')

	inits = {}
	for c in ha.init.condition:
		# print(str(c.literal1), str(c.literal2))
		inits.update({str(c.literal1):str(c.literal2)})

	
	sim_ha = 'from math import *\n'+\
			'import numpy as np\n'+\
			'from scipy.integrate import ode, odeint\n'+\
			'import sys, os\n'
	if PLOT:
		sim_ha += 'import matplotlib.pyplot as plt\n'+\
			'from matplotlib.backends.backend_pdf import PdfPages\n'

	observable_data = []
	observed_timespace = []
	
	obs_data_dict = {}

	with open(indatafile) as fp:
		fr = csv.reader(fp, delimiter=',')
		for row in fr:
			data = [float(row[i]) for i in range(len(row))]
			tm = data[0]
			observed_timespace.append(tm)
			i = 1
			for key in outputEqns.keys():
				if i < len(data) :
					if key in obs_data_dict:
						dk = obs_data_dict[key]
					else:
						dk = []
					dk.append(data[i])
					obs_data_dict.update({key:dk})
				i += 1
	for key in obs_data_dict:
		kl = obs_data_dict[key]
		observable_data.append(kl)

	print(observable_data, observed_timespace)

	sim_ha += 'observable_data = {0}\n'.format(observable_data)
	sim_ha += 'observed_timespace = {0}\n\n'.format(observed_timespace)

	with open('ha2ode/FIM_params.py', 'r') as f:
		s = f.read()

	sim_ha += s+'\n\n'+\
			'atol = {0}\n'.format(delta)

	# macros_updated = {}
	macros = ''
	for key in ha.macros.keys():
		if(key is None):
			continue
		macros += str(key)+ ' = ' + str(ha.macros[key]).format('^', '**')+"\n"
		# macros_updated.update()

	print('macros:', macros)
	#sim_ha += macros
	sim_ha += str('time')+ ' = ' + str(ha.variables['time'].right)+"\n"
	# get list of variables
	var_indices = {}
	ind_vars = {}
	default_value_ranges = {}
	param_indices = {}
	variables = '' #'['
	parameters = ''
	params = {}
	k = 0
	j = 0
	for var in ha.variables.keys():
		if var == 'time':
			continue
		var_indices.update({k:var})	
		ind_vars.update({var:k})

		rng = (str(ha.variables[var].left), str(ha.variables[var].right))
		default_value_ranges.update({var:rng})	

		if var in params.keys() or var not in inits.keys():
			param_indices.update({k:j})
			if j == 0:
				parameters += str(var)
			else:
				parameters += ', ' + str(var)
			j += 1
			params.update({var:rng})

		if k == 0:
			variables += str(var)
		else:
			variables += ', ' + str(var) 
		k +=1
	# variables += ']'
	# define function for each mode
	states = ha.states

	# sbox = getBox(params)

	'''func(mode, values){
		if mode == 1:
		if mode == 2:
		....	
	}'''

	# set the macros
	# macros = ''
	# for key in ha.macros.keys():
	# 	if(key is None):
	# 		continue
	# 	macros += '\t'+str(key)+ ' = ' + str(ha.macros[key]).format('^', '**')+"\n"
	sim_ha += 'def func(q0, t, mode):'+'\n'
	sim_ha += '\t'+ variables + ' = q0'+'\n'
	sim_ha += macros
	sim_ha += '\t'+'qdot = np.zeros('+str(k)+')'+'\n'
	for state in states:
		mode = state.mode
		ode_exprs = state.flow
		#jumps = state.jumps
		#invt = state.invariants

		# assign values and variables
		# x, v, tm, mode = q0 
		sim_ha += '\t'+'if mode == '+ str(mode)+ ':'+'\n'
		for eqn in ode_exprs:
			sim_ha += "\t\tqdot[{0}] = 1.0*{1}\n".format(ind_vars[eqn.var], str(eqn.expr).replace('^', '**')) 
	sim_ha += '\treturn qdot\n\n'

	# define a condition function to return the next mode, reset values
	'''check_condition(values, pre_mode){
		if mode == 1:
			return next_mode, reset
		}'''
	#[todo: zero crosiing detection]	

	# macros = ''
	# for key in ha.macros.keys():
	# 	if(key is None):
	# 		continue
	# 	macros += '\t'+str(key)+ ' = ' + str(ha.macros[key]).format('^', '**')+"\n"

	sim_ha += 'def check_condition(q0, mode):'+'\n'+\
			'\t#print(\'##\', q0, len(q0))\n'
	sim_ha += '\t'+ variables + ' = q0'+'\n'
	# sim_ha += macros
	for state in states:
		mode = state.mode
		# ode_exprs = state.flow
		jumps = state.jumps
		invt = state.invariants
		# sim_ha += '\t'+'if mode == '+ str(mode)+ ':'+'\n'	
		invariants = '('
		i = 0
		for c in invt:
			if i == 0:
				invariants += str(c)
			else:	
				invariants += ' and '+str(c)
			i += 1

		invariants += ')'
		if len(jumps) > 0:
			sim_ha += '\t'+'if  mode == '+ str(mode)+ ' :'+'\n' #+ ' and '+ invariants
		for jmp in jumps:
			event = '['
			grd = '('
			i = 0
			for c in jmp.guard:
				if c.binop == '>=' or c.binop == '>': # or c.binop == '=':
					ev = '({0} - {1}+atol)'.format(c.literal1, c.literal2)
				else:
					ev = '({1} - {0}+atol)'.format(c.literal1, c.literal2)

				if i == 0:
					grd += str(c).replace('=','==').replace('>==','>=').replace('<==','<=')
					event += ev
				else:	
					grd += 'and '+str(c).replace('=','==').replace('>==','>=').replace('<==','<=')
					event += ', ' + ev
				i += 1			
			grd += ')'
			event += ']'

			var_included = []
			# var_not_included = []
			reset = {}
			for r in jmp.reset:
				if r.var:
					reset.update({r.var:r.expr})
					var_included.append(r.var)
			for var in ind_vars.keys():
				if var not in var_included:
					# var_not_included.append(var)
					reset.update({var:var})
			tomode = jmp.toMode
			r_reset = str(tomode)+' , ('
			for i in range(k):
				if i == 0:
					r_reset += '{0}'.format(reset[var_indices[i]])
				else:
					r_reset += ', {0}'.format(reset[var_indices[i]])
				# i+= 1
			r_reset += ')'
			# sim_ha += '\t'+'
			sim_ha += '\t\t'+'if '+ grd + ':\n'
			sim_ha += '\t\t\t'+'return {0}\n'.format(r_reset)
			# sim_ha += '\t\t\t'+'return {0}, {1}\n'.format(r_reset, event)
	r_reset = 'mode, ('
	for i in range(k):
		if i == 0:
			r_reset += '{0}'.format(var_indices[i])
		else:
			r_reset += ', {0}'.format(var_indices[i])
		# i+= 1
	r_reset += ')'
	sim_ha += '\t'+'return {0}\n'.format(r_reset)
	sim_ha += '\n'

	sim_ha += 'def jumpEvents(mode, values):'+'\n'+\
			'\t#print(\'##\', values, len(values))\n'
	sim_ha += '\t'+ variables + ' = values'+'\n'
	# sim_ha += macros
	for state in states:
		mode = state.mode
		# ode_exprs = state.flow
		jumps = state.jumps		
		if len(jumps)> 0:
			sim_ha += '\tif  mode == '+ str(mode)+ ' :'+'\n' #+ ' and '+ invariants
		for jmp in jumps:

			event = '['
			# grd = '('
			i = 0
			for c in jmp.guard:
				if c.binop == '>=' or c.binop == '>': # or c.binop == '=':
					ev = '(({0} - {1})+atol)'.format(c.literal1, c.literal2)
				else:
					ev = '(({1} - {0})+atol)'.format(c.literal1, c.literal2)

				if i == 0:
					event += ev
				else:	
					event += ', ' + ev
				i += 1			
			# grd += ')'
			event += ']'
			tomode = jmp.toMode
			sim_ha += '\t\t'+'return {0}, {1}\n'.format(event, tomode)

	rr = '[], mode'
	sim_ha += '\treturn {0}\n\n'.format(rr)

	# simulatefunc(initial_values, time){
	#	pre_mode = start_mode
	#	pre_values = other initial conditions
	#	trace = []
	#	for t in time: # smaller time unit
	#		values = func(pre_mode, pre_values)
	#		mode, reset_values = check_condition(values, pre_mode)
	#		trace.append(values + reset_values)
	#		pre_mode = mode, pre_values = values
	#	return trace
	#}

	sim_ha += 'def eventCondition(events):\n'+\
			'\tcond = 1.0\n'+\
			'\tfor e in events:\n'+\
			'\t\t#print(\'eventCondition\', e)\n'+\
			'\t\tcond = cond * e\n'+\
			'\treturn cond\n\n'

	sim_ha += 'def ode_solve(init_values, init_mode, time):\n'
	# sim_ha += macros
	init_event = '['
	for i in range(k):
		var = var_indices[i]
		# iv = '({0} - init_values[{1}])'.format(var, i)
		iv = '({0})'.format(var)
		if i == 0:
			init_event += iv
		else:
			init_event += ','+iv
	init_event += ']'

	sim_ha += '\tdt = 0.01\n'+\
			'\tpre_time, last_time = 0, 0\n'+\
			'\tpre_mode = init_mode\n'+\
			'\tpre_values = init_values\n'+\
			'\t'+ variables + ' = init_values'+'\n'+\
			'\tpre_events = {0}\n'.format(init_event)+\
			'\tall_time = [pre_time]\n'+\
			'\tall_values = [pre_values]\n'+\
			'\tall_mode = [pre_mode]\n'+\
			'\twhile last_time <= time:\n'+\
			'\t\tlast_time = pre_time + dt\n'+\
			'\t\tsoln = odeint(func, pre_values, [pre_time, last_time], args = (pre_mode,))\n'+\
			'\t\tvalues = soln[-1]\n'+\
			'\t\tsys.stdout.flush()\n'+\
			'\t\t# check if jump event occurred\n'+\
			'\t\tpre_events, m1 = jumpEvents(pre_mode, pre_values)\n'+\
			'\t\tevents, m2 = jumpEvents(pre_mode, values)\n'+\
			'\t\t#print(\'events\', pre_events, events)\n'+\
			'\t\te1, e2 = eventCondition(pre_events), eventCondition(events)\n'+\
			'\t\t#print(\'-- event condition\', e1, e2)\n'+\
			'\t\tif e1*e2 < 0.0:\n'+\
			'\t\t\t#print(\'------ change events\', e1, e2)\n'+\
			'\t\t\tt1, values1 = pre_time, pre_values\n'+\
			'\t\t\tt2, values2 = last_time, values\n'+\
			'\t\t\tfor j in range(100):\n'+\
			'\t\t\t\tif np.abs(t1 - t2) < atol:\n'+\
			'\t\t\t\t\te_time = t2\n'+\
			'\t\t\t\t\t#print(\'------- jump event\', e_time)\n'+\
			'\t\t\t\t\tbreak\n'+\
			'\t\t\t\tm_t = (t1 + t2)/2\n'+\
			'\t\t\t\tsoln = odeint(func, values1, [t1, m_t], args = (m1,))\n'+\
			'\t\t\t\tm_values = soln[-1]\n'+\
			'\t\t\t\t#print(\'---- j\', j, \'-----\', t1, m_t, m1, m_values)\n'+\
			'\t\t\t\tevents, m_mode = jumpEvents(m1, m_values)\n'+\
			'\t\t\t\tm_e = eventCondition(events)\n'+\
			'\t\t\t\tif m_e * e1 > 0: # ub decreased\n'+\
			'\t\t\t\t\tt2 = (t1 + m_t)/2\n'+\
			'\t\t\t\t\t#print(\'----------------- ub fixed to m\', t1, m_t, t2)\n'+\
			'\t\t\t\telse: # ub increased\n'+\
			'\t\t\t\t\tt2_1 = (m_t + t2)/2\n'+\
			'\t\t\t\t\t#print(\'----------------- ub fixed to m\', m_t, t2, t2_1)\n'+\
			'\t\t\t\t\tt2 = t2_1\n'+\
			'\t\t\tlast_time = e_time\n'+\
			'\t\t\tsoln = odeint(func, pre_values, [pre_time, last_time], args = (pre_mode,))\n'+\
			'\t\t\te_values = soln[-1]\n'+\
			'\t\t\tnext_mode, new_values = check_condition(e_values, pre_mode)\n'+\
			'\t\telse:\n'+\
			'\t\t\tnext_mode, new_values = check_condition(values, pre_mode)\n'+\
			'\t\t#print(\'-- @@\', pre_time, last_time, pre_mode, next_mode, values, new_values)\n'+\
			'\t\t#print(\'-----------------------------------------\')\n'+\
			'\t\tevents, m_mode = jumpEvents(next_mode, new_values)\n'+\
			'\t\tall_values.append(pre_values)\n'+\
			'\t\tall_values.append(new_values)\n'+\
			'\t\tall_time.append(last_time)\n'+\
			'\t\tall_time.append(last_time)\n'+\
			'\t\tall_mode.append(pre_mode)\n'+\
			'\t\tall_mode.append(next_mode)\n'+\
			'\t\tpre_time, pre_mode, pre_values, pre_events = last_time, next_mode, new_values, events\n'+\
			'\treturn all_values, all_time, all_mode\n'

	
	sim_ha += '\n'
	sim_ha += 'def simulate(params, time):'+'\n'
	# sim_ha += macros
	
	init_mode = ha.init.mode
	inits = {}
	for c in ha.init.condition:
		print(str(c.literal1), str(c.literal2))
		inits.update({str(c.literal1):str(c.literal2)})
	print(var_indices, inits)
	
	param_names = '['
	init_pars = ''
	i = 0
	for p in params.keys():
		init_pars += '\t{0} = params[{1}]\n'.format(p, param_indices[ind_vars[p]])
		param_names += '\'{0}\''.format(p) if i == 0 else  ', \'{0}\''.format(p)
		i += 1
	param_names += ']'
	
	inits_all = {}
	inits_updated = {}
	for i in range(k):
		if var_indices[i] in inits:
			iv ='({0})'.format(inits[var_indices[i]])
		elif i in param_indices.keys():
			par_vn = var_indices[i]
			iv = '({0})'.format(par_vn)
		else:
			if par_vn in default_paramFromFile:
				iv = '('+default_paramFromFile[par_vn].to_infix()+')'
			else:
				iv = '(0.5*({0}+{1}))'.format(default_value_ranges[var_indices[i]][0], default_value_ranges[var_indices[i]][1])
		inits_all.update({var_indices[i]:iv})
		inits_updated.update({var_indices[i]:iv})

	print('inits_all --before', inits_all)
	for key in inits_updated:
		for key1 in inits_all:
			expr = inits_all[key1]
			# print(expr, type(expr))
			if key == expr:
				expr = expr.replace(key, inits_updated[key])
			inits_all.update({key1: expr})

	print('inits_all --after', inits_all)
	init_values = '['
	for i in range(k):
		iv =  inits_all[var_indices[i]]
		if i == 0:
			init_values += iv
		else:
			init_values += ','+iv
	init_values += ']'

	j = 0
	default_params = '['
	for i in param_indices:
		par_vn = var_indices[i]
		if par_vn in default_paramFromFile:
			d_iv = default_paramFromFile[par_vn].to_infix()
		else:
			d_iv = '(0.5*({0}+{1}))'.format(default_value_ranges[var_indices[i]][0], default_value_ranges[var_indices[i]][1])
		default_params += d_iv if j == 0 else (', '+ d_iv)
		j += 1
	default_params += ']'

	sim_ha += init_pars
	sim_ha += '\tinit_values = {0}\n'.format(init_values)
	sim_ha += '\tinit_mode = {0}\n'.format(init_mode)
	sim_ha += '\tall_values, all_time, all_mode = ode_solve(init_values, init_mode, time)\n'+\
			'\treturn all_values, all_time, all_mode\n\n'

	var_names = '['
	i = 0
	for key in ind_vars:
		if i == 0:
			var_names += '\'{0}\''.format(key)
		else:
			var_names += ', \'{0}\''.format(key)
		i+= 1
	var_names += ']'

	sim_ha += 'def plot(all_values, all_time):\n'+\
			'\tl_var = len(all_values[0])\n'+\
			'\tmatrix = [[] for i in range(l_var)]\n'+\
			'\tfor i in range(l_var):\n'+\
			'\t\tfor j in range(len(all_time)):\n'+\
			'\t\t\tmatrix[i].append(all_values[j][i])\n'+\
			'\tplot_names = {0}\n'.format(var_names)+\
			'\ti = 0\n'+\
			'\tfigs = []\n'+\
			'\tfig = plt.figure()\n'+\
			'\tfor val in matrix:\n'+\
			'\t\tif i % 3 == 0:\n'+\
			'\t\t\tfig = plt.figure()\n'+\
			'\t\tplt.subplot(3, 1, (i%3+1))\n'+\
			'#\t\tval = mat\n'+\
			'\t\tplt.plot(all_time, val)\n'+\
			'\t\tplt.ylabel(plot_names[i])\n'+\
			'\t\ti += 1\n'+\
			'\t\tif i% 3 == 0:\n'+\
			'\t\t\tfigs.append(fig)\n'+\
			'\tif i%3 != 0:\n'+\
			'\t\tfigs.append(fig)\n'+\
			'\treturn figs\n\n'
			

	sim_ha += 'def run(params, time, log = False):\n'+\
			  '\tif log:\n'+\
			  '\t\tprint(\'run--simulate\', str(params))\n'
	sim_ha += '\tall_values, all_time, all_mode = simulate(params, time)\n'
	sim_ha += '\tl_var = len(all_values[0])\n'+\
			'\tmatrix = [[] for i in range(l_var)]\n'+\
			'\tfor i in range(l_var):\n'+\
			'\t\tfor j in range(len(all_time)):\n'+\
			'\t\t\tmatrix[i].append(all_values[j][i])\n'
	for i in range(k):
		print(var_indices[i])
		sim_ha +='\t{0} = np.array(matrix[{1}])\n'.format(var_indices[i], i)

	rtn_str = '['
	rtn = '['
	i = 0
	for out in outputEqns.keys():
		sim_ha += '\t{0} = {1}\n'.format(out, outputEqns[out])
		rtn += str(out) if i == 0 else (', '+ str(out))
		rtn_str += '\'{0}\''.format(str(out)) if i == 0 else (', \'{0}\''.format(str(out)))
		i += 1
	rtn += ']'
	rtn_str += ']'

	plots_per_page = 3
	plot_cols = 1
	plot_rows = int(plots_per_page/plot_cols)

	sim_ha += '\treturn {0}, all_time\n\n'.format(rtn)

	sim_ha += 'def plot_output(matrix, all_time, plt_title = \'\'):\n'+\
			'\tplot_names = {0}\n'.format(rtn_str)+\
			'\tplot_s = \'\'\n'+\
			'\ti = 0\n'+\
			'\tfigs = []\n'+\
			'\tfig = plt.figure()\n'+\
			'\tfor val in matrix:\n'+\
			'\t\tif i % {0} == 0:\n'.format(plots_per_page)+\
			'\t\t\tfig = plt.figure()\n'+\
			'\t\tplt.subplot({0}, {1}, (i%{2}+1))\n'.format(plot_rows, plot_cols, plots_per_page)+\
			'\t\tlv = int(len(val)/6)\n'+\
			'\t\tplt.plot(all_time, val)\n'+\
			'\t\tplt.ylabel(plot_names[i])\n'+\
			'\t\tpt = \'{0}, {1:0.2f} = {2:0.2f}, {3:0.2f} \'.format(plt_title, val[-1], np.min(val[-lv:-1]), np.max(val[-lv:-1]))\n'+\
			'\t\tplt.title(pt)\n'+\
			'\t\tplot_s += \'[{0}: {1}, last, {2:0.2f}, out {3:0.2f}, range ({4:0.2f}, {5:0.2f}), time ({6:0.2f}, {7:0.2f})]; dp: '+str(default_params)+ '\'.format(plot_names[i], pt, val[-1], val[-2], np.min(val[-lv:-1]), np.max(val[-lv:-1]), all_time[-lv], all_time[-1])\n'+\
			'\t\ti += 1\n'+\
			'\t\tif i%{0} == 0:\n'.format(plots_per_page)+\
			'\t\t\tfigs.append(fig)\n'+\
			'\tif i%{0} != 0:\n'.format(plots_per_page)+\
			'\t\tfigs.append(fig)\n'+\
			'\tprint(plot_s)\n'+\
			'\treturn figs\n\n'
	# print(sim_ha)
	sim_ha += 'def getDefaultParams():\n'+\
			'\tdefault_params = {0}\n'.format(default_params)+\
			'\treturn default_params\n'
	sim_ha += 'def getParamNames():\n'+\
			'\tparam_names = {0}\n'.format(param_names)+\
			'\treturn param_names\n'
	sim_ha += 'def getFIM_COV(params, time):\n'+\
			'\t#default_params = getDefaultParams()\n'+\
			'\t#matrix, timespace = run(default_params, time)\n'+\
			'\t#obs_data = [matrix[i][0] for i in range(len(matrix))]\n'+\
			'\tobs_data = observable_data\n'+\
			'\tobs_time = observed_timespace\n'+\
			'\tparam_names = getParamNames()\n'+\
			'\tobservable_index = list(range({0}))\n'.format(len(outputEqns.keys()))+\
			'\tparam_range = {}\n'
	for i in params.keys():
		sim_ha += '\tparam_range.update({\''+str(i)+'\':('+str(params[i][0])+', '+str(params[i][1])+')})\n'	

	sim_ha +='\t#FIM_1, S1, T = getFIM_global(params, param_names, param_range, obs_data, observable_index, time, atol)\n'+\
			'\tFIM, S, T, M, TM = minifisher(params, param_names, param_range, obs_data, observable_index,obs_time, time, atol) #params, time, atol)\n'+\
			'\t#print(len(params), np.linalg.matrix_rank(FIM))#, np.linalg.matrix_rank(FIM_1))\n'+\
			'\t#FIM, S, T, M = getConnectedParameters(FIM, S, T, params, param_names)\n'+\
			'\t#getConnectedParameters(FIM_1, S1, T, params, param_names)\n'+\
			 '\treturn FIM, S, T, M, TM, obs_data, obs_time \n'

	sim_ha += 'def getDependentParams():\n'+\
			'\tparam_names = getParamNames()\n'+\
			'\tdefault_params = getDefaultParams()\n'+\
			'\tFIM, S, T, M, TM, obs_data, obs_time = getFIM_COV(default_params, time)\n'+\
			'\tindependent, dependent, insenstitive = getConnectedParameters(FIM, S, T, default_params, param_names)\n'+\
			'\treturn dependent\n'+\
			'def getSenseResult(ind):\n'+\
			'\tdefault_params = getDefaultParams()\n'+\
			'\t#FIM1, S1, T1, M1, TM1 = getFIM_COV(default_params, time)\n'+\
			'\tFIM, S, T, M, TM, obs_data, obs_time = getFIM_COV(ind, time)\n'+\
			'\treturn M, TM, S, T, obs_data, obs_time \n'

	sim_ha += 'if __name__ == "__main__":\n'+\
				'\tdefault_params = getDefaultParams()\n'
	sim_ha += '\tall_values, all_time, all_mode = simulate(default_params, time)\n'
	
	if PLOT:
		sim_ha += '\tfigs = plot(all_values, all_time)\n'+\
			'\tpp = PdfPages(\'sim_ha.pdf\')\n'+\
			'\tfor f in figs:\n'+\
			'\t\tpp.savefig(f)\n'+\
			'\tmatrix, all_time = run(default_params, time)\n'+\
			'\tfigs = plot_output(matrix, all_time)\n'+\
			'\tfor f in figs:\n'+\
			'\t\tpp.savefig(f)\n'+\
			'\tmatrix, all_time = run(default_params, time)\n'+\
			'\tfigs = plot_output(matrix, all_time)\n'+\
			'\tfor f in figs:\n'+\
			'\t\tpp.savefig(f)\n'+\
			'\tpp.close()\n\n'

		sim_ha +='\tsampled_times = np.random.choice(range(len(all_time)), 5)\n'+\
			'\tf = open(\"'+datafile+'\", \"w+\")\n'+\
			'\tfor i in sampled_times:\n'+\
			'\t\tt = all_time[i]\n'+\
			'\t\trow = str(t)\n'+\
			'\t\tfor m in matrix:\n'+\
			'\t\t\trow += \',\'+str(m[i])\n'+\
			'\t\trow += \'\\n\'\n'+\
			'\t\tf.write(row)\n'+\
			'\tf.close()\n'+\
			'\tgetFIM_COV(default_params, time)\n'

	f = open(simulateFile, "w+")
	f.write(sim_ha)
	f.close()
	st = ['sage', '--python3', simulateFile, '>', 'log.txt']
	# #print(str(st))
	return st
if __name__ == "__main__": 	
	st = main(sys.argv[1:])
	#output =  subprocess.check_output(st, timeout=6*3600)
