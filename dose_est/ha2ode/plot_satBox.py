from math import *
import numpy as np
import csv, math
import random as rnd
import copy, os, sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from scipy.integrate import odeint


import itertools

sat_csv = 'ex_1_satbox.csv'

params = [0.7, 0.4, 0.75, 0.7]#, 3.0]
param_names = ['k01', 'k02', 'k12', 'k21']#, 'V']

param_bounds = [[0.5, 0.9], [0.2, 0.6], [0.5, 1.0], [0.5, 0.9]]


n_par = len(params)	
contract = {}
for i in range(n_par):
	contract.update({param_names[i]:i})

sat_boxes = []
with open(sat_csv, mode= 'r') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		if row[0].startswith("k"):
			continue
		else:
			instances = {}
			j = 0
			while j < len(row)/2:
				instances.update({param_names[j]:(float(row[2*j]), float(row[2*j+1]))})
				j += 1
			sat_boxes.append(instances)

one_instance = []
sat = sat_boxes[0]
for p in param_names:
	one_instance.append((sat[p][0]+sat[p][1])/2)
one_instance.append(3.0)

def findsubsets(S,m):
	return set(itertools.combinations(S, m))

pp = PdfPages('satBox.pdf')
param_len = 2 
# for all_params in dependent:
# 	# param_len = len(all_params.keys())
subsets_2 = findsubsets(param_names, param_len)
for sub in subsets_2: #for each subset of size 2 
	# print('For param set', sub, type(sub))
	par = [contract[i] for i in sub]

	x = []
	w = []
	axisname = list(sub)
	for p in par:
		x.append(param_bounds[p][0])
		w.append(param_bounds[p][1] - param_bounds[p][0])
	
	fig_t = plt.figure()
	xlim = [0.9*x[0], 1.1*(x[0] + w[0])]
	ylim = [0.9*x[1], 1.1*(x[1] + w[1])]
	# print(xlim, ylim)
	plt.xlabel(axisname[0])
	plt.ylabel(axisname[1])
	# plt.ylim()
	# # plt.axhline(y=x[1], xmin=x[0], xmax=x[0]+w[0])
	# plt.axhline(y=x[1]+w[1], xmin=x[0], xmax=x[0]+w[0])
	# plt.axvline(x=x[0], ymin=x[1], ymax=x[1]+w[1])
	# plt.axvline(x=x[0]+w[0], ymin=x[1], ymax=x[1]+w[1])
	currentAxis = plt.gca()
	currentAxis.set_xlim(xlim)
	currentAxis.set_ylim(ylim)

	for sat in sat_boxes:
		x = []
		w = []
		x1 = []
		x2 = []
		for p in sub:
			x.append(sat[p][0])
			w.append(sat[p][1] - sat[p][0])
		p1, p2 = sub
		x1.append((sat[p1][0]+sat[p1][1])/2)
		x2.append((sat[p2][0]+sat[p2][1])/2)
		currentAxis.add_patch(Rectangle((x[0], x[1]), w[0], w[1], facecolor='grey', alpha=1))
		currentAxis.plot(x1, x2, 'bo')

	pp.savefig(fig_t)



def f(y0, t, p):
	# q = y0
	x1 = y0[0]
	x2 = y0[1]

	# 'k01', 'k02', 'k12', 'k21', 'V'
	k01 = p[0]
	k02 = p[1]
	k12 = p[2]
	k21 = p[3]
	v = p[4]

	# Auxillary equations
	qdot = np.zeros(2)
	
	#  ODEs
	qdot[0] = k12*x2 - (k01+k21)*x1
	qdot[1] = k21*x1 - (k02+k12)*x2
	#  ODE vector
	return qdot

def run(params, time):

	samples = 5000
	tspan = np.linspace(0, time, samples) 

	k01 = params[0]
	k02 = params[1]
	k12 = params[2]
	k21 = params[3]
	v = params[4]
	ic = [15, 0]

	soln = odeint(f, ic, tspan, args = (params, ))
	
	x1 = soln[:, 0]
	x2 = soln[:, 1]

	y = x1/v
	return [y, x2], tspan

data_csv = 'test/ex_1.csv'
data_times = []
data_vals = []
max_time = 10.0
with open(data_csv, mode= 'r') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		if row[0].startswith("k"):
			continue
		else:
			data_times.append(float(row[0]))
			data_vals.append(float(row[1]))
			if max_time < float(row[0]):
				max_time = float(row[0])

max_time = math.ceil(max_time)

cal_vals, cal_times = run(one_instance, max_time)

fig = plt.figure()
plt.plot(data_times, data_vals, 'ro')
plt.plot(cal_times, cal_vals[0], 'b-')

pp.savefig(fig)

pp.close()