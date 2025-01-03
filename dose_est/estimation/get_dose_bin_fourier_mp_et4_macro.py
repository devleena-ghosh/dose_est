#sage --python estimation/get_dose_opt.py -i eisen/model1/eisen_model_smple.drh -o eisen/model1/eisen_model_Out.txt -d eisen/model1/eisen_model_data.csv  > logs/eisen_log_dose.txt
#sage --python estimation/get_dose_opt_1state.py -i eisen/model1/eisen_model_1state.drh -o eisen/model1/eisen_model_Out.txt -d eisen/model1/eisen_model_1state.csv  > logs/eisen_log_dose.txt
# sage --python estimation/get_dose_opt_bin_drch.py -i eisen/model1/eisen_model_1state.drh -o eisen/model1/eisen_model_Out.txt -d eisen/model1/eisen_model_1state_dose.csv  > logs/eisen_log_dose.txt
# sage --python estimation/get_dose_opt_bin_drch.py -i eisen/model1/eisen_model_1state.drh -o eisen/model1/eisen_model_Out.txt -d eisen/model1/eisen_model_1state_dose.csv -s eisen/model1/eisen_sskm_data_172.csv > logs/eisen_log_dose.txt
# sage --python estimation/get_dose_bin_drch_cap_mp.py -i eisen/model1/eisen_model_1state.drh -o eisen/model1/eisen_model_Out.txt -d eisen/model1/eisen_model_1state_dose.csv -s eisen/model1/eisen_sskm_hypo.csv > logs/eisen_log_dose.txt &
#nohup sage --python estimation/get_dose_bin_drch_cap_mp.py -i eisen/model1/eisen_model_1state.drh -o eisen/model1/eisen_model_Out.txt -d eisen/model1/eisen_model_1state_dose.csv -s eisen/model1/eisen_sskm_hypo.csv > logs/eisen_log_dose_233.txt &
#nohup sage --python estimation/get_dose_bin_drch_cap_mp.py -i eisen/model1/eisen_model_1state.drh -o eisen/model1/eisen_model_Out.txt -d eisen/model1/eisen_model_1state_dose.csv -s eisen/model1/eisen_sskm_hypo_263.csv > logs/eisen_log_dose_263.txt &
#nohup sage --python estimation/get_dose_bin_drch_cap_mp.py -i eisen/model1/eisen_model_1state.drh -o eisen/model1/eisen_model_Out.txt -d eisen/model1/eisen_model_1state_dose.csv -s eisen/model1/eisen_sskm_hypo_024.csv > logs/eisen_log_dose_074.txt &
# nohup sage --python estimation/get_dose_bin_drch_cap_mp.py -i eisen/model1/eisen_model_1state.drh -o eisen/model1/eisen_model_Out.txt -d eisen/model1/eisen_model_1state_dose.csv -s eisen/model1/eisen_sskm_opt_data_065.csv > logs/eisen_log_dose_074.txt &
# nohup sage --python estimation/get_dose_bin_drch_cap_mp2.py -i eisen/model1/eisen_model_1state_der_tscale.drh -o eisen/model1/eisen_model_Out.txt -d eisen/model1/eisen_model_1state_dose_der_tscale.csv -s eisen/model1/eisen_sskm_opt_data_065.csv > logs/eisen_log_dose_tscale_275.txt &

# nohup sage --python estimation/get_dose_bin_fourier_mp.py -i eisen/model_hp/eisen_1s_d_scn2.drh -o eisen/model_hp/eisen_model_Out.txt -d eisen/model_hp/eisen_model_1state_dose_der_tscale.csv  -s eisen/model_hp/eisen_sskm_opt_dose_267_validate.csv > logs/eis_log_d_1s_mp_4.txt &

from __future__ import print_function
from ctypes.wintypes import MAX_PATH
import os
import subprocess
import re
import sys, getopt
import csv

import multiprocessing
# from multiprocessing import Pool#, Queue, Process
from pathos.multiprocessing import ProcessingPool

import collections
from collections import OrderedDict
from decimal import Decimal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from model.property import *

from model.haModel import *
from model.phaModel import *
from model.ha_factory import *
from model.node import *
from parser.parseSTL import *
from parser.parseParameters import *
from parser.parseEquations import *
from parser.parseExpr import *
from parser.parseCondition import *

#from model.interval import *
#from model.box import *
#from model.condition import *
import numpy as np
import random as rnd
from util.reach import *
from util.graph import *
from util.stack import *
from util.heap import *
import ha2smt.smtEncoder as smten
from paramUtil.interval import *
from paramUtil.box import *
import paramUtil.box_factory as bfact
from util.parseOutput import *
from util.smtOutput import *
#from paramUtil.readDataBB import *
from model.node_factory import *
# from ha2smt.smtEncoder import *
import ha2ode.HA2ode as hode

import sage.all as sage
import sagepy.sageCal as sagec
import sagepy.optimizeOffset as opt_off
import sagepy.drugFourier as drug_four

import numpy
import time

PLOT = hode.PLOT
if PLOT:
	import matplotlib
	# matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_pdf import PdfPages
	from matplotlib.patches import Rectangle

import itertools

import gp_classifier_dReach as gpc

# fig = plt.figure()

tempfolder = os.path.join('temp','testBox_dose')
SAT = 51
UNSAT = 52
A_UNSAT = 53
UNKNOWN = -1
TRUE = 1
FALSE = 0
UNDET = 2
ONLYMARKED = 1
NOMARK = 0
PATH_LEN = 1 #0
DELTA = 0.15
EPS = 0.2
N_GRIDS = 0.1
MIN_EPS = 0.001 # <10.0/777
MIN_DELTA = 0.002
DATA_NOISE = 0.5
noise = {}
DEBUG = True
# DEBUG = False
TEST = False
# TEST = True
POOL = True
REVERSE = True
AUC = 1
MAX_HPV = 0.02
MIN_HPV = 0.005
RATIO = 0.5

dReachCmd = "dReach"
dRealCmd = "dReal3"
# Optimizer = collections.namedtuple('Optimizer', ['x', 'fun', 'success'])
Instance = collections.namedtuple('Instance', ['p1', 'p2'])
BoxEvaluate = collections.namedtuple('BoxEvaluate', ['ip', 'op'])
Result = collections.namedtuple('Result', ['p0', 'p1', 'p2'])
one_time = 0
PropRow = 0
rank = 0
# RESUME = True 
RESUME = False
#TEST = False
# TEST = True
GP_Window = 200
GP_thres = 0.2
DATA_tp = 5
GPC = True
index = 4
Rows = 1 #20
count = 20
eg = 100
Nterms = 15 #30
IND = 1
# INRANGE = True
INRANGE = False
MS = 0
FIX = 0
TF = 0
SCALE = 120
EXTRA = 0
LOOP = 0
if LOOP == 1:
	MIN_PATH_LEN = 45
	MAX_PATH_LEN = 50
else:
	MIN_PATH_LEN = 0
	MAX_PATH_LEN = 2 #

scale = SCALE
if IND == 1:
	suf = 'dose_ed_1s_scn'+str(LOOP)+'_'
else:
	if FIX == 1:
		if MS == 1:
			suf = 'dose_ed_AUC{3}_hpr{0}_nd_ms_{1}_ex{2}_IR_RAT_{4}_TF_{5}_'.format(FIX, LOOP, EXTRA, AUC, RATIO, TF) \
			if INRANGE else 'dose_ed_AUC{3}_hpr{0}_nd_ex{2}_ms_{1}_RAT_{4}_TF_{5}_'.format(FIX, LOOP, EXTRA, AUC, RATIO, TF)
		else:
			suf = 'dose_ed_AUC{3}_hpr{0}_nd_1s_{1}_ex{2}_IR_RAT_{4}_TF_{5}_'.format(FIX, LOOP, EXTRA, AUC, RATIO, TF) \
			if INRANGE else 'dose_ed_AUC{3}_hpr{0}_nd_ex{2}_1s_{1}_RAT_{4}_TF_{5}_'.format(FIX, LOOP, EXTRA, AUC, RATIO, TF)
	else:
		if MS == 1:
			suf = 'dose_ed_AUC{3}_hpr{0}_nd_ms_{1}_ex{2}_IR_NORAT_TF_{4}_'.format(FIX, LOOP, EXTRA, AUC, TF) \
			if INRANGE else 'dose_ed_AUC{3}_hpr{0}_nd_ex{2}_ms_{1}_NORAT_TF_{4}_'.format(FIX, LOOP, EXTRA, AUC, TF)
		else:
			suf = 'dose_ed_AUC{3}_hpr{0}_nd_1s_{1}_ex{2}_IR_NORAT_TF_{4}_'.format(FIX, LOOP, EXTRA, AUC, TF) \
			if INRANGE else 'dose_ed_AUC{3}_hpr{0}_nd_ex{2}_1s_{1}_NORAT_TF_{4}_'.format(FIX, LOOP, EXTRA, AUC, TF)
	suf = suf.replace('.','_')
def toStr(expr):
	exprString = str(expr).replace('e^', 'exp')#.replace('t', 'tm')
	return exprString

class BoxInfo(object):
	"""docstring for ClassName"""
	def __init__(self, *tuple):
		super(BoxInfo, self).__init__()
		self.pri = tuple[0]
		self.box = tuple[1]
		self.delta = tuple[2]
		self.prop = tuple[3]

	def __lt__(self, nxt):
		return self.pri < nxt.pri

	def __gt__(self, nxt):
		return self.pri > nxt.pri

	def getInfo(self):
		return self.pri, self.box, self.delta, self.prop
		
	def __repr__(self):
		sk = str(self.pri) + ', ' + str(self.box) + ', '+ str(self.delta)+ ', '+ str(self.prop)
		return sk

	def __str__(self):
		return self.__repr__()
		
class Dosage: 
	def __init__(self, a = [], b = 0):
		if len(a) > 0:
			self.result = Result(p0=a[0], p1= a[1], p2 = a[2])
		else:
			self.result = Result(p0=UNSAT, p1= UNSAT, p2 = UNSAT)
		self.value = b	
	def __str__(self):
		return '('+str(self.result) + ' dose:'+ str(self.value)+')'
	def __repr__(self):
		return '('+str(self.result) + ' dose:'+ str(self.value)+')'

def findsubsets(S,m):
	return set(itertools.combinations(S, m))

def plotBox(currentAxis, b, combs, boxType= FALSE):
	#if boxType = TRUE:
	# print(boxType)
	col = 'white'
	if boxType == TRUE:
		col = 'blue'
	elif boxType == UNDET:
		col = 'white'
	elif boxType == FALSE:
		col = 'black'
	# plt.figure()	
	if b.size() == 2:
		b_edges = b.get_map()
		x = []
		w = []
		for it in combs:
			intrvl = b_edges[it]
			x.append(intrvl.leftBound())
			w.append(intrvl.width())
		currentAxis.add_patch(Rectangle((x[0], x[1]), w[0], w[1], facecolor=col, alpha=1))
	# plotBox.show()
	# return plt

def main(argv):
	k_length = MIN_PATH_LEN, MAX_PATH_LEN
	d = DELTA	
	#global EPS
	#EPS = 1 * d
	one_time = 0
	#plotName = 'plotBox_'+suf+'{0}.pdf'.format(index)
	#satName = 'satbox_'+suf+'{0}.csv'.format(index)
	resultName = 'result_'+suf+'{0}.csv'.format(index)
	
	# inputfile = sys.argv[1]
	# paramfile = sys.argv[2]
	# datafile = sys.argv[3]
	paramfile = ''
	paramdefaultfile = ''
	observedfile = ''
	
	try:
		opts, args = getopt.getopt(argv,"hi:p:o:d:f:s:",["ifile=","pfile=", "ofile=", "dfile=","defaultparam=","sskm="])
	except getopt.GetoptError:
			print("estimate_dosage.py -i <inputfile> -p <paramFile> -o <outputfile> -d <dataFile> -f <defaultparam> -s <observed>")
			sys.exit(2)
			
	for opt, arg in opts:
		if opt == '-h':
			print("estimate_dosage.py -i <inputfile> -p <paramFile> -o <outputfile> -d <dataFile> -f <defaultparam> -s <observed>")
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
			print("Input file is :" + inputfile)
		elif opt in ("-p", "--pfile"):
			paramfile = arg
			print("Param file is :" + paramfile)
		elif opt in ("-o", "--ofile"):
			outputfile = arg
			print("Output file is :" + outputfile)
		elif opt in ("-d", "--datafile"):
			datafile = arg
			print("Data file is :" + datafile)
		elif opt in ("-f", "--defaultparam"):
			paramdefaultfile = arg
			print("paramdefault file is :" + paramdefaultfile)
		elif opt in ("-s", "--sskm"):
			observedfile = arg
			print("observedfile file is :" + observedfile)
	
	inFileName = inputfile.split('.')[0]
	# outfile = inFileName+'_'+satName

	all_res_file = inFileName+'_'+resultName

	# ha = getModel(inputfile)
	# print('model parsed')
	# print(str(ha))
	default_paramFromFile = {}
	if len(paramdefaultfile) > 0:
		default_paramFromFile = getEquationsFile(paramdefaultfile)
		print('Default Params')
		for var in default_paramFromFile:
			print(var + ' : '+ str(default_paramFromFile[var]))

	'''
	all_sat_file = 'sat_box_all_ed_hpr_nd_1s__{0}.csv'.format(index) #.format(rank)
	all_q_file = 'queued_box_all_ed_hpr_nd_1s_{0}.csv'.format(index) #.format(rank)

	# resume_sat_file = 'sat_box_all_qn_gpc_dose.csv' #.format(rank)
	# resume_q_file = 'queued_box_all_qn_gpc_dose.csv' #.format(rank)
	if not RESUME:
		fp = open(all_sat_file, 'w')
		fp.close()
		fp = open(all_q_file, 'w')
		fp.close()
	'''
	print('---- sskm_rows ----')
	sskm_rows = {}
	with open(observedfile, 'r') as file:
		csvreader = csv.reader(file)
		header = next(csvreader)
		print(header, type(header))
		for row in csvreader:
			rf = {}
			rn = int(row[0])			
			hypo_params = []
			for i in range(0, len(row)):
				rf.update({header[i]:float(row[i])})

			for i in range(0, len(row)):
				if header[i] in ['p_47','p_48','ic_0','ic_1','ic_2','ic_3','ic_4','ic_5','ic_6','ic_7','ic_8','ic_9','ic_10','ic_11','ic_12','ic_13','ic_14','ic_15','ic_16','ic_17','ic_18','ic_19']:
					hypo_params.append((header[i], float(row[i])))
				if header[i] in ['d_1','p_1','p_19','p_7','p_40','p_10','p_37','p_13']:					
					hypo_params.append((header[i], float(row[i])))
				if header[i] in ['d_1_err','p_1_err','p_19_err','p_7_err', 'p_40_err','p_10_err','p_37_err','p_13_err']:
					hpv = float(row[i])
					mhpv = MAX_HPV
					if header[i] == 'd_1_err':
						hv = rf['d_1'] #, 0.0001)
					if header[i] == 'p_1_err':
						hv = rf['p_1']
						mhpv = 0.5*MAX_HPV
					if header[i] == 'p_19_err':
						hv = rf['p_19']
						mhpv = 0.5*MAX_HPV
					if header[i] == 'p_37_err':
						hv = rf['p_37']
						mhpv = 0.1*MAX_HPV
					if header[i] == 'p_7_err':
						hv = rf['p_7']				
						mhpv = 0.1*MAX_HPV
					if header[i] == 'p_40_err':
						hv = rf['p_40']				
						mhpv = 0.1*MAX_HPV
					if header[i] == 'p_40_err':
						hv = rf['p_40']				
						mhpv = 0.1*MAX_HPV
					if header[i] == 'p_10_err':
						hv = rf['p_10']				
						mhpv = 0.1*MAX_HPV
					if header[i] == 'p_13_err':
						hv = rf['p_13']				
						mhpv = 0.1*MAX_HPV
					# print()

					hpv1 = max(min(hpv/hv, mhpv), MIN_HPV) # ranges within 0.1% to 2%
					hypo_params.append((header[i], hpv1))
				
			#for i in range(11, len(row)-1):
				#hypo_params.append((header[i], float(row[i])))
			# Lt4 = float(row[i+1])
			# hypo
			# hypo_cal = float(row[i+2])
			if (rn-1) in range(Rows*(index-1), Rows*(index-1)+Rows):
				sskm_rows.update({(rn):(hypo_params, rf)})
				print(rn, rf)

			#break
	sskm_rows_keys = sorted(sskm_rows.keys(), reverse=REVERSE)
	print('--------', sskm_rows_keys)
	def getEstimatedParameters(model, params, d, klen, datafile, fixed, outputEqns):

		all_data = []
		all_props = []
		tp = 0
		with open(datafile) as fp:
			fr = csv.reader(fp, delimiter=',')
			for row in fr:
				tp += 1
				# if tp == PropRow:
				data = [float(row[i]) for i in range(len(row))]
				
				(prop, propn1, propn2, mode) = getPropertyFromData(data, fixed, outputEqns)
				 #convert2Prop(data, dtype) # convert a row to property
				# if DEBUG:
				print(mode, 'Property: ', prop) #, ' State: ', dtype)
				print(mode, 'PropertyNeg: --- ')
				print(propn1)
				print(propn2) #, ' State: ', dtype)
				(pn, pn1, pn2, mode) = getProperties((prop, propn1, propn2), mode) #, dtype)
				all_data.append(data)
				all_props.append((pn, pn1, pn2, mode))

				if tp >= DATA_tp:
					break

		sbox = getBox(params)
		# updateModel(model, sbox)

		dosage_box =  estimateDosageRange(model, all_props, sbox,  klen)

		print('sat boxes', len(dosage_box), len(all_props)) #, bt)
	
		return dosage_box#, [], [])

	def getBoxDelta(b):
		# return b.min_side_width()*N_GRIDS 
		return max(b.min_side_width()*N_GRIDS, MIN_DELTA)
	
	def ifBoxEPS(b):
		return (b.max_side_width() < MIN_EPS)
	
	def estimateDosageRange(model, all_props, sbox, klen):
		c_map = {}
		bmap = sbox.get_map()
		# i = 0
		# for it in bmap:
			# if i == 0:
		it = 'et4'#list(bmap.keys())[0]
		l, u = float(model.macros['mint4'].evaluate().value), float(model.macros['maxt4'].evaluate().value) #bmap[it].leftBound(), bmap[it].rightBound()*0.9
		print('estimateDosageRange', l, u, type(l), type(u))
		# l, u = 0.0253125, 0.028297973899825753
		evaluate_dict = {}
		# for i in range(len(all_props)):
		p, pn1, pn2, mode = all_props[0]
		# min_prop = (p, pn2, mode)
		# max_prop = (p, pn1, mode)
		prop = (p, pn1, pn2, mode)
		stime =  time.time()
		# lvr = getDosageEvaluation(model, sbox, prop, it, l, evaluate_dict, klen, minmax = 0) #evaluate(model, sbox, prop, it, l)
		# etime = time.time()
		# print('SAT', SAT, 'UNSAT', UNSAT, 'after lvr', etime - stime, 's', lvr)
		# uvr = getDosageEvaluation(model, sbox, prop, it, u, evaluate_dict,  klen, minmax = 0) #evaluate(model, sbox, prop, it, u)
		# etime1 = time.time()
		# print('SAT', SAT, 'UNSAT', UNSAT, 'after uvr', etime1 - etime, 's', uvr)
		print('getDoseLimit: finding minimum- maximum ', str(l), str(u)) # lvr, uvr)
		dose_u, dose_l = getDoseLimit(model, sbox, prop, it, l, u,  klen)
		# print('getDoseLimit: finding maximum ', str(l), str(u))#, lvr, uvr)
		sys.stdout.flush()
		# dose_v = getDoseLimit(model, sbox, prop, it, l, u, evaluate_dict,  klen, minmax = 1)
		# dose_v = u
		l, u = dose_l, dose_u
		# lv, uv = Dosage(r1, dose_l), Dosage(r2, dose_v)
		print('dosage model', l, u)
		
		if u > l:
			c_map.update({it:PyInterval(l, u)})
			return [Box(c_map)]
		else:
			return []

	def des(r, rc):
		rt = True if r == SAT and rc == UNSAT else False
		# print('des', 'r', r, type(r), 'rc', rc, type(rc))
		return rt
		
	def desired(dv):
		r = dv.result[0]
		rc1 = dv.result[1]
		rc2 = dv.result[1]
		rt1 = des(r, rc1)
		rt2 = des(r, rc2)
		return rt1, rt2

	def specifc(dv):
		r = dv.result[0]
		rc1 = dv.result[1]
		rc2 = dv.result[1]
		# if minmax == -1:
		rt1 = True if rc1 == SAT and r == UNSAT else False
		rt2 = True if rc2 == SAT and r == UNSAT else False
		return rt1, rt2

	def evaluate(model, sbox, prop, it, it_dose1, it_dose2, klen):

		(propNode, propNeg1, propNeg2, mode) = prop
		propN1 = getNegatedProperty(propNeg1) #, instance1)
		propN2 = getNegatedProperty(propNeg2) #, instance1)

		ps = [(propNode, mode), (propN1, mode), (propN2, mode)]
		if AUC in [1, 2]:
			import getFS as fs
			x = fs.getX((tg_t4, p43, it_dose1, AUC))
		else:
			x = 1.0
		model.macros.update({'eta':getExpression(('{0}'.format(x)))})

		print('avg_gamma', x, (1 - np.exp(-p43*tg_t4)))
		print('in evaluate', it, it_dose1, it_dose2)
		model1 = model.clone()
		model2 = model.clone()

		b_map1 = {}
		b_map1.update({it:PyInterval(0.99*it_dose1, 1.01*it_dose1)})
		sb1 = Box(b_map1)
		
		# updateModel(model1, sb1)
		#model1.addInit(it, it_dose1)
		model1.macros.update({it:it_dose1})
		print('evaluate -- after model update', model1.init, model1.macros[it])
		# move this code before getEstimation...
		# with open(temp_ha,'w+') as tfp:
		# 	tfp.write(str(model1))
		# print('evaluate -- Step-6.5: temp model saved', temp_ha, klen)

		b_map2 = {}
		b_map2.update({it:PyInterval(0.99*it_dose2, 1.01*it_dose2)})
		sb2 = Box(b_map2)
		# updateModel(model2, sb2)
		#model2.addInit(it, it_dose2)
		model2.macros.update({it:it_dose2})
		print('evaluate -- after model update', model2.init, model2.macros[it])
		# move this code before getEstimation...
		# with open(temp_ha,'w+') as tfp:
		# 	tfp.write(str(model))
		# print('evaluate -- Step-6.5: temp model saved', temp_ha, klen)
		models = [model, model1, model2]
		#propNode = getSTL(prop)[0]
		if DEBUG:
			print('In evaluate -- {0}, {1}, {2}, {3}'.format(sb1, it_dose1, sb2, it_dose2)) #,\t prop : '+str(propNode)+'\n\t negProp : '+str(propNeg1) )
			# print('@@ prop: '+ str(propNode),  str(propNeg1), str(propNeg2))
		doses = [0.0, it_dose1, it_dose2]
		rp = [1, 2]
		# negs = [0, 1, 2]
		inputs = [(models[ij], ps[ij], MIN_DELTA, klen, 0, 0, ij) for ij in rp]
		if POOL:
			pool = ProcessingPool(len(rp))
			# inputs = [[model, all_props, allboxes[j], klen, j] for j in range(num_proc)]
			# 	#print(inputs)
			results = pool.map(checkproperty, inputs)
		else:
			results = [checkproperty(inputs[j]) for j in range(len(rp))]

		if POOL:
			pool.close()
			pool.join()
			pool.clear()
		
		all_res = []
		# if minmax == -1:
		for jk in range(len(results)):  
			resij = [0, 0, 0] 
			res, ij = results[jk]
			(res1, i1, out1) = res
			resij[ij] = res1
			all_res.append((resij, doses[ij]))
		# resij[ij]

		return all_res

	def getDosageEvaluation(model, sbox, prop, it, d1, d2, klen):
		st1 = time.time()
		all_rvs = evaluate(model, sbox, prop, it, d1, d2, klen)
		dvs = [Dosage(allrv[0], allrv[1]) for allrv in all_rvs] 
		
		sb_map = {}
		for it1 in sbox.get_map():
			sb_map.update({it1:sbox.get_map()[it1]})
		sb_map.update({it:PyInterval(d1)})
		sb = Box(sb_map)
		ac_b1 =  getActualDosages(sb)
		sb_map = {}
		for it1 in sbox.get_map():
			sb_map.update({it1:sbox.get_map()[it1]})
		sb_map.update({it:PyInterval(d2)})
		sb = Box(sb_map)
		ac_b2 =  getActualDosages(sb)
		print('\t --- simulated box -----', str(ac_b1), str(ac_b2), d1, d2, dvs)
		# ac_b_t4, ac_b_t3 = ac_b[0], ac_b[1]
		#g_pars_lb = [d1]
		#g_pars_ub = [d2]
		#b_map = sb.get_map()
		#for key in sorted(b_map.keys()):
		# 	g_pars_lb.append(b_map[key].leftBound())
		# 	g_pars_ub.append(b_map[key].rightBound())
		
		# # # fn  = temp_ha_preprocessed

		'''print('\t --- simulated box -----', str(sb), sim.time)
		all_values, all_time = sim.run(g_pars_lb, sim.time)
		figs = sim.plot_output(all_values, all_time, plt_title=('SIM: LB: Dose {0:0.2f}, actual {1:0.2f}, raw {2:0.2f}, tsh range {3}'.format(g_pars_lb[0]*s_t4*777.0, ac_b_t4[0], g_pars_lb[0]*s_t4, (1.8*0.75, 1.8*1.25))))

		all_values, all_time = sim.run(g_pars_ub, sim.time)
		figs = sim.plot_output(all_values, all_time, plt_title=('UB: Dose {0:0.2f}, actual {1:0.2f}, raw {2:0.2f}, tsh range {3}'.format(g_pars_ub[0]*s_t4*777.0, ac_b_t4[0], g_pars_ub[0]*s_t4, (1.8*0.75, 1.8*1.25))))
		'''
				
		# rt = specifc(dv)
		# rt1 = desired(dv) #des(rv[0], rv[1])
		etime1 = time.time()
		print('\t getDosageEvaluation --',(etime1 - st1)/60.0, 'min', d1, d2, dvs)#, rt)
		return dvs

	def getDoseLimit(model, sbox, prop, it, l, u, klen): #, minmax = -1):
		l1, l2 = l, l
		u1, u2 = u, u
		# mvrs = getDosageEvaluation(model, sbox, prop, it, md, md, klen)
		r = Dosage()
		print('getDoseLimit -- search ', 'lv',  (l), 'uv', (u), type(l), type(u))
		sys.stdout.flush()
		while(l1 <= u1 or l2 <= u2) : 
			del1, del2 = u1 - l1, u2 - l2
			md1, md2 = (l1 + u1)/2, (l2 + u2)/2
			mvrs = getDosageEvaluation(model, sbox, prop, it, md1, md2, klen)
			mvr1, mvr2 = mvrs[0], mvrs[1]
			if(del1 < MIN_EPS and del2 < MIN_EPS): #and mvr1.result.p2 == UNSAT and mvr2.result.p1 == UNSAT):				
				print('< delta (', del1, ') : ', 'lv', (l1), 'uv', (u1))
				print('< delta (', del2, ') : ', 'lv', (l2), 'uv', (u2))
				print('select mvr', mvrs)
				# mvr1, mvr2 = mvrs[0], mvrs[1]
				r = md1, md2
				break
			else:	
				sys.stdout.flush()
				# mr = evaluate(model, sbox, prop, it, md)
				# mv = Dosage(mr, md)
				# if minmax == -1: # for minimum
				if(mvr2.result.p2 == SAT):# and not mvr.result.p0==UNSAT): # p2: tsh > max 
					# u1 = md  
					mr = [0, 0, mvr2.result.p2]
					l2 = md2
				else:
					# l1 = md
					mr = [0, 0, mvr2.result.p2]
					u2 = md2
				# else: # for maximum
				if(mvr1.result.p1 == SAT):#and not mvr.result.p0==UNSAT): # p1 : tsh < min
					# l1 = md
					mr = [0, mvr1.result.p1, 0]
					u1 = md1
				else:
					# u1 = md
					mr = [0, mvr2.result.p1, 0]
					l1 = md1
				print("@@@ --- search: ",'[lv', (l1), 'mv', (md1, mr), 'uv', (u1), '][lv', (l2), 'mv', (md2, mr), 'uv', (u2), ']')	
			
			# else:
			# 	print("drug dosage not found for the model", r)
			# 	sys.stdout.flush()
			# 	break; 
		return r


	ha2 = getModel(inputfile)
	if DEBUG:
		print('Step-1: model parsed ...') #, str(ha1))

	res_header = ['row','hypo_obs_TSH','hypo_obs_FT4','hypo_c_TSH','hypo_c_FT4', 'd_1','LT4', 'eu_base_TSH', 'eu_base_FT4','eu_sp_TSH','eu_sp_FT4']
	res_csv = open(all_res_file, 'a+')
	wr = ['{0}'.format(rr) for rr in res_header]+ ['dose_l', 'dose_u']+ ['dose_l', 'dose_u'] + ['range','HPV']
	res_writer = csv.writer(res_csv, delimiter=',')
	res_writer.writerow(wr)
	res_csv.close()

	figs_to_plot = []
	ijk = 0
	for ss_rn in sskm_rows_keys: #sskm_rows.keys():
		# ss_rn, ss_cl = rn_cl
		hypo_params, rf = sskm_rows[ss_rn]
		print('sskm_rows', ss_rn, rf)

		# inputs = [(model, p1, MIN_DELTA, PATH_LEN, False), (model, p2, MIN_DELTA, PATH_LEN, True), (model, p3, MIN_DELTA, PATH_LEN, True)]
		# if POOL:
		# 	pool = ProcessingPool(3)
		# 	# inputs = [[model, all_props, allboxes[j], klen, j] for j in range(num_proc)]
		# 	# 	#print(inputs)
		# 	results = pool.map(checkproperty, inputs)
		# else:
		# 	results = [checkproperty(inputs[j]) for j in range(3)]

		# if POOL:
		# 	pool.close()
		# 	pool.join()
		# 	pool.clear()
		# # for jk in range(3):   
		# p11, res1 = results[jk]
		ha1 = ha2.clone()
		print('#### -------------------------- for row {0} -----'.format(ss_rn), rf['LT4'], rf['hypo_obs_TSH'], rf['hypo_c_TSH'], rf['eu_base_TSH'], rf['eu_sp_TSH'])
		print('--- model initialisation ---')
		# ha1 = ha11 #.simplify()
		ha1.macros.update({'RAT':getExpression(('{0}'.format(RATIO)))})
		ha1.macros.update({'s0':getExpression(('{0}'.format(scale)))})
		for hh in ['d_1_err','p_1_err','p_19_err','p_37_err', 'p_7_err', 'p_10_err', 'p_13_err', 'p_40_err']:
			hpv1 = MAX_HPV
			if not hh == 'd_1_err':
				hpv1 = 0.5*MAX_HPV
			hpv = getExpression(('{0}'.format(hpv1)))
			ha1.macros.update({hh:hpv})
			print('macros -- default', hh, hpv1)

		for ik in range(len(hypo_params)):
			hypo_par_name, hypo_par_val = hypo_params[ik]
			hypo_val = hypo_par_val
			# if FIX == 1 and TF == 1:
			# 	hypo_val = hypo_par_val * (1+ MAX_HPV)
			hpv = getExpression(('{0}'.format(hypo_val)))
			# ha1.macros[hypo_par_name] = hpv
			if hypo_par_name in ['d_1', 'p_1','p_19','p_37', 'p_7', 'p_10', 'p_13', 'p_40']:
				hypo_pn = 'd'+hypo_par_name
			else:
				hypo_pn = hypo_par_name
				 
			ha1.macros.update({hypo_pn:hpv})
			print('macros', hypo_pn, hypo_val)
		stm = time.time()
		
		# for key in ha1.macros.keys():
		# 	print('--- {0} : {1} ---'.format(str(key), str(ha1.macros[key])))
		print('-------------------------')
		NumModes = len(ha1.states)
		# global PATH_LEN	
		#k_length = 30*NumModes if NumModes > 1 else 2
		ijk += 1
		# if ijk > 2:
		# 	break

		# results = evaluate(ha1, all_props, allboxes[j], klen, 0)
		def getPF(tg1):
			if tg1 == 24.0:
				p = 1.2
				f = 1.2957
			elif tg1 == 12.0:
				p = 0.6
				f = 1.201996
			elif tg1 == 8.0:
				p = 0.4
				f = 1.1729
			else:
				p = 1.0/20 #2*tg_t4/N
				f = 4.0
			return p, f

		#tg_t4, tg_t3, p43, p45, s_t4, s_t3  = ha1.getMacroValues(['tg_t4', 'tg_t3', 'p_43', 'p_45', 's_t4', 's_t3'])
		#tg_t4, tg_t3, p43, p45, s_t4, s_t3 = float(tg_t4), float(tg_t3), float(p43), float(p45), float(s_t4), float(s_t3)
		
		tg_t4, tg_t3, p43, p45, s_t4, s_t3, s_0  = ha1.getMacroValues(['tg_t4', 'tg_t3', 'p_43', 'p_45', 's_t4', 's_t3', 's0'])
		p43, p45, s_t4, s_t3, s_0 = float(p43), float(p45), float(s_t4), float(s_t3), float(s_0)
		tg_t4, tg_t3 = float(tg_t4.replace('s0', str(s_0))), float(tg_t3.replace('s0', str(s_0)))
		eps_1, delta_1 = getPF(tg_t4)
		eps_2, delta_2 = getPF(tg_t3)

		# if DEBUG:
		print('Step-2: macros', tg_t4, tg_t3, p43, p45, eps_1, eps_2, delta_1, delta_2, s_t4, s_t3)

		def getFourierFunction(tg_t4, tg_t3, p_43, p_45, eps_1, eps_2, delta_1, delta_2):
			SIMPLE = TRUE
			# Nterms = 5			
			sage.var('et4s', 'et3s', 'off_t3', 'off_t4', 'q_0s', 'eta')

			# if SIMPLE:
			# 	(fn, der_fn, f1, a0) = drug_four.getFourierSimple(Nterms, eg)
			# 	# a0 = eval('a0+off')
			# else:
			# 	(fn, der_fn, f1, a0) = drug_four.getFourier(Nterms)
			# 	# a0 = eval('a0+off')
			(four_max, f1, f2, a0) = drug_four.createFile(Nterms)
			
			#dfn = getFourierDerivative(fn)
			if AUC in [0, 1]:
				fn = eta*four_max
			else:
				fn = eta + four_max
			pfr_t4 = fn(t_gap = tg_t4, k_a = p_43, pi=math.pi, off = 0, d0 = et4s, eps = eps_1, delta = delta_1)
			defs_t4 = drug_four.getFourierDerivative(pfr_t4)	
			defs_4 = defs_t4(tm = q_0s)
			pfr_t4 = pfr_t4(tm = q_0s)
			# inita_t4 = a0(t_gap = tg_t4, k_a = p_43, pi=math.pi, off = 0, d0 = et4s, eps = eps_1, delta = delta_1)
			# initf_t4 = pfr_t4(tm = 0) #+ off_t4
			initf_t4 = fn(tm = 0, t_gap = tg_t4, k_a = p_43, pi=math.pi, off = 0, d0 = et4s, eps = eps_1, delta = delta_1) 
			#(dose_t4/(p_43*tg_t4))
			
			#print("four ", pfr, "\n derivative: ",toString(defs),"\n der: ", toString(defs2), "\n init: ", toString(inita) , "\n initf: ", toString(initf))
			#print("four ", pfr_t4, "\n derivative: ",toStr(defs_t4),"\n initf: ", toStr(initf_t4))
			
			# print('\n T4 derivative of fourier: \n')
			# print(defs_t4)
			# print(inita_t4, initf_t4)
			
			pfr_t3 = fn(t_gap = tg_t3, k_a = p_45, pi=math.pi, off = 0, d0 = et3s, eps = eps_2, delta = delta_2)
			defs_t3 = drug_four.getFourierDerivative(pfr_t3)
			defs_3 = defs_t3(tm = q_0s)
			pfr_t3 = pfr_t3(tm = q_0s)
			#defs2 = dfn(t_gap = tg, k_a = ka, eps = p, pi=math.pi, delta= f, off = 0)
			# inita_t3 = a0(t_gap = tg_t3, k_a = p_45, pi=math.pi, off = 0, d0 = et3s, eps = eps_2, delta = delta_2)
			# initf_t3 = pfr_t3(tm = 0) #+ off_t3
			initf_t3 = fn(tm = 0, t_gap = tg_t3, k_a = p_45, pi=math.pi, off = 0, d0 = et3s, eps = eps_2, delta = delta_2)
			#initf_t3 = (dose_t3/(p_45*tg_t3))
			#print("four ", pfr, "\n derivative: ",toString(defs),"\n der: ", toString(defs2), "\n init: ", toString(inita) , "\n initf: ", toString(initf))
			#print("four ", pfr_t3, "\n derivative: ",toStr(defs_t3),"\n initf: ", toStr(initf_t3))
			
			# print('\n T3 derivative of fourier: \n')
			# print(defs_t3)
			# print(inita_t3, initf_t3)

			# var('et4, et3')
			# (fet1, chk1) = getRange(p1, N, tg_t4)
			# (fet2, chk2) = getRange(p2, N, tg_t3)	
			# fn1 = fn(t_gap = tg_t4, k_a = p_43, eps = p1, d0 = et4, pi=math.pi, delta = f1, off = 0)
			# fn2 = fn(t_gap = tg_t3, k_a = p_45, eps = p2, d0 = et3, pi=math.pi, delta = f2, off = 0)
			
			# #print(fn, fn1)
			# #sys.stdout.flush()

			# (off_t3, tm1) = getOffset(fn1, tg_t4 - fet1, chk1)
			# (off_t4, tm2) = getOffset(fn2, tg_t3 - fet2, chk2)

			init_et4 = initf_t4(tg_t4 = tg_t4, p_43 = p_43, t_gap = tg_t4, k_a = p_43, eps_t4 = eps_1, dose_t4 = et4s, pi=math.pi, delta_t4 = delta_2)#, off_t4 = offset1)
			init_et3 = initf_t3(tg_t3 = tg_t3, p_45 = p_45, t_gap = tg_t3, k_a = p_45, eps_t3 = eps_2, dose_t3 = et3s, pi=math.pi, delta_t3 = delta_2)#, off_t3 = offset2)
		
			return fn, pfr_t4, pfr_t3, defs_4, defs_3, init_et4, init_et3

		fn_four, pfr_t4, pfr_t3, defs_t4, defs_t3, initf_t4, initf_t3 = getFourierFunction(tg_t4, tg_t3, p43, p45, eps_1, eps_2, delta_1, delta_2)

		print('replace', tg_t4, tg_t3, p43, p45)
		print('pfr_t4', pfr_t4)
		dt4 = getExpression(('{0}'.format(defs_t4)).replace('tm', '(q_0s)').replace('e^', 'exp'))
		# dt3 = getExpression(('{0}'.format(defs_t3)).replace('tm', '(q_0s)').replace('e^', 'exp'))
		#dt4 = getExpression(('{0}'.format(pfr_t4)).replace('e^', 'exp'))#.replace('q_0s', 'tm'))
		#dt3 = getExpression(('{0}'.format(pfr_t3)).replace('e^', 'exp')) #.replace('q_0s', 'tm'))
		et4_in = getExpression('{0}'.format(initf_t4))
		print('defs_t4', dt4)
		# et3_in = getExpression('{0}'.format(initf_t3))
		# ha1.macros['dose_t4'] = dt4
		# ha1.macros['dose_t3'] = dt3
		# ha1.macros['et4_init'] = et4_in
		# ha1.macros['et3_init'] = et3_in

		ha1.macros.update({'dose_t4':dt4})
		# ha1.macros.update({'dose_t3':dt3})
		ha1.macros.update({'et4_init':et4_in})
		# ha1.macros.update({'et3_init':et3_in})

		if DEBUG:
			print('Step-3: replace macros 1')#, str(ha1))

		def getActualDosages(dose_Box, flag = 0): #, tg_t4, tg_t3, p43, p45, eps_1, eps_2, delta_1, delta_2):
			dose_l = dose_Box.min_left_coordinate_value()
			dose_u = dose_Box.max_right_coordinate_value()
			print('dose_l', str(dose_l), 'dose_u', str(dose_u))
			# dose_t4, dose_t3 = params
			#(off1, ffs1, p1, f1, tm1) = optimizeSmootherfunction(N, tg_t4, p_43, dose_t4)
			# print(p1, f1)
			#(off2, ffs2, p2, f2, tm2) = optimizeSmootherfunction(N, tg_t3, p_45, dose_t3)
			# print(p2, f2)
			dt4_l, dt4_u = dose_l['et4']*s_t4, dose_u['et4']*s_t4
			# dt3_l, dt3_u = dose_l['et3s']*s_t3, dose_u['et3s']*s_t3

			off_t4 = 0, 0 #(0.1*(dt4_l/(1 - np.exp(-p43 * tg_t4)))), (0.1*(dt4_u/(1 - np.exp(-p43 * tg_t4))))
			# off_t3 = (0.1*(dt3_l/(1 - np.exp(-p45 * tg_t3)))), (0.1*(dt3_l/(1 - np.exp(-p45 * tg_t3))))

			# (r4_l, res4_l) = sagec.getSSActual(fn_four, tg_t4, p43, eps_1, 1.0, delta_1, dt4_l, off_t4[0])
			# (r4_u, res4_u) = sagec.getSSActual(fn_four, tg_t4, p43, eps_1, 1.0, delta_1, dt4_u, off_t4[1])
			r4_l = dt4_l
			r4_u = dt4_u
			# (r4_l, res4_l) = sagec.getactualValue(Nterms, tg_t4, p43, eps_1, 1.0, delta_1, dt4_l, off_t4[0])
			# (r4_u, res4_u) = sagec.getactualValue(Nterms, tg_t4, p43, eps_1, 1.0, delta_1, dt4_u, off_t4[1])
			# (r3_l, res3_l) = sagec.getactualValue(Nterms, tg_t3, p45, eps_2, 1.0, delta_2, dt3_l, off_t3[0])
			# (r3_u, res3_u) = sagec.getactualValue(Nterms, tg_t3, p45, eps_2, 1.0, delta_2, dt3_u, off_t3[1])
			res = r4_l*777, r4_u*777 #(r3_l*651, r3_u*651))

			if flag == 1:
				if AUC in [1,2]:
					r1, r2 = r4_l*(1 - np.exp(-p43 * tg_t4)), r4_u*(1 - np.exp(-p43 * tg_t4))
				else: #AUC = 0
					import getFS as fs
					r1, r2 = fs.getActual((tg_t4, p43, r4_l)), fs.getActual((tg_t4, p43, r4_u))
				return r1*777.0, r2*777.0
			else:
				return res

		outputEqns1 = getEquationsFile(outputfile)
		for var in outputEqns1:
			print(var + ' : '+ str(outputEqns1[var]))
			ha1.macros.update({var:outputEqns1[var]})

		outputEqns = {}
		for var in outputEqns1.keys():
			expr = outputEqns1[var]
			for key in reversed(ha1.macros.keys()):
				expr = expr.replace(key, ha1.macros[key])
			for key in ha1.macros.keys():
				expr = expr.replace(key, ha1.macros[key])
			outputEqns.update({var:expr})

			print('updated '+ var + ' : '+ str(outputEqns[var]))

		# if DEBUG:
		print('Step-4: replace outputEqns 2') #, str(ha1))
		print('Step-5: HA before simplify')
			#print(str(ha1))

		sys.stdout.flush()
		ha = ha1.simplify(skip = ['dose_t4', 'dose_t3', 'q_10s'], nrplc=['eta', 'et4s', 'et3s','et4','et3']) 
		# if DEBUG:
		print('Step-6: HA after simplify')
		# print(str(ha.macros))

		sys.stdout.flush()

		print('Step-6.5: temp model saved')
		'''detect parameters'''
		#if DEBUG:
		print('Step-7: model parsed', time.time() - stm, 's')
		sys.stdout.flush()
		
		
		all_params = {}
		if not paramfile == '':
			all_params = getParam(paramfile)
		else:
			inits = {}
			for c in ha.init.condition:
				print('Inits', str(c.literal1), str(c.literal2))
				inits.update({str(c.literal1):str(c.literal2)})

			for var in ha.variables.keys():
				if var == 'time':
					continue
				rng = ha.variables[var]
				if var in all_params.keys() or var not in inits.keys():
					all_params.update({var:rng})

		
		'''detect parameters'''
		print('---- all params ----')
		for par in all_params:
			print(par + ' : '+ str(all_params[par]))
		
		sys.stdout.flush()
		#dataSet = Data(datafile)

		#print(str(pha))

		# param_len = 2 
		param_len = len(all_params.keys())
		subsets_2 = findsubsets(list(all_params.keys()), param_len)
		
		#if PLOT:

		# for sub in subsets_2: #for each subset of size 2 

		sub = list(subsets_2)[0]
		print('For param set', sub, type(sub))
		params = {}
		other_params = {}
		for key in all_params:
			
			# print(' all params: ', key, str(all_params[key]), type(all_params[key]))
			if key in sub:
				params.update({key:all_params[key]})

				print('params: ', key, str(all_params[key]), type(all_params[key]))
				# if FIX_PAR:
				# 	d_iv = '({2}*{0}+(1-{2})*{1})'.format(all_params[key].getleft(), all_params[key].getright(), RATIO)
				# 	#other_params.update({key:Node(d_iv)})
				# 	print('params (FIX): ', key, d_iv, type(d_iv))
			else:
				other_params.update({key:all_params[key]})
				#pha.addInit(key, all_params[key])
				print('params fixed: ', key, str(all_params[key]), type(all_params[key]))

		pha = convertHA2PHA(ha, params)
		for key in other_params:
			pha.addInit(key, other_params[key])

		if FIX:
			for key in all_params:
				pha.addInit(key, getExpression('RAT*low_{0}/s_{0}+ (1-RAT)*up_{0}/s_{0}'.format(key)))
			# if FIX == 1 and TF == 1:
			# 	hypo_val = hypo_par_val * (1+ MAX_HPV)
			# 	d_iv = '({2}*{0}+(1-{2})*{1})'.format(all_params[key].getleft(), all_params[key].getright(), RATIO)
				# 	#other_params.update({key:Node(d_iv)})
				# 	print('params (FIX): ', key, d_iv, type(d_iv))
		print('params', [str(params[k]) for k in params.keys()])
		print('model parameters', str(pha.parameters))
		print('Init: ', str(pha.init))

		'''temp_ha = inFileName+'_'+suf+'temp_{0}.drh'.format(index)
		# temp_ha_preprocessed = inFileName+'_'+suf+'temp_{0}.preprocessed.drh'.format(index)
		print(temp_ha)
		with open(temp_ha, 'w+') as tfp:
			tfp.write(str(pha))

		simulateFile = 'simulateHA_{0}.py'.format(1)
		sim_ha = hode.createSimulator(datafile, temp_ha, outputfile, 0.001, paramdefaultfile)
		f = open(simulateFile, "w+")
		f.write(sim_ha)
		f.close()
		import simulateHA_1 as sim
		
		default_params = sim.getDefaultParams()
		param_names = sim.getParamNames()		
		print('default_params', param_names, default_params)'''

		sys.stdout.flush()
		if TEST:
			exit()
		#eu_sp_TSH,eu_sp_FT4,eu_base_TSH,eu_base_FT4
		# fixed_obs1 = rf['TSH_eu'], rf['FT4_eu']
		# fixed_obs = rf['TSH_es'], rf['FT4_es']
		fixed_obs1 = rf['eu_sp_TSH'], rf['eu_sp_FT4']
		fixed_obs = rf['eu_base_TSH'], rf['eu_base_FT4']
		#(satparam, unsatparam, undetparam) = getEstimatedParameters(pha, params, dataSet, d, k)
		satparam = getEstimatedParameters(pha, params, d, k_length, datafile, (fixed_obs, fixed_obs1), outputEqns)
		# checked_range = fixed_obs1[0]*(1-DATA_NOISE), fixed_obs1[0]*(1+DATA_NOISE), fixed_obs1[1]*(1-DATA_NOISE), fixed_obs1[1]*(1+DATA_NOISE)
		checked_range = [(fo*(1-DATA_NOISE), fo*(1+DATA_NOISE)) for fo in fixed_obs1]
		ac_b =  getActualDosages(satparam[0]) if len(satparam) > 0 else  (-1, -1)
		ac_b1 =  getActualDosages(satparam[0], flag =1) if len(satparam) > 0 else (-1, -1)
		res_csv = open(all_res_file, 'a+')
		wr = ['{0}'.format(rf[rr]) for rr in res_header]+ [ac_b[0], ac_b[1]]+ [ac_b1[0], ac_b1[1]] + [cr for cr in checked_range] + [MAX_HPV]

		res_writer = csv.writer(res_csv, delimiter=',')
		res_writer.writerow(wr)
		print('########## row {0} given {1} calculated {2}'.format(ss_rn, rf['LT4'], satparam, rf))
		sys.stdout.flush()
		res_csv.close()

		print('SAT boxes ---')
		for b in satparam:
			ac_b =  getActualDosages(b)
			ac_b1 =  getActualDosages(b, flag = 1)
			print('getActualDosages', ac_b, ac_b1)
		print('##################################')
		# break
		# if PLOT:
	res_csv.close()
	'''pp = PdfPages(plotName)
	for fig in figs_to_plot:
		pp.savefig(fig)
	pp.close()'''
		
def getPropSMT(tp, pid, i, neg):
	fn ='temp_'+suf+'{0}_'.format(index)+str(tp)+'_'+str(rank)+'_'+str(pid)+'_'+str(i)
	if neg == 1:
		fn = 'tempC1_'+suf+'{0}_'.format(index)+str(tp)+'_'+str(rank)+'_'+str(pid)+'_'+str(i)
	elif neg == 2:
		fn = 'tempC2_'+suf+'{0}_'.format(index)+str(tp)+'_'+str(rank)+'_'+str(pid)+'_'+str(i)
	return fn
	
def getSAT(model, i, tp = 0, pid = 0, neg = False):
	# if neg:
	# if neg:
	# 	fn = 'tempC_p_'+str(rank)+'_'+str(pid)+'_'+str(i)
	# else:
	# 	fn = 'temp_p_'+str(rank)+'_'+str(pid)+'_'+str(i)
	# fname = os.path.join(tempfolder, fn+'.smt2.model')

	fn = getPropSMT(tp, pid, i, neg)
	fname = os.path.join(tempfolder, fn+'_0_0.smt2.model')
	#print('getSAT', fname)
	if DEBUG:
		print('Reading sat instance :', fname)
	satinstance = parseInstance(fname)
	#print(satinstance.variables[0])
	satinstance.addModel(model)
	satinstance.addDepth(i)
	return satinstance
	
# checkproperty(model, (propNeg, mode), MIN_DELTA, PATH_LEN, neg = True)
def checkproperty(params):
	model, prop_mode, delta, klen, tp, pid, neg = params
	# def checkproperty(model, prop_mode, sbox, delta, klen, tp, pid = 0, neg = False):
	prop, mode = prop_mode[0], str(prop_mode[1])
	#g = model.getGraph()
	st = model.init.mode
	sk = ''
	# s = ''
	# for it in model.variables:
	# 	s += it + ':' + str(model.variables[it])+ ' , '
	# print('##checkproperty', 'model.variables: ', s, 'model.init', str(model.init), 'model -- props')
	#smts = []
	i = 0
	res = (UNSAT, i, '')
	# if neg:
	# 	for pp in prop:
	# 		model.addGoals(mode, pp)
	# else:
	model.addGoals(mode, prop)
	fn = getPropSMT(tp, pid, i, neg)
	print('checkproperty: for dosage value: ', model.macros['et4'])
	fname = os.path.join(tempfolder, fn+'.drh')
	fname_pp = os.path.join(tempfolder, fn+'.preprocessed.drh')
	with open(fname, 'w+') as of:
		of.write(str(model))
	# print('checkproperty', fname)
	# st = [dReachCmd, "-k", str(klen), "-z", fname, "--precision", str(delta), "--model"] 

	# st = [dReachCmd, "-k", str(klen), "-z", fname, "--precision", str(delta), "--ode-step", str(0.001), "--model"] 
	if LOOP == 1:
		st = [dReachCmd, "-l", str(klen[0]), "-k", str(klen[1]), "-z", fname, "--precision", str(delta), '--ode-parallel', '--ode-cache',"--model"] 
	else:
		# if scale == 1.0:
		# 	st = [dReachCmd, "-k", str(klen[1]), "-z", fname, "--precision", str(delta), "--ode-step", str(0.01), '--ode-parallel', '--ode-cache',"--model"] 
		# else:
		# st = [dReachCmd, "-k", str(klen[1]), "-z", fname, "--precision", str(delta),  "--ode-step", str(0.001), "--model"]
		st = [dReachCmd, "-k", str(klen[1]), "-z", fname, "--precision", str(delta),  "--model"] #, "--ode-order", str(10)] #'--ode-parallel', '--ode-cache', 
	#st = [dReachCmd, "-k", str(klen), "-z", fname, "--precision", str(delta), "--model"] 
	# st = [dReachCmd, "-k", str(klen), "-z", fname, "--precision", str(delta), '--parallel', '--ode-parallel', "--model"] #--ode-step", str(0.2), '--ode-cache', '--parallel', '--ode-parallel', '--model']
	# if DEBUG:
	sk += '\t----- '+str(' '.join(st))
	#print('\t----- '+str(st))
	# p = subprocess.Popen(st, stdout=subprocess.PIPE)
	# (output, err) = p.communicate(timeout=1200)  
	'''This makes the wait possible'''
	# p_status = p.wait()	
	# out = p_status

	# p = subprocess.run(st) #, capture_output=True)
	try:
		output =  subprocess.check_output(st, stderr=subprocess.STDOUT) #, timeout=3*3600)
		out = 0
	# except subprocess.TimeoutExpired as e1:
	# 	out = A_UNSAT
	# 	output = e1.stdout
		
	except subprocess.CalledProcessError as e:
		out = e.returncode
		output = e.stdout
	except Exception as e:
		print('Running call again....')
		if DEBUG:
			print('\t----- '+str(st))
		# print('\t----- '+str(st))
		# p = subprocess.Popen(st, stdout=subprocess.PIPE)
		# (output, err) = p.communicate(timeout=1200)  
		'''This makes the wait possible'''
		# p_status = p.wait()	
		# out = p_status

		# p = subprocess.run(st) #, capture_output=True)
		try:
			output =  subprocess.check_output(st, stderr=subprocess.STDOUT)#, timeout=2*3600)
			out = 0
		except subprocess.CalledProcessError as e:
			out = e.returncode
			output = e.stdout
	# if DEBUG:
	# 	print('dReal res:', out, output)

	# start_time = time.time()
	# end_time = time.time()

	# if end_time - start_time:
	# 	p.kill()
	# 	print('Killed following call to dReal... ')#, st)
	# 	print('Running call again....')	
	# 	print('\t----- '+str(st))
	# 	(output, err) = p.communicate()
	# 	'''This makes the wait possible'''
	# 	p_status = p.wait()	

	'''This will give you the output of the command being executed'''
	# if DEBUG:
	sk +=  "\n\t----- Output: " + str(out) + ', depth -- ' + str(klen) +  ', out : '		
	
	sys.stdout.flush()
	sys.stderr.flush()
	
	if(out == SAT or b'delta-sat' in output):
		sk +=  ' = delta-sat '+ str(SAT)
		res = (SAT, i, output)
		print(sk)
		return res, neg #, fname_pp
	elif(out == UNSAT or b'unsat' in output):
		sk +=  ' = unsat'
		res = (UNSAT, i, '')
		print(sk)
		return res, neg #, fname_pp
	# elif(out == A_UNSAT or b'unsat' in output):
	# 	sk +=  ' = assumed unsat after timeout'
	# 	res = (A_UNSAT, i, '')
	# 	print(sk)
	# 	return res, neg #, fname_pp
	else:				
		res = (UNKNOWN, i, '')
		sk +=  ' = ERROR'
		sk += str(output)
		print(sk)
		return res, neg #, fname_pp
	
		

def updateModel(model, sbox):
	#print('updateModel', box, 'model.params: ')
	#for it in model.parameters:
	#	print(it, str(model.parameters[it]))
	edges = sbox.get_map()
	for it in edges:
		intrvl = edges[it]
		# if DEBUG:
		# 	print('InUpdate Model: ', intrvl.leftBound(), str(intrvl.leftBound()))
		param = Range(Node(intrvl.leftBound()), Node(intrvl.rightBound()))
		if it in model.parameters:
			model.parameters.update({it: param})
		if it in model.variables:
			model.variables.update({it: param})
	s = ''
	for it in model.parameters:
		s += it + ':' + str(model.parameters[it])+ ' , '
	if DEBUG:
		print('updatedModel', 'model.params: ', s)
	#return model
		
def getNegatedProperty(prop, instance = None):	
	#print('getNegatedProperty:'+'prop: ', prop.to_prefix())
	propneg = prop
	#print('prop: ', prop.to_prefix())
	# if DEBUG:
	# 	print('negated prop: ', propneg) #.to_prefix())
	return propneg
	
def getProperties(propstr, mode): #, dtype):	
	(pp, pn1, pn2) = propstr
	print(mode, 'getProperties', len(propstr))
	prop = []
	for p in pp:
		cond1 = []
		for p1 in p:
			cond1.append(getCondition(p1))
		prop.append(cond1)

	propNeg1 = []
	for p in pn1:
		cond1 = []
		for p1 in p:
			cond1.append(getCondition(p1))
		propNeg1.append(cond1)
	
	propNeg2 = []
	for p in pn2:
		cond1 = []
		for p1 in p:
			cond1.append(getCondition(p1))
		propNeg2.append(cond1)

	if DEBUG:
		print('prop: ', prop)
		print('negated prop: ', propNeg1, propNeg2)
	
	return (prop, propNeg1, propNeg2, mode)

def getBox(params):
	edges = {}
	for par in params:
		rng = params[par]
		left = rng.leftVal()
		right = rng.rightVal()
		it = PyInterval(left, right)
		#it.mark()
		edges.update({par: it})	
		
	sbox = Box(edges)
	return sbox

def getPropertyFromData(data, fixed, outputEqns):
	fixed_obs, fixed_eusp = fixed
	print('getPropertyFromData', 'data', data, 'fixed_eusp', fixed_eusp, 'fixed_obs', fixed_obs)
	#tsh_eu = fixed_obs[0]
	Y = data
	mode = 1 if MS == 0 else 6
	tm = 47.0*24/scale #9.8 #data[1]
	data_noise = 0.01 #1 # DATA_NOISE
	# for key in outputEqns.keys():
	# 	data_noise.update({key:0.4}) #DATA_NOISE})
	
	time_noise = 0.005 #DATA_NOISE*0.06
	# tm_0, tm_1  = tm - 10.0, tm + 10.0 
	if LOOP == 1:
		tm_0, tm_1  = 47.0*24/scale, 50.0*24/scale#9.8, 9.99 #tm*(1-time_noise), tm*(1+time_noise)
	else:
		tm_0, tm_1  = 49.0*24/scale, 50.0*24/scale
		if INRANGE:
			tm_0, tm_1  = 48.0*24/scale, 50.0*24/scale
	print('tm', tm, tm_0, tm_1)
	'''prop = '((mode = {0}) & (tm > {1}) & (tm < {2}) & ('.format(mode, tm_0, tm_1)
	propn = '((mode = {0}) & (tm > {1}) & (tm < {2}) & ! ('.format(mode, tm_0, tm_1)
	st = 2 '''
	#prop = '((tm = {0}) & ('.format(tm)
	#propn = '((tm = {0}) & ! ('.format(tm)
	#prop = '((mode = {2}) & (tm > {0}) & (tm < {1}) & ('.format(tm_0, tm_1, mode)
	#propn = '((mode = {2}) & (tm > {0}) & (tm < {1}) & ! ('.format(tm_0, tm_1, mode)
	# prop = '((mode = {0}) & (tm > {1}) & (tm < {2}) & ('.format(mode, tm_0, tm_1)
	# propn = '((mode = {0}) & (tm > {1}) & (tm < {2}) & ! ('.format(mode, tm_0, tm_1)
	# propn1 = '((mode = {0}) & (tm > {1}) & (tm < {2}) & ! ('.format(mode, tm_0, tm_1)
	# propn2 = '((mode = {0}) & (tm > {1}) & (tm < {2}) & ! ('.format(mode, tm_0, tm_1)
	if MS == 0:
		prop = ['(tm > {0})'.format(tm_0), '(tm < {0})'.format(tm_1)]
		propn1 = ['(tm > {0})'.format(tm_0), '(tm < {0})'.format(tm_1)]
		propn2 = [ '(tm > {0})'.format(tm_0), '(tm < {0})'.format(tm_1)]
	else:
		prop = ['(tm >= {0})'.format(tm)]
		propn1 = ['(tm >= {0})'.format(tm)]
		propn2 = [ '(tm >= {0})'.format(tm)]
	# prop = '((mode = {0}) & ('.format(int(mode))q_o
	# propn = '((mode = {0}) & ! ('.format(int(mode))
	st = 2
	i = st
	j = 0
	for key in outputEqns.keys():
		if i < len(data) :
			# if data[i] >= 0:
			k10, k11, k12, k13 = fixed_eusp[j], fixed_obs[j], (1-DATA_NOISE), (1+DATA_NOISE)
			# k10, k11, k12, k13 = fixed_eusp[j], fixed_obs[j], fixed_obs[j]*(1-DATA_NOISE), fixed_obs[j]*(1+DATA_NOISE)
			k21, k22, k23 = ((data[i]+data[i+1])/2, data[i], data[i+1]) #*(1-data_noise), data[i+1]*(1+data_noise))
			# k1 = min(min(max(k12, k22), k21), min(k13, k23)) #, fixed_obs[j]*(1-data_noise))
			# k2 = max(max(k12, k22), max(min(k13, k23), k21)) #, fixed_obs[j]*(1+data_noise))
			#k1 = min(k10*k12, k11) # min of eusp and euobs but not crossing TSH min
			#k2 = max(k10*k13, k11) # max of eusp and euobs but not crossing TSH max
			#k1 = k22 #max(min(k10*k12, k11*k12), k22) # min of eusp and euobs but not crossing TSH min
			#k2 = k23 #min(max(k10*k13, k11*k13), k23) # max of eusp and euobs but not crossing TSH max

			# k1 = max(min(k10*k12, k11*k12), k22) # min of eusp and euobs but not crossing TSH min
			# k2 = min(max(k10*k13, k11*k13), k23) # max of eusp and euobs but not crossing TSH max
			k1 = k10*k12 # min of eusp and euobs but not crossing TSH min
			k2 = k10*k13 # max of eusp and euobs but not crossing TSH max
			# else:				
				# k1, k2 = (data[i]*(1+data_noise), data[i+1]*(1-data_noise))
			#eqn = key #
			eqn = outputEqns[key]
			# pr = '(({0}) > {1}) & (({0}) < {2})'.format(eqn, k1, k2)

			pr = ['(({0}) >= {1})'.format(eqn, k1),'(({0}) <= {1})'.format(eqn, k2)]

			
			pr1 = ['(({0}) < {1})'.format(eqn, k1)] #*(1-data_noise))]
			pr2 = ['(({0}) > {1})'.format(eqn, k2)] #*(1+data_noise))]

			if j > 0:		# for Ft4
				pr1 = ['(({0}) > {1})'.format(eqn, k2)] #*(1+data_noise))]
				pr2 = ['(({0}) < {1})'.format(eqn, k1)] #*(1-data_noise))]

			print(key, 'target', k1, k2, 'noisy', k1*(1-data_noise), k2*(1+data_noise), 'others', k10*k12, k10*k13)

			# prop += pr if i == st else ' & '+ pr 
			# propn += pr if i == st else ' & '+ pr 
			prop += pr
			propn1 += pr1
			propn2 += pr2

			# propn1.append(pr1)
			# propn2.append(pr2)

		i+= 2
		j += 1
		if j >= 1:
			break
	# prop += '));' 
	# propn += '));' 
	return [prop], [propn1], [propn2], int(mode)

	
if __name__ == "__main__": 	
   main(sys.argv[1:])


