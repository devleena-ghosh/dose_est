from __future__ import print_function
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

#from model.interval import *
#from model.box import *
#from model.condition import *
from util.reach import *
from util.graph import *
from util.stack import *
from util.heap import *
from ha2smt.smtEncoder import *
from paramUtil.interval import *
from paramUtil.box import *
from paramUtil.box_factory import *
from util.parseOutput import *
from util.smtOutput import *
#from paramUtil.readDataBB import *
from model.node_factory import *
from ha2smt.smtEncoder import *

import numpy
import matplotlib
import time
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

import itertools


# fig = plt.figure()

tempfolder = os.path.join('temp','testBox')
SAT = 51
UNSAT = 52
UNKNOWN = -1
TRUE = 1
FALSE = 0
UNDET = 2
ONLYMARKED = 1
NOMARK = 0
EPS = 0.001
noise = {}
DEBUG = False
dRealCmd = "dReal"



def rational_cf(x, y, names = ['x', 'y'], order = 3):
	def getEqn(popt):
		num = []
		for i in range(order):
			num.append(names[0]+'*'+str(popt[i]))
		num = '+'.join(num)
		den = []
		for i in range(order-1):
			den.append(names[0]+'*'+str(popt[i+order]))
		den = '+'.join(den)
		s = num + '/' + den
		return s

	def rational(x, p, q):
		"""
		The general rational function description.
		p is a list with the polynomial coefficients in the numerator
		q is a list with the polynomial coefficients (except the first one)
		in the denominator
		The zeroth order coefficient of the denominator polynomial is fixed at 1.
		Numpy stores coefficients in [x**2 + x + 1] order, so the fixed
		zeroth order denominator coefficent must comes last. (Edited.)
		"""
		return np.polyval(p, x) / np.polyval(q + [1.0], x)

	def rational3_3(x, *params): #p0, p1, p2, q1, q2):
		p = [params[i] for i in range(order)]
		q = [params[order+i] for i in range(order-1)]

		return rational(x, p, q)
		# calculate bic for regression
	def calculate_bic(n, mse, num_params):
		bic = n * log(mse) + num_params * log(n)
		return bic

	p = [0.2, 0.3, 0.5]
	q = [-1.0, 2.0]
	pre_bic = 99999999
	preopt = []
	for i in range(order):
		params = [p[j] for j in range(i+1)] + [q[j] for j in range(i)]
		popt, pcov = curve_fit(rational3_3, x, y, p0=tuple(params))
		yhat = rational3_3(x, *popt)
		mse = mean_squared_error(y, yhat)
		num_params = len(popt)
		bic = calculate_bic(len(y), mse, num_params)
		if bic > pre_bic:
			print('chosen order', i-1, preopt)
			break
		pre_bic = bic
		preopt = popt
	popt = preopt
	f = plt.figure()
	plt.plot(x, y, label='original')
	plt.plot(x, ynoise, '.', label='data')
	plt.plot(x, rational3_3(x, *popt), label='fit')

	return popt, getEqn(popt), f


def main(argv):
	k = 10
	d = 0.0001
	
	#global EPS
	#EPS = 1 * d
	
	# inputfile = sys.argv[1]
	# paramfile = sys.argv[2]
	# datafile = sys.argv[3]
	paramfile = ''
	
	try:
		opts, args = getopt.getopt(argv,"hi:p:o:d:",["ifile=","pfile=", "ofile=", "dfile="])
	except getopt.GetoptError:
			print("Box.py -i <inputfile> -p <paramFile> -o <outputfile> -d <dataFile>")
			sys.exit(2)
			
	for opt, arg in opts:
		if opt == '-h':
			print("Box.py -i <inputfile> -p <paramFile> -o <outputfile> -d <dataFile>")
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
		
	
	# ha = getModel(inputfile)
	# print('model parsed')
	#print(str(ha))

	ha1 = getModel(inputfile)

	outputEqns = getEquationsFile(outputfile)
	for var in outputEqns:
		print(var + ' : '+ str(outputEqns[var]))
		ha1.macros.update({var:outputEqns[var]})
	#print(str(ha1))
	ha = ha1.simplify()
	#print(str(ha))
	'''detect parameters'''
	print('model parsed')
	

	all_params = {}
	if not paramfile == '':
		all_params = getParam(paramfile)
	else:
		inits = {}
		for c in ha.init.condition:
			# print(str(c.literal1), str(c.literal2))
			inits.update({str(c.literal1):str(c.literal2)})

		for var in ha.variables.keys():
			if var == 'time':
				continue
			rng = ha.variables[var]
			if var in all_params.keys() or var not in inits.keys():
				all_params.update({var:rng})
		
	
	'''detect parameters'''
	# for par in all_params:
	# 	print(par + ' : '+ str(all_params[par]))
		
	#dataSet = Data(datafile)

	pha = convertHA2PHA(ha, all_params)
	#print(str(pha))

	def findsubsets(S,m):
		return set(itertools.combinations(S, m))

	param_len = 2 #len(all_params.keys())
	subsets_2 = findsubsets(list(all_params.keys()), param_len)
	
	pp = PdfPages('plotBox.pdf')
	for sub in subsets_2: #for each subset of size 2 
		print('For param set', sub, type(sub))
		params = {}
		# other_params = {}
		for key in sub:
			# print(key)
			params.update({key:all_params[key]})
			d_iv = '(0.5*({0}+{1}))'.format(all_params[key].getleft(), all_params[key].getright())
			# other_params.update({key:d_iv})
			pha.addInit(key, d_iv)
			print('params fixed: ', key, d_iv)
		print(params)

		#(satparam, unsatparam, undetparam) = getEstimatedParameters(pha, params, dataSet, d, k)
		(satparam, unsatparam, undetparam) = getEstimatedParameters(pha, params, d, k, datafile, outputEqns)
		print('##################################')
		print('SAT boxes ---')
		with open(outfile, 'w') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',')
			
			i = 0
			names = []
			for b in satparam:
				b_map = b.get_map()
				row = []
				for key in sorted(b_map.keys()):
					row.append(key)
					names.append(key)
				i += 1			
				spamwriter.writerow(row)
				if i > 0:
					break

			xdata = []
			for b in satparam:
				b_map = b.get_map()
				row = []
				xr = []
				for key in sorted(b_map.keys()):
					row.append((b_map[key].leftBound(), b_map[key].rightBound()))
					xr.append((b_map[key].leftBound()+b_map[key].rightBound())/2)
				xdata.append(xr)
				spamwriter.writerow(row)

		X = [xdata[i][0] for i in range(xdata)]
		Y = [xdata[i][1] for i in range(xdata)]

		popt, eqn, f = rational_cf(X, Y, names)
		print(names, popt, eqn)

		print('##################################')
		# break
		sbox = getBox(params)
		x_scale = [1.0, 1.0]
		b_edges = sbox.get_map()
		lb = len(b_edges)
		n_combs = []
		keys = sorted(b_edges.keys())
		for i in range(lb):
			for j in range(i+1, lb):
				comb = (keys[i], keys[j])
				n_combs.append(comb)
		c = 0
		for combs in n_combs:
			x = []
			w = []
			axisname = []
			i = 0
			for it in combs:
				s = x_scale[i]
				intrvl = b_edges[it]
				x.append(intrvl.leftBound()*s)
				w.append(intrvl.width()*s)
				axisname.append(it)
				i += 1
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
		
			print('SAT ---', c)
			i = 0
			for b in satparam:
				#print('class: ', str(p))
				# for b in b1:
				# print(str(b), b.max_side_width(), EPS)
				plotBox(currentAxis, b, combs, TRUE)
				i+= 1
			print(i)

			print('UNSAT ---', c)
			i = 0
			for b in unsatparam:
				#print('class: ', str(p))
				# print('box: ')
				#for b in unsatparam[p]:
				# for b in b1:
				#print(str(b))
				plotBox(currentAxis, b, combs, FALSE)		
				i+= 1
			print(i)
			
			print('UNDET ---', c)
			i = 0
			for b in undetparam:
				#print('class: ', str(p))
				# print('box: ')
				#for b in undetparam[p]:
				# for b in b1:
				#print(str(b), b.max_side_width(), EPS)
				plotBox(currentAxis, b, combs, UNDET)
				i+= 1
			print(i)
			
			# boxset = undetparam[p]
			# mergeSet = mergeBoxList(boxset)
			# i = 0
			# for b in mergeSet:
			# 	print(str(b))
				#i += 1
				#if(i == 2):
					#break
			#print(paramValues)
			pp.savefig(f)
			pp.savefig(fig_t)
			c+= 1
	pp.close()

# someX, someY = 0.5, 0.5
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

def checkproperty(model, prop, sbox, delta, k, pid = 0, neg = False):
	g = model.getGraph()
	st = model.init.mode
	s = ''
	for it in model.variables:
		s += it + ':' + str(model.variables[it])+ ' , '
	#print('##checkproperty', 'model.variables: ', s)
	if DEBUG:
		print('delta: ', delta)
	smts = []
	i = 0
	for path in g.getKPaths(st, k):
		# print('checkproperty', path)
		depth = len(path)
		#smt = ''
		smtEncode = generateSMTforPath(model, path, delta)	
		propertySmt = to_SMT(prop, smtEncode)
		# print('checkproperty', propertySmt)
		#propertySmtneg = Node(prop).negate().toSMT()
		#if(neg):
		#	smtEncode.addGoal(propertySmtneg)
		#else:
		smtEncode.addGoal(propertySmt)			
		smt = smtEncode.toString(neg)
		# print('2.######')
		#if(i == 1):
		#	break
		fn = 'temp_p'+str(pid)+'_'+str(i)
		if(neg):
			fn = 'tempC_p'+str(pid)+'_'+str(i)
		fname = os.path.join(tempfolder, fn+'.smt2')
		with open(fname, 'w') as of:
			of.write(smt)
		i +=1		
		#sys.stdout.flush()
		
		st = [dRealCmd, fname, "--precision", str(delta), '--model']
		
		if DEBUG:
			print('\t----- '+str(st))
		# p = subprocess.Popen(st, stdout=subprocess.PIPE)
		# (output, err) = p.communicate(timeout=1200)  
		'''This makes the wait possible'''
		# p_status = p.wait()	
		# out = p_status

		# p = subprocess.run(st) #, capture_output=True)
		try:
			output =  subprocess.check_output(st, timeout=6*3600)
			out = 0
		except subprocess.CalledProcessError as e:
			out = e.returncode
			output = e.stdout
		except Exception as e:
			print('Running call again....')	
			if DEBUG:
				print('\t----- '+str(st))
			try:
				output =  subprocess.check_output(st)
				out = 0
			except subprocess.CalledProcessError as e:
				out = e.returncode
				output = e.stdout
		# print(out, output)

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
		if DEBUG:
			print ("\t----- Output: " + str(out), 'depth ', depth) #,  'out', output)		
		
		sys.stdout.flush()
		
		if(out == 0 and b'delta-sat' in output):
			if DEBUG:
				print('delta-sat')
			return (SAT, depth-1, output)
		elif(out == 0 and b'unsat' in output):
			if DEBUG:
				print('unsat')
			res = (UNSAT, depth-1, '')
		else:				
			res = (UNKNOWN, depth-1, '')
			if DEBUG:
				print('ERROR')
			return res
	return res	
		
def evaluate(args):
	model, prop, sbox, d, k, pid = args
	if DEBUG:
		print(prop, sbox, d, k, pid)
	sk = 'Evaluate - Pid: ' + str(pid) +' Checking box : ' + str(sbox)
	''' delta is a fraction of min dimension of the box'''
	delta = d #box.min_side_width() * d
	
	updateModel(model, sbox)
	#propNode = getSTL(prop)[0]
	Instance = collections.namedtuple('Instance', ['p1', 'p2'])
	(propNode, propNeg1) = prop
	if DEBUG:
		print('In evaluate') #,\t prop : '+str(propNode)+'\n\t negProp : '+str(propNeg1) )
		print('@@ prop: '+ str(propNode),  str(propNeg1))
	(res1, i1, out1) = checkproperty(model, propNode, sbox, delta, k, pid)
	# print('1. ######')
	flag = False
	if(res1 == UNSAT):
		ret = (FALSE, None)
		sk += ' -- FALSE, None'
	elif (res1 == SAT):
		instance1 = getSAT(model, i1, pid, False) #;getSATPoint(i1)
		propNeg = getNegatedProperty(propNeg1, instance1)
		if DEBUG:
			print('@@ NegProp: '+ str(propNeg))
		#propNeg = getSTL(propneg)[0]
		(res2, i2, out2) = checkproperty(model, propNeg, sbox, delta, k, pid, neg = True)	
		if (res2 == UNSAT):
			ret = (TRUE, None)
			sk += ' -- TRUE, None'
		elif (res2 == SAT): #if(res1 == SAT and res2 == SAT):		
			instance2 = getSAT(model, i2, pid, True) #.getSATPoint(i2)
			#if(instance1 == instance2):
			b1 = instance1.getSATBox()
			b2 = instance2.getSATBox()
			instance = Instance(p1 = instance1, p2 = instance2)
			if b1.adjacent(b2):
				ret = (UNDET, instance)
				sk += ' -- UNDET, adjacent ' #[' + str(b1)+ '], [' + str(b2)+']'
				#print(str(instance1), str(instance2))
				#flag = True
			elif b1.intersects(b2):
				b3 = boxIntersection(b1, b2)
				if b1.fullyContains(b3) and b2.fullyContains(b3):
					ret = (UNDET, None)
					sk += ' -- UNDET, Fully Intersects ' #[' + str(b1)+ '], [' + str(b2)+']'
					#print(str(instance1), str(instance2))
					#flag = True
				else:
					ret =  (UNDET, instance)
					sk += ' -- UNDET, Intersects' # [' + str(b1)+ '], [' + str(b2)+']'
			else:
				ret =  (UNDET, instance)
				sk += ' -- UNDET, Instance' # [' + str(b1)+ '], [' + str(b2)+']'
		else: #UNKNOWN
			ret =  (UNDET, None)
			sk += ' -- UNDET, None'
	else:
		ret = (UNDET, None)
		sk += ' -- UNDET, None'
	#sys.stdout.flush()
	print(sk)
	'''
	if POOL:
		queue.put((pid, ret))
		# queue.put(ret)
		# queue.close()
		# queue.put_nowait((pid, ret))
	else:
		queue.put((pid, ret))
		# queue.put(ret)

	return
	'''
	if flag:
		exit()
	return ret
			
def getSAT(model, i, pid = 0, neg = False):
	if neg:
		fn = 'tempC_p'+str(pid)+'_'+str(i)
	else:
		fn = 'temp_p'+str(pid)+'_'+str(i)
	fname = os.path.join(tempfolder, fn+'.smt2.model')
	if DEBUG:
		print('Reading sat instance :', fname)
	satinstance = parseInstance(fname)
	#print(satinstance.variables[0])
	satinstance.addModel(model)
	satinstance.addDepth(i)
	return satinstance
		
def getBoxes(model, prop, sbox, d, k, act_pt = None):	
	#mgr = multiprocessing.Manager()
	
	sat_box = []
	unsat_box = []
	undet_box = []
	
	stack =  Stack() 
	stack.push(sbox)
	# List = [box]
	# pool = multiprocessing.Pool(processes=POOL)
	np1 = int(multiprocessing.cpu_count()-2)
	print('##### You have {0:1d} CPUs'.format(np1+1))
	sys.stdout.flush()
	i = 0	
	while(not stack.isEmpty()):
	# while(len(List) > 0):
		#b = stack.pop()
		# np = 1 #
		np = min(np1, stack.size())
		POOL = True if np > 1 else False
		#np = 1 # min(np1, stack.size())
		#POOL = False #True if np > 1 else False
		print('Using multiprocessing : ', POOL, np)
		allboxes = [stack.pop() for j in range(np)]		
		if POOL:				
			# models = [model for j in range(np)]		
			# props = [prop for j in range(np)]		
			# deltas = [d for j in range(np)]		
			# ks = [k for j in range(np)]
			'''
			# queue = multiprocessing.Queue()
			queue = multiprocessing.SimpleQueue()
			jobs = []
			results = {}
			for j in range(np):
				job = multiprocessing.Process(target=evaluate, args=(model, prop, allboxes[j], d, k, queue, POOL, j))
				job.start()				
				jobs.append(job)
				sys.stdout.flush()
				# pid, ret = queue.get_nowait() # will block
				# pid, ret = queue.get() # will block
				# results.update({pid:ret})
			print('No of jobs: ', np, len(jobs), len(allboxes))

			sys.stdout.flush()
			for job in jobs:
				# job.join()
				# print(job.is_alive())
				pid, ret = queue.get() # will block
				results.update({pid:ret})
			for job in jobs:
				job.join()
			# 	if not job.is_alive():
			# 		pid, ret = queue.get() #_nowait() # will block
			# 		results.update({pid:ret})
			print(results)
			sys.stdout.flush()
			'''
			pool = ProcessingPool(np)
			inputs = [[model, prop, allboxes[j], d, k, j] for j in range(np)]
			#print(inputs)
			results = pool.map(evaluate, inputs)
		else:
			results = [evaluate([model, prop, allboxes[j], d, k, 0]) for j in range(np)]

		for j in range(np):		
			b = allboxes[j]	
			#print(results)
			sys.stdout.flush()
			(r, instance) = results[j]		
			
			if DEBUG:
				print('Box min: '+ str(b.min_side_width())+ ' max :' + str(b.max_side_width()))
				print('Result: ', r, instance)


			if(r == FALSE):
				'''if given range subset of b then return false '''
				#return (False, b)
				unsat_box.append(b)
				print('@@@@@@@@@@@@@@@@@@ False box : ', str(b))
				#break
			elif(r == TRUE):	
				''' if given range subset of b then return true '''
				sat_box.append(b)
				print('@@@@@@@@@@@@@@@@@@ True box : ', str(b))
				#return (True, b)
				# break
			else: 
				''' UNDET '''
				if(instance is not None):
					if DEBUG:
						print('Undet box : ', str(b))
					if(b.max_side_width() < EPS):
						inst1 = instance.p1
						b1 = inst1.getSATBox()
						sat_box.append(b) # whole box us sat
						#print(' delta SAT box')
						
						#undet_box.append(b)
						print('@@@@@@@@@@@@@@@@@@ < EPS sat box : ', str(b))
						#if(act_pt is not None and b.contains(act_pt)):
						#	print('Valid box '+ str(b)+ 'removed')
						#	break;
					else:
						delta1 = d #min(sbox.minDimension() * 0.1, d)
						boxes = heuristicPartition(b, instance, delta1)
						if(len(boxes) > 0):
							for b1 in boxes:
								b2 = b1[1]
								#print(b1, type(b1), b2, type(b2))
								if b1[0] == 2:
									print('@@@@@@@@@@@@@@@@@@ Cover Undet box : ', str(b))
									undet_box.append(b)
								else:
									if DEBUG:
										print('-------------------------- in queue', str(b2))
									stack.push(b2)
						else: # |b| < EPS
							inst1 = instance.p1
							#b1 = inst1.getSATBox()
							#sat_box.append(b1)
							#print(' delta SAT box')
							
							undet_box.append(b)
							print('@@@@@@@@@@@@@@@@@@ not divided Undet box : ', str(b))
							#break
							#if(act_pt is not None and b.contains(act_pt)):
							#	print('Valid box '+ str(b)+ 'removed')
							#	break
				else:
					# undet_box.append(b)
					print('@@@@@@@@@@@@@@@@@@@@@@ Error box : ', str(b), '--- added again if box is bigger')
					if (b.max_side_width() < EPS):
						undet_box.append(b)
					else:
						if DEBUG:
							print('-------------------------- in queue', str(b))
						stack.push(b)
		#continue
		#if i == 1:
		#	break	
		i += 1
	return (sat_box, unsat_box, undet_box) 		
	
def updateModel(model, sbox):
	#print('updateModel', box, 'model.params: ')
	#for it in model.parameters:
	#	print(it, str(model.parameters[it]))
	edges = sbox.get_map()
	for it in edges:
		intrvl = edges[it]
		if DEBUG:
			print('InUpdate Model: ', intrvl.leftBound(), str(intrvl.leftBound()))
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
			
'''
In case of adjacent sat instances.. divide the box through the adjacent dimension
intersecting boxes -- error box -- remove it
smaller box -- cover it and remove
otherwise, divide along the middle point

'''
def heuristicPartition(sbox, instance, delta):
	boxesWPriority = []
	boxes = []
	
	inst1 = instance.p1
	inst2 = instance.p2
	
	point = inst1.getSATPoint()
	negpoint = inst2.getSATPoint()

	satb = inst1.getSATBox()
	negsatb = inst2.getSATBox()

	

	if DEBUG:
		print('points:', point, negpoint)
	
	if point.empty() or negpoint.empty():
		return []
	global noise
	emap = {}
	b_map = sbox.get_map()
	for it in b_map.keys():		
		emap.update({it: PyInterval(0.01 * noise[it])})

	adjacentEdges = satb.adjacentEdges(negsatb)
	adjacent = False
	if len(adjacentEdges) > 0:
		for it in adjacentEdges:
			#print('adjacent', it)
			intrvl = b_map[it]
			intrvl.mark()
		adjacent = True
	#emap = {}
	#b_map = sbox.get_map()
	#for it in b_map.keys():		
	#	emap.update({it: PyInterval(EPS)})
	
	if DEBUG:
		print('heuristicPartition: Box = ', str(sbox)) 
	
	''' middle point between propSat and propNegSat '''
	mPoint = middlePoint(point, negpoint) 
	
	if adjacent:
		boxes = bisect1D(sbox, emap, ONLYMARKED, mPoint) # adjacent sat instances.. divide the box through the adjacent dimension
		#print('adjacent', boxes)
	else:
		md = distance(point, negpoint)
		if DEBUG:
			print('Distance : ', md)
		if ( md > delta): # Prop and PropNeg SAT points may be partitioned into diff box
			if DEBUG:
				print('Distance between two points is greater than Delta')
			if sbox.contains(mPoint):
				if DEBUG:
					print('Box contains middle point')
				boxes = bisect1D(sbox, emap, NOMARK, mPoint)
			else:
				boxes = bisect1D(sbox, emap, NOMARK) # less likely scenario , default case
		elif (md > 0.0): 
			''' get the minimum cover box containing 2 SAT points and remove the cover box from actual box and partition'''
			if DEBUG:
				print('Distance between two points is lesser than Delta='+str(delta)) 
			b1 = inst1.getSATBox()
			b2 = inst2.getSATBox()
			''' this is the hull of the propSat and propNegSat boxes '''
			cover = get_cover([b1, b2]) 
			#if(cover.max_side_width() > delta):
			boxes = remove(sbox, cover, emap, NOMARK)
			#else:
			#	''' both the point is in neighboring region ; discard this region'''
			#	print('Cover box < delta')
			#	return []
			#dispPoint = randomDisplacementOpposite(negpoint, point, EPS)
			#if (box.contains(dispPoint)):
			#	boxes = bisect1D(box, emap, NOMARK, dispPoint) # 
			#else:
			#	boxes = bisect1D(box, emap, NOMARK) # default case
			#break;
			# boxes.append(cover)
			boxesWPriority.append((2, cover))
			
		else: 
			''' rare case: both the point is same; discard this region'''
			if DEBUG:
				print('Both points are same')
			b1 = inst1.getSATBox()
			cover = b1.addDelta(delta)
			boxes = remove(sbox, cover, emap, NOMARK)
			# boxes.append(cover)
			#return []
			boxesWPriority.append((2, cover))

	for b in boxes:
		if(not b.empty()):
			item = (1, b)
		# if(b.contains(point)):
		# 	item = (1, b) #queue.minPriority() - 1, b);
		# elif(b.contains(negpoint)):
		# 	item = (1, b) #queue.maxPriority() + 1, b);
		# else:
		# 	item = (1, b) #queue.maxPriority() + 1, b); 
			boxesWPriority.append(item)
	return boxesWPriority
	
def parameterCheck(model, sbox, d, k, p1):
	sat_box = []
	unsat_box = []
	undet_box = []
	i = 0
	#for data in dataSet.getData():
	#	if i == 1:
	#		break
	(prop, propneg) = p1 #convert2Prop(data, dtype) # convert a row to property
	if DEBUG:
		print('Property: ', prop) #, ' State: ', dtype)
		print('PropertyNeg: ', propneg) #, ' State: ', dtype)
	(pn, negpn) = getProperties((prop, propneg)) #, dtype)
	#(negpn, negpn1) = getProperties(propneg, dtype)
	#edges = {}
	#edges.update({'radius': float(data[radius])})
	#edges.update({'g': float(data[g])})
	#act_pt = Point(edges)
	(bt, bf, bu) = getBoxes(model, (pn, negpn), sbox, d, k)
	# print(bt, bf, bu)
	sat_box = bt
	unsat_box = bf
	undet_box = bu
	
	'''if i == 0:
		# sat_box = [bt]
		# unsat_box = [bf]
		# undet_box = [bu]
		sat_box.append(bt)
		unsat_box.append(bf)
		undet_box.append(bu)
	else:
		#sat_box = intersectionBoxMap(sat_box, bt)
		sat_box.append(bt)
		unsat_box.append(bf)
		undet_box.append(bu)'''

	#i = i+1
		
	return (sat_box, unsat_box, undet_box)

def getNegatedProperty(prop, instance):	
	#print('getNegatedProperty:'+'prop: ', prop.to_prefix())
	'''negpn = prop #.negate().to_cnf()
	tm_name = 'tm' #getVarName(instance.getModel(), 'tm')
	mode_name = 'mode'
	#print('getNegatedProperty: '+ str(tm_name))
	tmInst = getVarInstance_t(instance , tm_name)[0]
	tname = instance.getVarName(tmInst)

	modeInst = getVarInstance_t(instance , mode_name)[0]
	modeName = instance.getVarName(modeInst)

	print('getNegatedProperty, '+ str(tm_name) +', '+ str(tmInst), tmInst.endValue.leftVal(), tmInst.endValue.rightVal())	
	print('getNegatedProperty, '+ str(mode_name) +', '+ str(modeInst), modeInst.endValue.leftVal(), modeInst.endValue.rightVal())	
	c1 = Node('>=', [Node(tname), Node(tmInst.endValue.leftVal())])
	c2 = Node('<=', [Node(tname), Node(tmInst.endValue.rightVal())])

	c3 = Node('=', [Node(mode_name), Node(modeInst.endValue.leftVal())])

	clause = Node('&', [c1, c2])
	#clause = Node('&', [c3])
	# propneg = Node('&', [clause, negpn])

	# prop = Node('&', [clause, pn])
	# prop = pn '''
	propneg = prop
	#print('prop: ', prop.to_prefix())
	if DEBUG:
		print('negated prop: ', propneg.to_prefix())
	return propneg
	
def getProperties(propstr): #, dtype):	
	(p, pc) = propstr
	pn = getSTL(p)[0]
	negpn =  (getSTL(pc)[0]).to_cnf() #pc #pn.negate().to_cnf() #(getSTL(pc)[0]).to_cnf() #pn.negate().to_cnf()
	print(pn, negpn)
	print(type(pn), type(negpn))
	#clause = Node('=', [Node('mode'), Node(str(dtype))])
	
	propneg = negpn #Node('&', [clause, negpn])
	prop = pn # Node('&', [clause, pn])
	#prop = pn
	#propneg = negpn
	if DEBUG:
		print('prop: ', prop.to_prefix())
		print('negated prop: ', propneg.to_prefix())
	return (prop, propneg)

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

def getPropertyFromData(data, outputEqns):
	print(data)
	Y = data
	tm = data[0]
	data_noise = 0.2
	tm_0, tm_1  = tm*(1-data_noise), tm*(1+data_noise)
	prop = '((tm > {0}) & (tm < {1}) & ('.format(tm_0, tm_1)
	propn = '((tm > {0}) & (tm < {1}) & ! ('.format(tm_0, tm_1)

	i = 1
	for key in outputEqns.keys():
		if i < len(data) :
			if data[i] >= 0:
				k1, k2 = (data[i]*(1-data_noise), data[i]*(1+data_noise))
			else:				
				k1, k2 = (data[i]*(1+data_noise), data[i]*(1-data_noise))
			#eqn = key #
			eqn = outputEqns[key]
			pr = '(({0}) > {1}) & (({0}) < {2})'.format(eqn, k1, k2)

			prop += pr if i == 1 else ' & '+ pr 
			propn += pr if i == 1 else ' & '+ pr 
	prop += '));' 
	propn += '));' 
	return prop, propn

def getEstimatedParameters(model, params, d, k, datafile, outputEqns):
	sbox = getBox(params)
	#print(box, type(box))
	sat_box = []
	unsat_box = []
	undet_box = []
	global noise
	# noise.update({'radius':0.001})
	# # noise.update({'g': 0.001})	
	# noise.update({'K': 0.001})
	for par in params:
		noise.update({par: 0.001})
	tp = 0
	with open(datafile) as fp:
		fr = csv.reader(fp, delimiter=',')
		for row in fr:
			data = [float(row[i]) for i in range(len(row))]
			prop, propn = getPropertyFromData(data, outputEqns)
			print('------------------- Run for row ', tp+1, '-------------------------')
			print('----- data ----', data, '\n ', prop, '\n ', propn)
			(bt, bf, bu) = parameterCheck(model, sbox, d, k, (prop,propn))

			if tp == 0:
				sat_box = bt
				unsat_box = bf
				undet_box = bu
			else:
				sat_box = intersectionBoxMap(sat_box, bt)
				# sat_box.append(bt)
				for b in bf:
					unsat_box.append(b)
				for b in bu:
					undet_box.append(b)

			tp  += 1
			# sat_box.update({tp:tb})
			# unsat_box.update({tp:fb})
			# undet_box.update({tp:ub})
			if tp >= 1:
				break
	
	return (sat_box, unsat_box, undet_box)
	
if __name__ == "__main__": 	
   main(sys.argv[1:])

