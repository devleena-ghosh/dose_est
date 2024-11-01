#!/usr/bin/env sage -python
from sage.all import *
from sage.symbolic.integration.integral import definite_integral
from sage.symbolic.integration.integral import indefinite_integral
import numpy as np
from scipy.integrate import odeint
from numpy  import sin, cos, exp, sqrt
#from numba import jit

var('d0, k_a, eps, t_gap, n, tm, pg, pi, off, delta, dd0, g, ep')
d = eval('(d0/(1 - e**(-k_a * t_gap)))')
a = eval('(d * e**(-k_a*(t_gap-eps)))')
pg = eval('(2 *pi/t_gap)')
maxe = eval('0.5*(g + (g**2 + ep)**(0.5))')
maxlse = eval('(log(e**(ep*g) + 1)/ep)')
pi = math.pi
assume(t_gap - eps > 0)

def getFourier(nterms):
	#print('getFourier', nterms)
	s = eval('(tm-t_gap+eps)')
	x = eval('((eps**2)/(4 *(d/delta)))')

	f1 = eval('(d * e**(-k_a * tm))')
	f2 = eval('(a + (s**2)/(4 * x))')
	# print(f1)
	# print(f2)

	a01 = definite_integral(f1,tm,0,(t_gap-eps))
	a02 = definite_integral(f2,tm,(t_gap-eps), t_gap)
	a0 = eval('(2 *(a01+a02)/t_gap)')
	#print(a0)

	f1c = eval('f1 * cos(pg * n * tm)')
	f2c = eval('f2 * cos(pg * n * tm)')
	an1 = definite_integral(f1c,tm,0,(t_gap- eps))
	an2 = definite_integral(f2c,tm,(t_gap- eps), t_gap)
	an = eval('(2 *(an1 + an2)/t_gap)')

	#print(an)

	f1s = eval('f1 * sin(pg * n * tm)')
	f2s = eval('f2 * sin(pg * n * tm)')
	bn1 = definite_integral(f1s,tm,0,(t_gap- eps))
	bn2 = definite_integral(f2s,tm,(t_gap- eps), t_gap)
	bn = eval('(2 *(bn1 + bn2)/t_gap)')
	#print(bn)

	# print('a0 = ', a0)
	# print('an = ', an)
	# print('bn = ', bn)
	# print('\n')

	four = eval('a0/2 + off')
	anc = eval('an * cos(pg * n * tm)')
	bns = eval('bn * sin(pg * n * tm)')

	for i in range(1,int(nterms)):
		ansub = anc(n = i)
		bnsub = bns(n = i)
		# print(ansub)
		# print('+')
		# print(bnsub)
		# print('+')
		four = eval('four + ansub + bnsub')

	#print('\n fourier upto 10 terms: \n')
	#print(four)
	four_max = maxe(g=four, ep=0.00001)
	return (four_max, f1, f2, a0/2)

def getFourierSimple(nterms, ep = 200, flag = 0):
	# print('getFourier', nterms)
	f11 = eval('(d * e**(-k_a * tm))')
	# if flag == 0:
	f1 = eval('(d * e**(-k_a * tm))**0.5')
	f2 = eval('1.0')
	# print(f1)
	# print(f2)

	a01 = definite_integral(f1,tm,0,t_gap)
	a0 = eval('(2 *a01/t_gap)')
	#print(a0)

	f1c = eval('f1 * cos(pg * n * tm)')
	an1 = definite_integral(f1c,tm,0, t_gap)
	an = eval('(2 *an1/t_gap)')

	#print(an)

	f1s = eval('f1 * sin(pg * n * tm)')
	bn1 = definite_integral(f1s,tm,0,t_gap)
	bn = eval('(2 *bn1/t_gap)')
	#print(bn)

	# print('a0 = ', a0)
	# print('an = ', an)
	# print('bn = ', bn)
	# print('\n')

	four = eval('a0/2+off')
	anc = eval('an * cos(pg * n * tm)')
	bns = eval('bn * sin(pg * n * tm)')

	for i in range(1,int(nterms)):
		ansub = anc(n = i)
		bnsub = bns(n = i)
		# print(ansub)
		# print('+')
		# print(bnsub)
		# print('+')
		four = eval('four + ansub + bnsub')

	#print('\n fourier upto 10 terms: \n')
	#print(four)
	#four_max = maxe(g=four, ep=d0*ep)
	# if flag == 1:
	# 	# four_max = maxlse(g=four, ep=ep) #*(0.5/(d0+0.00001)))
	# 	four_max = maxlse(g=four, ep=ep*1/(d0+0.00001))
	# elif flag == 2:
	# 	four_max = maxe(g=four, ep=0.00001)
	# else:
	four_max = eval('(four)**2')
	# deriv = diff(four_max, tm)
	#print(four_max)
	return (four_max, four_max, f11, a0/2)

def createFile(nterms,  ep=200, flag=0):
	(four_max, f1, f2, a0) = getFourierSimple(nterms, ep, flag)
	var('tg', 'ka', 'd1')
	sk = 'from scipy.integrate import odeint'+'\n'+\
			'import numpy as np'+'\n'+\
			'from numpy import sin, exp, cos'+'\n'+\
			'import math'+'\n'+\
			'from math import e, pi, sqrt'+'\n'
	sk += 'def dose_fs(tm, kargs):'+'\n'+\
			'\t tg, ka, d0, AUC = kargs'+'\n'+\
			'\t dose =  {0}'+'\n'+\
			'\t return dose'+'\n'+\
		'def ffs(y, t, kargs):'+'\n'+\
			'\t tg, ka, d0, AUC = kargs'+'\n'+\
			'\t tm = y[0]'+'\n'+\
			'\t x0 = y[1]'+'\n'+\
			'\t z = y[2]'+'\n'+\
			'\t y1 = y[3]'+'\n'+\
			'\t dose = dose_fs(tm, kargs)'+'\n'+\
			'\t qdot = np.zeros(4)'+'\n'+\
			'\t qdot[0] = 1.0'+'\n'+\
			'\t qdot[1] = dose'+'\n'+\
			'\t qdot[2] = -ka*z'+'\n'+\
			'\t qdot[3] = z'+'\n'+\
			'\t return qdot'+'\n'+\
		'def getX(kargs):'+'\n'+\
			'\t# print(kargs)'+'\n'+\
			'\t tg, ka, d0, AUC = kargs'+'\n'+\
			'\t tt = np.linspace(0, int(tg), int(tg*3600))'+'\n'+\
			'\t dd = (d0/(1 - exp(-ka * tg)))'+'\n'+\
			'\t soln = odeint(ffs, [0, 0, dd, 0], tt, args=(kargs,))'+'\n'+\
			'\t issdv = soln[:,1][-1]'+'\n'+\
			'\t ssd_auc_div = soln[:,3][-1]'+'\n'+\
			'\t# print(dd, tt, ssd_auc_div)'+'\n'+\
			'\t if AUC == 1:'+'\n'+\
			'\t\t x = ssd_auc_div/issdv'+'\n'+\
			'\t else:'+'\n'+\
			'\t\t x = (ssd_auc_div - issdv)/soln[:,0][-1]'+'\n'+\
			'\t return x'+'\n'+\
		'def getActual(kargs):'+'\n'+\
			'\t# print(kargs)'+'\n'+\
			'\t tg, ka, d0 = kargs'+'\n'+\
			'\t tt = np.linspace(0, int(tg), int(tg*3600))'+'\n'+\
			'\t dd = (1/(1 - exp(-ka * tg)))'+'\n'+\
			'\t soln = odeint(ffs, [0, 0, dd, 0], tt, args=(kargs,))'+'\n'+\
			'\t issdv = soln[:,1][-1]'+'\n'+\
			'\t ssd_auc_div = soln[:,3][-1]'+'\n'+\
			'\t print(dd, tt, ssd_auc_div)'+'\n'+\
			'\t d11 = issdv/ssd_auc_div'+'\n'+\
			'\t return d11\n'
	f = open("getFS.py", "w")
	f.write(sk.format(str(four_max(t_gap = tg, k_a = ka, pi=math.pi, off = 0))).replace('e^', 'exp').replace('^', '**'))
	f.close()
	# import getFS as fs
	# x = fs.getX((tg, k, din))
	return (four_max, f1, f2, a0)
	
# def getSteadyStateDerivative():
#   ssd = eval('(d * e**(-k_a * tm))')

#   ssf = diff(ssd, tm)
#   initssf = ssd(tm = 0)

#   return (ssf, initssf)

def getSteadyStateDerivative():
	d = eval('(d0/(1 - e**(-k_a * t_gap)))')
	pi = math.pi
	assume(t_gap>0)

	ssd = eval('(d * e**(-k_a * tm))')

	ssf = diff(ssd, tm)
	initssf = ssd(tm = 0)
	# print(ssf, initssf)
	return (ssf, initssf)

def getFourierDerivative(fn):
	der = diff(fn, tm)
	#print('\n derivative of fourier: \n')
	#print(der)
	return der

def getIntegral(fn):
	it = indefinite_integral(fn,tm)
#   print(it)
	return it


def getSimulated(t_d, nterms, tg, k, din, p, f, offset, simple=True):
	def toStr(expr):
		exprString = str(expr).replace('e^', 'exp').replace('^', '**')
		return exprString
	
	if simple:
		(pfr, f1, f2, a0) = getFourierSimple(nterms)
	else:
		(pfr, f1, f2, a0) = getFourier(nterms)
	fn = pfr(t_gap = tg, k_a = k, d0 = din, pi=math.pi, off = 0.0, eps = p, delta = f)
	
	dfn = getFourierDerivative(fn)
	initf = fn(tm=0.0) 
	tspan = np.linspace(0, t_d*24, t_d*24*5)
	# print('################')
	# print(dfn)
	# print('####################')
	#@jit(nopython=True)
	def ode_f(y, t):
		q0 = y[0]
		q1 = y[1]
		qdot = np.zeros(2)
		qdot[0] = 1.0
		qdot[1] = dfn(tm = q0)
		return qdot
		
	y0 = [0.0, float(initf)+offset]
	soln = odeint(ode_f, y0, tspan)
	
	q0 = soln[:,0]
	q1 = soln[:,1]
	return q1, q0
	

def getFSfuntion(nterms, tg, k, din, simple=False):
	if simple:
		(pfr, f1, f2, a0) = getFourierSimple(nterms)
	else:
		(pfr, f1, f2, a0) = getFourier(nterms)
	fn = pfr(t_gap = tg, k_a = k, d0 = din, pi=math.pi, off = 0)	
	return fn

def getOffset(fn, tc1, p):
	#print(p, fn)
#   print('Check : ',tc1, p)

	dfn = getFourierDerivative(fn)
	res1 = bisection(dfn, tc1-p, tc1, 0.001)

	offset = fn(tm = res1)
	#print('foff: ', offset)
	if(offset > 0.0):
		t1 = (res1 + 0.001)
	#   print(t1)
		res2 = bisection(dfn, t1, tc1, 0.001)
		offset2 = fn(tm = res2)
		if(offset2 < 0.0 ):
			#print('off: ', offset2, res2)
			return (-1 *offset2, res2)
		else:
			#print('off: ', 0.0, -1)
			return (0.0, -1)
	else:
		#print('off: ', offset, res1)
		t1 = (res1 + 0.001)
		#print(t1)
		res2 = bisection(dfn, t1, tc1, 0.001)
		offset2 = fn(tm = res2)
		if(offset2 < 0.0 and offset2 < offset ):
			#print('off: ', offset2, res2)
			return (-1 *offset2, res2)
		elif(offset2 > offset):         
			return (-1 *offset, res1)
			#print('off: ', 0.0, -1)
		else:
			return (0.0, -1)

def getOffsetF(fn, tc1, p, tg):
	#print(p, fn)
	(off, res) = getOffset(fn, tc1, p)
	if(res > 0.0):
		t = res
	else:
		t = tc1-p
	minres = t
	#print('Check : ',tc1, p)
	minOff = fn(tm=t)
	if(minOff > 0.0):
		minOff = 0.0
		minres = -1
	t = tc1-p
	while(t <= tg):
		off = fn(tm=t)
		if(off > 0.0):
			off = 0.0
		if(off < minOff):
			minOff = off
			minres = t
		t = t+ 0.001

	if(minOff < 0.0):
		return(-1*minOff, minres)
	else:
		return(0.0, -1)
	

# def getOffsetF(fn, tg):
#   #print(p, fn)
# # print('Check : ',tc1, p)
#   I1 = (0,tg)
#   #pp = minimize_constrained(fn, [I1], (tg))
#   pp = minimize(fn, [tg])
#   res = pp[0]
#   offset = fn(tm = res)
#   print('offsetF: ', offset, res)
#   if(offset < 0.0):
#       return( -1*offset, res)
#   else:
#       return( 0.0, -1)

def bisection(fn, a,b,tol):
	c = (a+b)/2.0
	while (b-a)/2.0 > tol:
		if fn(tm=c) == 0:
			return c
		elif fn(tm=a)*fn(tm=c) < 0:
			b = c
		else :
			a = c
		c = (a+b)/2.0
		
	return c
