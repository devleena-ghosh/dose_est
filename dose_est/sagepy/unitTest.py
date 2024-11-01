from __future__ import print_function
import os
import subprocess
import re
import sys, getopt
from sageCal import *
from optimizeOffset import *
from drugFourier import *

k_el = 0.693 / (7 * 24)
k =( 5 * k_el )
N = 20
tg = (7*24)
f = 0.8

din = 1
#var('drug_init')

def main(argv):	
	global N, tg, f, k, k_el
	
	(fn, f1, f2, a0) = getFourier(N)
	#dfn = getFourierDerivative(fn)
	pfr = fn(t_gap = tg, k_a = ka, pi=math.pi, off = 0)
	defs = getFourierDerivative(pfr)
	#defs2 = dfn(t_gap = tg, k_a = ka, eps = p, pi=math.pi, delta= f, off = 0)
	#inita = a0(t_gap = tg, k_a = ka, eps = p, pi=math.pi, delta= f)
	initf = pfr(tm = 0)
	#print("four ", pfr, "\n derivative: ",toString(defs),"\n der: ", toString(defs2), "\n init: ", toString(inita) , "\n initf: ", toString(initf))
	print("four ", pfr, "\n derivative: ",toStr(defs),"\n initf: ", toStr(initf))
	
	
	(der, init) = getSteadyStateDerivative()
	defs = der(pi=math.pi)
	initf = init(pi=math.pi)
	print("\n derivative: ",toStr(defs),"\n initf: ", toStr(initf))
		
	
	dval = 0.8 * 1
	print(dval, f, din)		

	(off, ffs, p1, f1, tm) = optimizeSmootherfunction(N, tg, ka, dval)
	p = p1
	f = f1
	print('OffsetCal: ', off, p, f, din)
	pfr1 = fn(d0 = dval, eps = p, delta = f, t_gap = tg, k_a = ka, pi=math.pi)
	defs = getFourierDerivative(pfr1)
	initf = pfr1(tm = 0)
	offset = getOptimizedOffset(N, tg, ka, p, f, dval)
	print("four ", pfr, "\n derivative: ",toStr(defs),"\n initf: ", toStr(initf))
	print('offset: ', float(offset), din)
	


def toStr(expr):
	exprString = str(expr).replace('e^', 'exp')#.replace('t', 'tm')
	return exprString

if __name__ == "__main__": 
   main(sys.argv[1:])
