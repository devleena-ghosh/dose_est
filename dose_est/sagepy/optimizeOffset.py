#!/usr/bin/env sage -python
from sage.all import *
from sage.symbolic.integration.integral import definite_integral
from sage.symbolic.integration.integral import indefinite_integral

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sagepy.sageCal import *
#from sagepy.optimizeOffset import *
from sagepy.drugFourier import *


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


var('d0, k_a, eps, t_gap, n, tm, pg, pi, off, delta, dd0')
d = eval('(d0/(1 - e**(-k_a * t_gap)))')
a = eval('(d * e**(-k_a*(t_gap-eps)))')
pg = eval('(2 *pi/t_gap)')
pi = math.pi
assume(t_gap - eps > 0)
assume(t_gap>0)

Tg = 24 #8
Nterms = 20
Frac = 1.0 #0.8
Din = 100 #0.05
ka = 1.3
Res = 0
Nrip = 1.0
MinEps = 0.001
MaxEps = (Nrip * Tg)/ Nterms
MinFrac = 2.0
MaxFrac =  4.0 #(Din/0.001)
MaxSlope = (Din/MinFrac)/MinEps
MinSlope = (Din/MaxFrac)/(MaxEps) #0.001
MinT = Tg - ((Nrip+1)* Tg/Nterms)
MaxT = (Tg - Tg/(2*Nterms))
Tm = Tg
Numrp = Nrip
ff = eval('1.0')
#chk = Tg/Nterms
#chkoff = 0
SIMPLE = True

#from sage.symbolic.integration.integral import definite_integral
import time
def main(argv):
	st = time.time()
	global Nterms, Tg, ka, Din, Frac, Tm, Numrp #, chk, chkoff
	global MinT, MaxT, MinEps, MaxEps, MinSlope, MaxSlope, MinFrac, MaxFrac

	tg = Tg
	nterms = Nterms
	frac = Frac
	din = Din
	k = ka
	p = 1.0/Nterms #(0.5 * 10/Nterms)#0.2
	print(p)
	f = 8

	# MinEps = 0.001
	# MaxEps = (Nrip * Tg)/ Nterms
	# MinFrac = 1
	# MaxFrac =  4 #(Din/0.001)
	# MaxSlope = (Din/MinFrac)/MinEps
	# MinSlope = (Din/MaxFrac)/(MaxEps) #0.001
	#MinT = (Tg - ((Nrip+1)* Tg/Nterms))
	#MaxT = (Tg - Tg/(2*Nterms))

	# res = bisection(ffr1, tg/3, tg, 0.001)
	# # plt9 = line([(res,0), (res,din)], rgbcolor = (1, 0, 1))
	# print('bisection : ', res)
	# global Res
	# Res = res
	# d1 = d(t_gap = tg, k_a = k, d0 = din, pi=math.pi)
	# r = -1 * (math.log(0.001/d1)) / k
	# print('time ', r)

	# (o1, tm1) = getOffset(ffr1, tg, 1)
	# print('offset: ', o1, tm1)
	# Tm = tm1

	# minf = minimize(lambda x: ffunc(x), [tg-1])
	# print('minimize: ', minf, ffr1(tm = minf[0]))
	
	et = p #Nrip*Tg/(Nterms)
	#fet1 = math.floor(et)

	# Numrp = math.floor(Nrip) #math.ceil(Nrip)
	# print(Nrip, Numrp)
	# if(Nrip - Numrp > 0.001):
	#   print(Nrip, Numrp, fet)
	# else:
	#   fet = fet -1
	#   print(Nrip, Numrp, fet)

	(fet, chk) = getRange(et, Nterms, Tg)
	print('chk: ',chk, fet, et * Nterms/Tg)

	
	fr1, _, _, _ = getFourierSimple(nterms)
	ffr1 = fr1(t_gap = tg, k_a = k, eps = et, d0 = din, pi=math.pi, delta = fet, off = 0)
	#off1 = getOptimizedOffset(nterms, tg, k, et, fet, din, ffr1, True)
	#print('simple offset', off1)
	(off1, tm2) = getOffset(ffr1, tg- fet, chk)
	print('Simple offset ', off1, tm2)

	ffr2 = fr1(t_gap = tg, k_a = k, eps = et, d0 = din, pi=math.pi, delta = fet, off = off1)
	plt8 = plot(ffr1(tm), tm, 0, tg, color = 'blue')
	plt9 = plot(ffr2(tm), tm, 0, tg, color = 'green') + line([(tm2,0), (tm2,-off1)], rgbcolor = (1, 0, 1)) 

	rmin = Din
	print('actual value: ', p, f, rmin, tg)
	(rmin1, resMin) = getSSDActualValue(nterms, tg, k, p, 1.0, f, rmin, SIMPLE = True)
	print(rmin1, resMin)

	ddash = eval('(dd0/(1 - e**(-k_a * t_gap)))')
	ssd = eval('(ddash * e**(-k_a * tm))')
	ssdV = ssd(t_gap = tg, k_a = k, pi=math.pi)

	plt9_1 = plot(ssdV(dd0 = rmin1), tm, 0, tg, color = 'red')
	plt11 = plt8+ plt9 +plt9_1 #+plt4+plt5+plt6+plt7#+plt8 #+plt10#+plt9
	filename = os.path.join('../sagepy', 'fourier2.png')
	plt11.save(filename)
	
	
	(fn1, f11, f21, a01) = getFourier(nterms)
	
	(off, ffs, p, f, tm2) = optimizeSmootherfunction(nterms, tg, k, din)
	et = p #Nrip*Tg/(Nterms)
	fet1 = math.floor(et)

	# Numrp = math.floor(Nrip) #math.ceil(Nrip)
	# print(Nrip, Numrp)
	# if(Nrip - Numrp > 0.001):
	#   print(Nrip, Numrp, fet)
	# else:
	#   fet = fet -1
	#   print(Nrip, Numrp, fet)

	(fet, chk) = getRange(et, Nterms, Tg)
	print('chk: ',chk, fet, fet1, et * Nterms/Tg)

	#upffn = fn1(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)

	#(o, tm2) = getOffset(upffn, tg - p, chk) #(tg/nterms))
	of = off
	print('Old offset: ', off, tm2)
	of = getOptimizedOffset(nterms, tg, k, p, f, din)
	print('New offset: ', of, tm2)
	print('Optimized parameter : ', p, f, tm2)


	ffs1 = -1 * ffs(eps = p, delta = f)
	plt6 = plot(ffs1(tm), tm, 0, tg, color = 'cyan')

	pfr2 = fn1(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)
	plt7 = plot(pfr2(tm), tm, 0, tg, color = 'red')

	pf1 = f11(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)
	pf2 = f21(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)
	plt5 = plot(pf1(tm), tm, 0, tg -p, color = 'black') + plot(pf2(tm), tm, tg -p, tg, color = 'black')

	pfr1 = fn1(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = of)
	plt4 = plot(pfr1(tm), tm, 0, tg, color = 'green')
	
	pfr3 = fn1(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)
	plt10 = plot(pfr3(tm), tm, 0, tg, color = 'brown')
	(off, tm2) = getOffset(pfr3, tg- fet, chk)
	plt12 = line([(tm2,0), (tm2,-off)], rgbcolor = (1, 0, 1)) 
	plt122 = line([(tg-et,0), (tg-et,din)], rgbcolor = (1, 0, 1))


	rmin = Din
	print('actual value: ', p, f, rmin, tg)
	(rmin2, resMin2) = getactualValue(nterms, tg, k, p, 1.0, f, rmin, SIMPLE = False)
	print(rmin2, resMin2)

	plt9_2 = plot(ssdV(dd0 = rmin2), tm, 0, tg, color = 'red')

	plt13 = plt10 + plt12 + plt122 +plt4+plt5+plt9_2 #+plt6 + plt7 + 
	filename = os.path.join('../sagepy', 'fourier21.png')
	plt13.save(filename)
	
	#show(plt13)
	#print('a0')
	#print(a01)
	nterms = 20
	q0, tspan = getSimulated(2, nterms, tg, k, din, p, f, 0, True)
	st1 = time.time()
	q1, tspan = getSimulated(2, nterms, tg, k, din, p, f, off1, True)
	fig = plt.figure()
	plt.plot(tspan, q1, 'b')
	#plt.plot(tspan, q0, 'g')
	horiz_line_data = np.array([0.0 for i in range(len(tspan))])
	horiz_line_data1 = np.array([-off1 for i in range(len(tspan))])
	plt.plot(tspan, horiz_line_data, 'r--') 
	plt.plot(tspan, horiz_line_data1, 'r--') 
	pp = PdfPages('fourier_sim.pdf')
	pp.savefig(fig)
	pp.close()
	print('################')
	print(rmin1, resMin)
	print(rmin2, resMin2)
	print(p, f)
	'''pp = PdfPages('fourier.pdf')
	for fig in figs:
		pp.savefig(fig)

	pp.close()'''
	print('Time taken: ', time.time() - st, time.time() - st1)

def setOptimizeGolbal(nterms, tg, k, din, ff= None, simple=False):
	global Nterms, Tg, ka, Din
	global MinEps, MaxEps, MinFrac, MaxFrac, MinSlope, MaxSlope, MinT, MaxT
	global Numrp, Nrip #chk, chkoff,
	global SIMPLE
	SIMPLE = simple
	Nterms = nterms
	Tg = tg
	ka = k
	Din = din
	
	Nrip = 1.0
	MinEps = 0.001
	MaxEps = (Nrip * Tg)/ Nterms
	MinFrac = 2.0
	MaxFrac =  4.0 #(Din/0.001)
	MaxSlope = (Din/MinFrac)/MinEps
	MinSlope = (Din/MaxFrac)/(MaxEps) #0.001
	MinT = Tg - ((Nrip+1)* Tg/Nterms)
	MaxT = (Tg - Tg/(2*Nterms))
	Numrp = Nrip
	
	print('setOptimizeGolbal', Nterms, Tg, ka, Din, MinEps, MaxEps, MinFrac, MaxFrac, MinSlope, MaxSlope, MinT, MaxT)
	
	#if not ff:
	#	ff = getFSfuntion(nterms, tg, k, din, simple)
	print('setOptimizeglobal', Nterms, Tg, ka, Din)#ff)
	#chk = Tg/Nterms
	# chkoff = 0

	# Numrp = math.floor(Nrip) #math.ceil(Nrip)
	# fet = math.floor(Nrip * Tg/Nterms) 
	# print('Nrip, Numrp, fet', Nrip, Numrp, fet)

	# if(Nrip - Numrp > 0.001):
	#   chkoff = fet
	#   print('ge ', Nrip, Numrp, chkoff)
	# else:
	#   chkoff = fet-1
	#   print('le ', Nrip, Numrp, chkoff)

	# print('chk, chkoff: ',chk, chkoff)
	

def ffunc(x):
	global Nterms, Tg, ka, Din
	fr1 = getFourierSimple(Nterms)
	ffr1 = fr1(t_gap = Tg, k_a = ka, d0 = Din, pi=math.pi)
	return ffr1(tm = x[0])

def optimizeSmootherfunction(nterms, tg, k, din, fn= None, simple= False):
	global ff
	if not fn:		
		ff = getFSfuntion(nterms, tg, k, din, simple)
	else:
		ff = fn
	print('optimizeSmootherfunction', ff)
	setOptimizeGolbal(nterms, tg, k, din)
	sys.stdout.flush()
	#p1 = (1, 8, tg-1)
	p1 = (MinEps, MaxFrac, tg - MinEps - tg/nterms)
	func_cached = CachedFunction(objFunction)
	func_wrap = lambda x: func_cached(tuple(x))
	pp = minimize_constrained(lambda x: objFunction(x)[0], 
							 [lambda x: objFunction(x)[1], lambda x: objFunction(x)[2],
							 lambda x: objFunction(x)[3], lambda x: objFunction(x)[4],
							 lambda x: objFunction(x)[5], lambda x: objFunction(x)[6]],
						   #  lambda x: objFunction(x)[7], lambda x: objFunction(x)[8]],
						   #  lambda x: objFunction(x)[9]], #lambda x: objFunction(x)[10]],
							 p1)
	#print('pp: ', pp)

	# offset = getFSfuntion(nterms, tg, k, din)
 #  off = offset(eps = pp[0], delta = pp[1], tm= pp[2])
	
 #  print('off: ', off, pp[0], pp[1], Tm) #pp[2])

	off = objFunction((pp[0], pp[1], pp[2]))[0]

	#ffs = -1 * ff
	print('optimizeSmootherfunction:', off, pp[0], pp[1], pp[2])

	return(off, ff, pp[0], pp[1], pp[2])

	
def objFunction(x):
	global  ff, Nterms, Tg, ka, Din, MaxSlope, MinSlope, MinEps, MaxEps #, chk, chkoff

	fn = ff 
	#getFSfuntion(Nterms, Tg, ka, Din, SIMPLE)
	#print('negate ',fn)
	#print('eps, delta', x[0], x[1])
	fnc = fn(eps = x[0], delta = x[1])
	(cof, chk) = getRange(x[0], Nterms, Tg)
	#(obj, tm2) = getOffset(fnc, Tg - x[0], chk)#(Tg/Nterms))
	(obj, tm2) = getOffset(fnc, Tg - cof, chk)#(Tg/Nterms))
	#(obj, tm2) = getOffsetF(fnc, Tg - cof, chk, Tg)#(Tg/Nterms))

	#c01 = -1 * fnc
	#c02 = 1+fnc
	c11 = x[0]- MinEps # eps >= 0.001
	#c12 = Tg/2 - x[0] # eps <= tg/10
	c12 = MaxEps - x[0] # eps <= N ripples
	#c21 = x[1] - MinFrac # f >= min frac , 
	#c22 = MaxFrac - x[1] # f <= din/0.001
	#c31 = x[2] - Tg + x[0] # tm >= tg-eps
#   c31 = x[2] - MinT
#   c32 = MaxT - x[2] # tm <= tg
	c31 = x[2] - (Tg - cof - chk) # tm >= tg-eps
	c32 = (Tg - cof) - x[2] # tm <= tg
	c41 = x[1] - MinSlope*x[0]
	c42 = MaxSlope*x[0] - x[1]

	#print('objfnc ', obj, tm2, cof, chk)
	return (obj, c11, c12, c41, c42, c31, c32) #c21, c22,

def getRange(et, nterms, tg):
	fet = math.floor(et)
	
	nrp = et * nterms/tg
	#nrp1 = round(nrp, 0)

	nmrp1 = math.floor(nrp) #math.ceil(Nrip)
	nmrp2 = math.ceil(nrp)
	#print('getrange ', nrp, nmrp)
	cof = 0.0
	ep = tg/nterms
	if(nrp - nmrp1 > 0.5):
		nmrp1 = nmrp1 + 1;
	# else:
	#   nmrp2 = nmrp2 - 0.5;
	# if(nrp - nmrp < 0.001):
	#   cof = nmrp* tg/nterms
	# # print('ge ',nrp, nmrp, cof)
	# else:
	#   if(nmrp >= 1):
	#       cof = (nmrp -1)* tg/nterms
	#   #cof = fet
	# # print('le ',nrp, nmrp, cof)
	
	# if(nmrp2 - nmrp1 < 0.001):
	#   nmrp2 = nmrp2 + 1
	#ep = (nmrp2* tg/nterms - et)
	ep = tg/nterms 
	cof = nmrp1* tg/nterms
	#print('range: ', nrp, cof, ep, nmrp1, nmrp2)
	return (cof, ep)

def getOptimizedOffset(nterms, tg, k, p, f, dval, fn=None, simple=False):
	#global Nterms #, chkoff, chk
	if not fn:
		fn = getFSfuntion(nterms, tg, k, dval, simple)
	fnc = fn(eps = p, delta = f, t_gap = tg, k_a = k, d0 = dval, pi=math.pi)
	#(off, tm2) = getOffset(fnc, Tg - p, chk)#(Tg/Nterms))
	(chkoff, chk) = getRange(p, nterms, tg)
	#(off, tm2) = getOffset(fnc, tg - chkoff, chk) #(Tg/Nterms))# getOffsetF(fnc, tg)
	#print(off, tm2)
	(off, tm2) = getOffsetF(fnc, tg-chkoff, chk, tg) #(Tg/Nterms))# 
	# print(off, tm2)
	return off

def getOffsetFnc(fn, tg, p, nterms):
	#global chk, chkoff
	(chkoff, chk) = getRange(p, nterms, tg)
	print('range: ',chkoff, chk)
	#(off, tm2) = getOffset(fnc, Tg - p, chk)#(Tg/Nterms))
	#(off, tm2) = getOffset(fn, tg - chkoff, chk)#(Tg/Nterms)) #getOffsetF(fn, tg)
	(off, tm2) = getOffsetF(fnc, tg-chkoff, chk, tg)
	return off

if __name__ == "__main__": 
	main(sys.argv[1:])




