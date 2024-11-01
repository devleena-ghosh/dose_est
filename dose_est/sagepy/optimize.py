#!/usr/bin/env sage -python
from sage.all import *
from sage.symbolic.integration.integral import definite_integral
from sage.symbolic.integration.integral import indefinite_integral

from sageCal import *

var('d0, k_a, eps, t_gap, n, tm, pg, pi, off, delta, dd0')
d = eval('(d0/(1 - e**(-k_a * t_gap)))')
a = eval('(d * e**(-k_a*(t_gap-eps)))')
pg = eval('(2 *pi/t_gap)')
pi = math.pi
assume(t_gap - eps > 0)
assume(t_gap>0)

Tg = 10.05184 * 2
Nterms = 20
Frac = 0.8
Din = Frac * 4
ka = 0.18384693747612374 * 5 /2
Res = 0
Nrip = 1.5
MinEps = 0.001
MaxEps = (Nrip * Tg)/ Nterms
MinFrac = 2.0
MaxFrac =  8.0 #(Din/0.001)
MaxSlope = (Din/MinFrac)/MinEps
MinSlope = (Din/MaxFrac)/(MaxEps) #0.001
MinT = Tg - ((Nrip+1)* Tg/Nterms)
MaxT = (Tg - Tg/(2*Nterms))
Tm = Tg
Numrp = Nrip
chk = Tg/Nterms
chkoff = 0

#from sage.symbolic.integration.integral import definite_integral

def main(argv):
	global Nterms, Tg, ka, Din, Frac, Tm, Numrp, chk, chkoff
	global MinT, MaxT, MinEps, MaxEps, MinSlope, MaxSlope, MinFrac, MaxFrac

	tg = Tg
	nterms = Nterms
	frac = Frac
	din = Din
	k = ka
	#p = 1
	#f = 8

	# MinEps = 0.001
	# MaxEps = (Nrip * Tg)/ Nterms
	# MinFrac = 1
	# MaxFrac =  4 #(Din/0.001)
	# MaxSlope = (Din/MinFrac)/MinEps
	# MinSlope = (Din/MaxFrac)/(MaxEps) #0.001
	#MinT = (Tg - ((Nrip+1)* Tg/Nterms))
	#MaxT = (Tg - Tg/(2*Nterms))

	fr1 = getFourierSimple(nterms)
	ffr1 = fr1(t_gap = tg, k_a = k, d0 = din, pi=math.pi)

	plt8 = plot(ffr1(tm), tm, 0, tg, color = 'blue')

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


	(off, ffs, p, f, tm2) = optimizeSmootherfunction(nterms, tg, k, din)
	(fn1, f11, f21, a01) = getFourier(nterms)

	#upffn = fn1(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)

	#(o, tm2) = getOffset(upffn, tg - p, chk) #(tg/nterms))
	of = off
	of = getOptimizedOffset(tg, k, p, f, din)
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


	plt11 = plt4+plt5+plt6+plt7 +plt8 #+plt10#+plt9
	plt11.save(filename = 'images/fourier2.png')

	et = Nrip*Tg/(Nterms)
	fet = math.floor(et)

	Numrp = math.floor(Nrip) #math.ceil(Nrip)
	print(Nrip, Numrp)
	if(Nrip - Numrp > 0.001):
		print(Nrip, Numrp, fet)
	else:
		fet = fet -1
		print(Nrip, Numrp, fet)

	print('chk: ',chk, fet)

	pfr3 = fn1(t_gap = tg, k_a = k, eps = et, d0 = din, pi=math.pi, delta = 8, off = 0)
	plt10 = plot(pfr3(tm), tm, 0, tg, color = 'brown')
	(off, tm2) = getOffset(pfr3, tg- fet, chk)
	plt12 = line([(tm2,0), (tm2,-off)], rgbcolor = (1, 0, 1)) 
	plt122 = line([(tg-et,0), (tg-et,din)], rgbcolor = (1, 0, 1))
	plt13 = plt10 + plt12 + plt122
	plt13.save(filename = 'images/fourier21.png')

	rmin = 0.01171875
	(rmin1, resMin) = getactualValue(nterms, tg, k, p, frac, f, rmin)
	print(rmin1, resMin)
	#show(plt)
	#print('a0')
	#print(a01)

def setOptimizeGolbal(nterms, tg, k, din):
	global Nterms, Tg, ka, Din
	global MinEps, MaxEps, MinFrac, MaxFrac, MinSlope, MaxSlope, MinT, MaxT
	global Numrp, chk, chkoff, Nrip

	Nterms = nterms
	Tg = tg
	ka = k
	Din = din
	
	Nrip = 1.5
	MinEps = 0.001
	MaxEps = (Nrip * Tg)/ Nterms
	MinFrac = 2.0
	MaxFrac =  8.0 #(Din/0.001)
	MaxSlope = (Din/MinFrac)/MinEps
	MinSlope = (Din/MaxFrac)/(MaxEps) #0.001
	MinT = Tg - ((Nrip+1)* Tg/Nterms)
	MaxT = (Tg - Tg/(2*Nterms))
	Numrp = Nrip
	chk = Tg/Nterms
	chkoff = 0

	Numrp = math.floor(Nrip) #math.ceil(Nrip)
	fet = math.floor(Nrip * Tg/Nterms) 
	print(Nrip, Numrp, fet)

	if(Nrip - Numrp > 0.001):
		chkoff = fet
		print(Nrip, Numrp, chkoff)
	else:
		chkoff = fet-1
		print(Nrip, Numrp, chkoff)

	print('chk: ',chk, chkoff)

def ffunc(x):
	global Nterms, Tg, ka, Din
	fr1 = getFourierSimple(Nterms)
	ffr1 = fr1(t_gap = Tg, k_a = ka, d0 = Din, pi=math.pi)
	return ffr1(tm = x[0])

def getFourierSimple(nterms):

	f1 = eval('(d * e**(-k_a * tm))')
	# print(f1)
	# print(f2)

	a01 = definite_integral(f1,tm,0,t_gap)
	a0 = eval('(2 *a01/t_gap)')
	#print(a0)

	f1c = eval('f1 * cos(pg * n * tm)')
	an1 = definite_integral(f1c,tm,0, t_gap)
	an = eval('(2 *an1 /t_gap)')

	#print(an)

	f1s = eval('f1 * sin(pg * n * tm)')
	bn1 = definite_integral(f1s,tm,0,t_gap)
	bn = eval('(2 *bn1/t_gap)')
	#print(bn)

	# print('a0 = ', a0)
	# print('an = ', an)
	# print('bn = ', bn)
	# print('\n')

	four = eval('a0/2')
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
	return (four)

def optimizeSmootherfunction(nterms, tg, k, din):
	#print(Nterms, Tg, ka, Din, Tm)
	setOptimizeGolbal(nterms, tg, k, din)

	#p1 = (1, 8, tg-1)
	p1 = (MinEps, MaxFrac, tg - MinEps-chk)
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
 # 	off = offset(eps = pp[0], delta = pp[1], tm= pp[2])
	
 #	print('off: ', off, pp[0], pp[1], Tm) #pp[2])

 	ff = getFSfuntion(nterms, tg, k, din)
 	off = objFunction((pp[0], pp[1], pp[2]))[0]

 	#ffs = -1 * ff
	print(off, pp[0], pp[1], pp[2])

	return(off, ff, pp[0], pp[1], pp[2])

    
def objFunction(x):
	global Nterms, Tg, ka, Din, MaxSlope, MinSlope, Tm

	fn = getFSfuntion(Nterms, Tg, ka, Din)
	#print('negate ',fn)
	#print('eps, delta', x[0], x[1])
	fnc = fn(eps = x[0], delta = x[1])
	#(obj, tm2) = getOffset(fnc, Tg - x[0], chk)#(Tg/Nterms))
	(obj, tm2) = getOffset(fnc, Tg - chkoff, chk)#(Tg/Nterms))

	#c01 = -1 * fnc
	#c02 = 1+fnc
	c11 = x[0]- MinEps # eps >= 0.001
	#c12 = Tg/2 - x[0] # eps <= tg/10
	c12 = MaxEps - x[0] # eps <= N ripples
	#c21 = x[1] - MinFrac # f >= min frac , 
	#c22 = MaxFrac - x[1] # f <= din/0.001
	#c31 = x[2] - Tg + x[0] # tm >= tg-eps
#	c31 = x[2] - MinT
#	c32 = MaxT - x[2] # tm <= tg
	c31 = x[2] - (Tg - x[0] - chk) # tm >= tg-eps
	c32 = (Tg - x[0]) - x[2] # tm <= tg
	c41 = x[1] - MinSlope*x[0]
	c42 = MaxSlope*x[0] - x[1]
	return (obj, c11, c12, c41, c42, c31, c32) #c21, c22,

def getFSfuntion(nterms, tg, k, din):
	(pfr, f1, f2, a0) = getFourier(nterms)
	fn = pfr(t_gap = tg, k_a = k, d0 = din, pi=math.pi, off = 0)
	#offfn = -1 * fn;
	return fn

def getOptimizedOffset(tg, k, p, f, dval):
	fn = getFSfuntion(Nterms, tg, k, dval)
	fnc = fn(eps = p, delta = f)
	#(off, tm2) = getOffset(fnc, Tg - p, chk)#(Tg/Nterms))
	(off, tm2) = getOffset(fnc, Tg - Numrp, chk)#(Tg/Nterms))
	return off

def getOffset(fn, tc1, p):
	#print(p, fn)
#	print('Check : ',tc1, p)

	dfn = diff(fn, tm) #getFourierDerivative(fn)
	res1 = bisection(dfn, tc1-p, tc1, 0.001)

	offset = fn(tm = res1)
	#print('foff: ', offset)
	if(offset > 0.0):
		t1 = (res1 + 0.001)
	#	print(t1)
		res2 = bisection(dfn, t1, tc1, 0.001)
		offset2 = fn(tm = res2)
		if(offset2 < 0.0 ):
		#	print('off: ', offset2, res2)
			return (-1 *offset2, res2)
		else:
		#	print('off: ', 0.0, -1)
			return (0.0, -1)
	else:
	#	print('off: ', offset, res1)
		return (-1*offset, res1)

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

def getOffsetFnc(fn, tg):
	#(off, tm2) = getOffset(fnc, Tg - p, chk)#(Tg/Nterms))
	(off, tm2) = getOffset(fn, tg - chkoff, chk)#(Tg/Nterms))
	return off

if __name__ == "__main__": 
	main(sys.argv[1:])




