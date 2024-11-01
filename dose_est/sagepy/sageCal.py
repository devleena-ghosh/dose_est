#!/usr/bin/env sage -python
from sage.all import *
from sage.symbolic.integration.integral import definite_integral
from sage.symbolic.integration.integral import indefinite_integral
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sagepy.optimizeOffset import *
from sagepy.drugFourier import *

var('d0, k_a, eps, t_gap, n, tm, pg, pi, off, delta, dd0')
d = eval('(d0/(1 - e**(-k_a * t_gap)))')
a = eval('(d * e**(-k_a*(t_gap-eps)))')
pg = eval('(2 *pi/t_gap)')
maxe = eval('0.5*(g + (g**2 + ep)**(0.5))')
pi = math.pi
assume(t_gap - eps > 0)

Tg = 0
Nterms = 0
Frac = 0
Din = 0
ka = 0

#from sage.symbolic.integration.integral import definite_integral

def main(argv):

	# tg = 10.05184
	# nterms = 20
	# din = 0.8 *3
	# k = 0.18384693747612374
	# p = 1
	# f = 8

	global Tg, Nterms, Frac, Din, ka

	tg = 10.05184 * 2
	nterms = 20
	frac = 0.8
	din = frac * 4
	k = 0.18384693747612374 * 5 /2
	p = 1
	f = 8

	Tg = tg
	Nterms = nterms
	Frac = frac
	Din = din
	ka = k

	(fn, f1, f2, a0) = getFourier(nterms)
	dfn = getFourierDerivative(fn)

	pfr = fn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi,delta = f, off = 0)
	defs = dfn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi,delta = f, off = 0)
	pf1 = f1(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)
	pf2 = f2(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)

	#offset = getOffset(pfr, defs, nterms, tg, p)
	(off, ffs, p, f, tm2) = optimizeSmootherfunction(nterms, tg, k, din)
	offset = getOptimizedOffset(nterms, tg, k, p, f, din)
	print('sageCal offset: ', offset)


	(fn1, f11, f21, a01) = getFourier(nterms)
	pfr1 = fn1(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = offset)

	plt1 = plot(pfr(tm), tm, 0, 20)
	i = tg
	plt2 = plot(pf1(tm), tm, 0, tg -p, color = 'red') + plot(pf2(tm), tm, tg -p, tg, color = 'red')
	while(i <= 20):
		r0 = i
		r1 = i+tg -p
		r2 = i + tg
		plot2 = plt2+ plot(pf1(tm), tm, i, r1, color = 'red') + plot(pf2(tm), tm, r1, r2, color = 'red')
		i = i+tg

	plt3 = plot(pfr1(tm), tm, 0, 20, color = 'green')
	plt = plt1+plt2+plt3
	plt.save(filename = 'images/fourier.png')
	#show(plt)
#	print('a0')
#	print(a01)

	(off, ffs, p, f, tm2) = optimizeSmootherfunction(nterms, tg, k, din)
	(fn1, f11, f21, a01) = getFourier(nterms)
	pfr1 = fn1(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = off)

	pf1 = f1(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)
	pf2 = f2(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)

	# i = tg
	plt5 = plot(pf1(tm), tm, 0, tg -p, color = 'black') + plot(pf2(tm), tm, tg -p, tg, color = 'black')
	# while(i <= 20):
	# 	r0 = i
	# 	r1 = i+tg -p
	# 	r2 = i + tg
	# 	plot5 = plt5+ plot(pf1(tm), tm, i, r1, color = 'red') + plot(pf2(tm), tm, r1, r2, color = 'red')
	# 	i = i+tg

	plt4 = plot(pfr1(tm), tm, 0, 20, color = 'brown')

	ffs1 = ffs(eps = 1, delta = 8)
	plt6 = plot(ffs1(tm), tm, 0, 20, color = 'yellow')
	ffs2 = ffs(eps = p, delta = f)
	plt7 = plot(ffs2(tm), tm, 0, 20, color = 'brown')

	plt = plt1+plt2+plt3+plt4+plt5
	plt.save(filename = 'images/fourier1.png')

	plt11 = plt6+plt7
	plt11.save(filename = 'images/fourier2.png')
	
	#show(plt)
	#print('a0')
	#print(a01)

	(rmin, rmax ) = (0.009765625, 0.1181640625)
	k_a = (5 * ( 0.693 / ( 7 * ( ( 2 * 3.1412 / 5 ) * 24 ) ) ) )
	t_g = ( ( 2 * 3.1412 / 5 ) * 24 )
	(rmin1, resMin) = getSSDActualValue(nterms, t_g, k_a, 0.8, rmin)
	(rmax1, resMax) = getSSDActualValue(nterms, t_g, k_a, 0.8, rmax)
		#res = (resMin, resMax)
	res = (rmin1, rmax1)
	print('result ', res)


def getOffsetOld(fn, dfn, nterms, tg, p):
	#print(fn)
	#print(dfn)
	
	res1 = bisection(dfn, tg-p, tg, 0.001)
#	print(res1)
	offset = fn(tm = res1)
	#print(offset)
	if(offset > 0):
		t1 = (res1 + 0.001)
	#	print(t1)
		res2 = bisection(dfn, t1, tg, 0.001)
		#print(res2)
		offset2 = fn(tm = res2)
		#print(offset2)
		if(offset2 < 0 ):
			#fr = fn - offset2;
			return -1 *offset2
		else:
			return 0
	else:
		return -1*offset

def getactualValue(nterms, tg, k, p, frac, f, din, offset = -1):
	# print('getactualValue ',nterms, tg, k, p, frac, din,f)

	ddash = eval('(dd0/(1 - e**(-k_a * t_gap)))')
	ssd = eval('(ddash * e**(-k_a * tm))')
	ssdV = ssd(t_gap = tg, k_a = k, pi=math.pi)

	(fn, f1, f2, a0) = getFourier(nterms)
	dfn = getFourierDerivative(fn)

	pfr = fn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)
	defs = dfn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)
	pf1 = f1(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)
	pf2 = f2(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)

	#offset = getOffset(pfr, defs, nterms, tg, p)
	#print('offset: ', offset)
	if offset == -1:
		offset = getOptimizedOffset(nterms, tg, k, p, f, din)
#	print('offset: ', offset)
	
	#offset = getOffsetFnc(pfr, tg, p, nterms)
	# print('sageCal offset: ', offset, tg, p, nterms)

	#(fn1, f11, f21, a01) = getFourier(nterms)
	pfr1 = fn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = offset)
	
	ifn1 = definite_integral(pfr1, tm, 0, tg)
	issd = definite_integral(ssdV, tm, 0, tg)
	eqn = eval('issd == ifn1')
	#print(issd, ifn1)
	#print(eqn)
	av = solve(eqn, dd0)
	# print(av[0].left(), "=", av[0].right())
	res = RR(av[0].right())
	# print(res, res/frac)
	return (res, res/frac)

	# av1 = find_root(eqn, dd0, 0, tg)
	# print(av1)

def getSSActual(fn, tg, k, p, frac, f, din, offset = -1):
	# print('getactualValue ',nterms, tg, k, p, frac, din,f)

	ddash = eval('(dd0/(1 - e**(-k_a * t_gap)))')
	ssd = eval('(ddash * e**(-k_a * tm))')
	ssdV = ssd(t_gap = tg, k_a = k, pi=math.pi)

	# (fn, f1, f2, a0) = getFourier(nterms)
	# dfn = getFourierDerivative(fn)

	pfr = fn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)
	# defs = dfn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)
	# pf1 = f1(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)
	# pf2 = f2(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)

	#offset = getOffset(pfr, defs, nterms, tg, p)
	#print('offset: ', offset)
	if offset == -1:
		offset = getOptimizedOffset(nterms, tg, k, p, f, din)
		pfr = fn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = offset)
#	print('offset: ', offset)
	
	#offset = getOffsetFnc(pfr, tg, p, nterms)
	# print('sageCal offset: ', offset, tg, p, nterms)

	#(fn1, f11, f21, a01) = getFourier(nterms)
	# pfr1 = fn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = offset)
	
	ifn1 = definite_integral(pfr, tm, 0, tg)
	issd = definite_integral(ssdV, tm, 0, tg)
	eqn = eval('issd == ifn1')
	# print(issd, ifn1)
	# print(eqn)
	av = solve(eqn, dd0)
	# print(av[0].left(), "=", av[0].right())
	res = RR(av[0].right())
	# print(res, res/frac)
	return (res, res/frac)

def getSSDActualValue(nterms, tg, k, frac, din):
	(der, init) = getSteadyStateDerivative()
	initf = init(t_gap = tg, k_a = k, pi=math.pi)
	eqn = eval('initf == din')
	print(eqn)
	avn = solve(eqn, d0)
	res = RR(avn[0].right())
	#print(res, res/frac)
	return (res, res/frac)

if __name__ == "__main__": 
	main(sys.argv[1:])




##############3-------------------------- discarded ---------------------#######
# pfr = four.subs(pg == (2 * pi /t_gap))
	# pfr = pfr.subs(t_gap == 10)
	# pfr = pfr.subs(k_a == 0.55)
	# pfr = pfr.subs(eps == pi/3)
	# pfr = pfr.subs(d0 == 0.4)

	# defs = der.subs(pg == (2 * pi /t_gap))
	# defs = defs.subs(t_gap == 10)
	# defs = defs.subs(k_a == 0.55)
	# defs = defs.subs(eps == pi/3)
	# defs = defs.subs(d0 == 0.4)


	# pf1 = f1.subs(pg == (2 * pi /t_gap))
	# pf1 = pf1.subs(t_gap == 10)
	# pf1 = pf1.subs(k_a == 0.55)
	# pf1 = pf1.subs(eps == pi/3)
	# pf1 = pf1.subs(d0 == 0.4)

	# pf2 = f2.subs(pg == (2 * pi /t_gap))
	# pf2 = pf2.subs(t_gap == 10)
	# pf2 = pf2.subs(k_a == 0.55)
	# pf2 = pf2.subs(eps == pi/3)
	# pf2 = pf2.subs(d0 == 0.4)

	# show(plot(pfr(tm), tm, 0, 40)+ plot(pf1(tm), tm, 0, 20 - 1, color='red'))

# fourdashn = ((pg * n) * ((bn * cos((pg * n)* tm)) - (an * sin((pg * n)* tm))))
# fdash = 0
# fourdashnSub = fourdashn.subs(n == 1)
# print(fourdashnSub)
# fdash = fdash + fourdashnSub
# for i in range(1,1):
# 	fourdashnSub = fourdashn.subs(n == 1)
# 	print(fourdashnSub)
# 	print('+')
# 	fdash = fdash + fourdashnSub

# #print('\n derivative of fourier upto 15 terms: \n')
# #print(fdash)

