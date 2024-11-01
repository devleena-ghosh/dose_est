#!/usr/bin/env sage -python
from sage.all import *
from sage.symbolic.integration.integral import definite_integral
from sage.symbolic.integration.integral import indefinite_integral
from optimizeOffset import *
from sageCal import *
import decimal

var('d0, k_a, eps, t_gap, n, tm, pg, pi, off, delta, dd0')
# d = eval('(d0/(1 - e**(-k_a * t_gap)))')
# a = eval('(d * e**(-k_a*(t_gap-eps)))')
# pg = eval('(2 *pi/t_gap)')
# pi = math.pi
# assume(t_gap>0)
# assume(t_gap - eps > 0)
#N = 20
#from sage.symbolic.integration.integral import definite_integral
# n = 0
# f1 = d0
# f = d0
# tp = 0
folder = 'images'
#tg = 10.05184 * 2
N = 20
frac = 0.93
din = frac * 32.0
#k = 0.18384693747612374 * 5 /2
#k = ( 2 * ( 0.693 / ( ( ( 2 * 3.1412 / 5 ) * 24 ) / 12 ) ) )
#tg = ( 2 * ( ( 2 * 3.1412 / 5 ) * 24 ) / 3 )
#k = ( 2 * ( 0.693 / ( ( ( 2 * 3.1412 / 5 ) * 24 ) / 12 ) ) )
#tg = ( 2 * ( ( 2 * 3.1412 / 5 ) * 24 ) / 3 )
#--- hyper v2 ---#
k = ( 2 * ( 0.693 / ( ( ( 2 * 3.1412 / 5 ) * 24 ) / 4 ) ) )
tg = ( 2 * ( ( 2 * 3.1412 / 5 ) * 24 ) / 3 )
#( ( ( 2 * 3.1412 / 5 ) * 24 ) / 3 )
p = 1
f = 8
#optOff = 0.0

def main(argv):

	global N, tg, k, din, p, f

	(optOff1, ffs, p1, f1, tm1) = optimizeSmootherfunction(N, tg, k, din)
	p = p1
	f = f1

	print('OFF1: ', optOff1, p, f)

	optOff = getOptimizedOffset(N, tg, k, p, f, din)

	print('OFF2: ', float(optOff), p, f)

	plotFourier1(tg,k,p,din,f, N, optOff)

	off = optOff
	print('OFF: ', optOff, off, p, f)

	# t_half = 8
	# k_a = 0.693 / t_half
	# t_gap =1.25 * 2;

	k_a = k #0.18384693747612374 * 5 /2
	t_gap = tg # 10.05184 * 2

	plotFunction(t_gap, k_a, din, N, optOff)

	#din = 0.01171875
	plotactualValue(N, tg, k, p, frac, f, optOff, din)

def plotFunction(tg,k,din, nterms, optOff):
	print('plotFunction', tg,k,din, nterms, optOff)
	d = eval('(d0/(1 - e**(-k_a * t_gap)))')
	pg = eval('(2 *pi/t_gap)')
	pi = math.pi
	assume(t_gap>0)

	fn = eval('(d0 * e**(-k_a*tm))')
	func = fn(t_gap = tg, k_a = k, d0 = din)

	(four, ff1, a0) = getFourierFfn(nterms)
	ffn = four(t_gap = tg, k_a = k, d0 = din, pi=math.pi)
	print(ffn(tm=0),ff1(t_gap = tg, k_a = k, d0 = din, pi=math.pi), a0(t_gap = tg, k_a = k, d0 = din, pi=math.pi))

	i = tg
	plot1 =  plot(func(tm), tm, 0, tg, color = 'blue', label = 'time')
	
	f1 = eval('(d * e**(-k_a*tm))')
	ssd = f1(t_gap = tg, k_a = k, d0 = din)
	plot3 =  plot(ssd(tm), tm, 0, tg, color = 'blue', label = 'time')

	# plot2 = plt2
	r0 = i
	r1 = tg
	lv = func(tm = tg)
	nv = lv + din

	while(i <= 40):
		plot1 = plot1 + line([(r1,lv), (r1,nv)], rgbcolor = (0, 0, 1))
		r0 = i
		r1 = i + tg
		#print(i, r0, r1)

		func = fn(t_gap = tg, k_a = k, d0 = nv)
		plot1 = plot1+ plot(func(tm-i), tm, r0, r1, color = 'blue')

		lv = func(tm = r1-i)
		nv = lv + din

		#print('last value : ', lv, 'din : ', din, 'next value : ', nv)
		i = i+tg 

	i = tg
	r0 = i
	r1 = tg

	lv1 = ssd(tm = tg)
	nv1 = ssd(tm = 0)
	while(i <= 10):
		plot3 = plot3 + line([(r1,lv1), (r1,nv1)], rgbcolor = (0, 0, 1))
		r0 = i
		r1 = i + tg
		#print(i, r0, r1)

		plot3 = plot3 + plot(ssd(tm-i), tm, r0, r1, color = 'blue')

		lv1 = ssd(tm = r1-i)
		nv1 = ssd(tm = r0-i)
	#	print('last value : ', lv1, 'din : ', din, 'next value : ', nv1)
		i = i+tg 

	plot1.axes_labels(['time', 'drug concentration'])
	plot1.legend(True)

	plot2 =  plot(ffn(tm), tm, 0, 10, color = 'blue', label = 'time')

	plot1.save(filename = 'images/actual_dc.png')
	plot2.save(filename = 'images/ff10.png')
	plot3.save(filename = 'images/ssd_dc.png')



def getFourierFfn(nterms):
	d = eval('(d0/(1 - e**(-k_a * t_gap)))')
	pg = eval('(2 *pi/t_gap)')
	pi = math.pi
	assume(t_gap>0)

	f1 = eval('(d * e**(-k_a*tm))')

	a0 = definite_integral(eval('2*f1/t_gap'),tm,0,t_gap)

	f1c = eval('2* f1 * cos(pg * n * tm)/t_gap')
	an = definite_integral(f1c,tm,0,t_gap)

	#print(an)

	f1s = eval('2* f1 * sin(pg * n * tm)/t_gap')
	bn = definite_integral(f1s,tm,0,t_gap)
	#print(bn)

	# print('a0 = ', a0)
	# print('an = ', an)
	# print('bn = ', bn)
	# print('\n')

	four = eval('a0/2')
	anc = eval('an * cos(pg * n * tm)')
	bns = eval('bn * sin(pg * n * tm)')

	for i in range(1,nterms):
		ansub = anc(n = i)
		bnsub = bns(n = i)
		# print(ansub)
		# print('+')
		# print(bnsub)
		# print('+')
		four = four+eval('ansub + bnsub')		

	return (four, f1, a0/2)

# def function(n,t):

# 	fn = eval('(d * e**(-k_a*t))')
# 	if(n == 0):
# 		func = fn(t_gap = tg, k_a = k, d0 = din)
# 	else:
# 		lv = fn(t_gap = tg, k_a = k, d0 = din, t = t - n*tg)
# 		func = fn(t_gap = tg, k_a = k, d0 = lv+din)
# 	return func(t = t);

def plotFourier1(tg,k,p,din,f, nterms, optOff):

	(fn, f1, f2, a0) = getFourier(nterms)
	dfn = getFourierDerivative(fn)

	pfr = fn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta =f, off = 0)
	defs = dfn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta =f, off = 0)
	pf1 = f1(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta =f, off = 0)
	pf2 = f2(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta =f, off = 0)

	#(offset, tm1) = getOffset(pfr, tg - p, (tg/N))
	offset = round(optOff, 10)
	print('plotFourier1-offset: ', optOff, offset, din)

	print(pfr(tm=0))

	(fn1, f11, f21, a01) = getFourier(nterms)
	pfr1 = fn1(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta =f, off = offset)

	plt1 = plot(pfr(tm), tm, 0, 40, color = 'green', legend_label='Fourier series approximation')
	i = tg
	plt2 = plot(pf1(tm), tm, 0, tg -p, color = 'blue') + plot(pf2(tm), tm, tg -p, tg, color = 'blue', legend_label='Piecewise steady state approximation')
	# plot2 = plt2
	r0 = i
	r2 = tg
	lv = pf2(tm = tg)
	nv = pf1(tm = 0)
	while(i <= 40):
		plt2 = plt2+ line([(r2,lv), (r2,nv)], rgbcolor = (0, 0, 1))
		r0 = i
		r1 = i+tg -p
		r2 = i + tg
	#	print(i, r0, r1, r2)
		plt2 = plt2+ plot(pf1(tm-i), tm, r0, r1, color = 'blue') + plot(pf2(tm-i), tm, r1, r2, color = 'blue')

		lv = pf2(tm = r2-i)
		nv = pf1(tm = r0-i)
	#	print(lv,nv)

		i = i+tg

	d = eval('(d0/(1 - e**(-k_a * t_gap)))')
	a = eval('(d * e**(-k_a*(t_gap-eps)))')
	pg = eval('(2 *pi/t_gap)')
	pi = math.pi
	assume(t_gap>0)
	assume(t_gap - eps > 0)

	ssd = d(t_gap = tg, k_a = k, d0 = din)
	s0 = a(t_gap = tg,  eps = p, k_a = k, d0 = din)

	d1 = eval('(d0/(1 - e**(-k_a * t_gap)))')
	ssd1 = eval('(d * e**(-k_a * tm))')
	ssdFn = ssd1(t_gap = tg, k_a = k, d0 = din)

	plt7 = plot(ssdFn(tm), tm, 0, tg, color = 'black', legend_label='actual steady state function')
	i = tg
	r0 = i
	r1 = tg
	lv1 = ssdFn(tm = tg)
	nv1 = din
	while(i <= tg):
		plt7 = plt7 + line([(r1,lv1), (r1,nv1)], rgbcolor = (0, 0, 0))
		r0 = i
		r1 = i + tg
		#print(i, r0, r1)

		plt7 = plt7 + plot(ssdFn(tm-i), tm, r0, r1, color = 'black')

		lv1 = ssdFn(tm = r1-i)
		nv1 = din
	#	print('last value : ', lv1, 'din : ', din, 'next value : ', nv1)
		i = i+tg 

	plt3 = plot(pfr1(tm), tm, 0, 40, color = 'red', legend_label='Fourier series approx. with offset')
	plt4 = line([(0,s0+ssd/f), (tg, s0+ssd/f)],  rgbcolor=(1, 0, 1), linestyle='--')
	plt5 = line([(tg-p,0), (tg-p, din)],  rgbcolor=(0, 1, 0),  linestyle='--' ) + line([(tg,0), (tg, din)],  rgbcolor=(0, 1, 0),  linestyle='--' ) 
	plt6 = line([(0, -offset), (40, -offset)],  rgbcolor=(1, 0, 1),  legend_label='offset', linestyle='--' )
	
	pltf1 =  plt1 + plt2 + plt3 + plt6 #+plt7# fourier offset
	pltf2 =  plt1 + plt2 + plt7# fourier
	pltf3 =  plt2 + plt4 + plt5 + plt7# piecewise
	pltf1.save(filename = 'images/fourier_offset.png')
	pltf2.save(filename = 'images/piecewise_fourier.png')
	pltf3.save(filename = 'images/piecewise.png')
	#show(plt)
	print('a0')
	print(a01)

def getactualValueSSD():
	# 0.01171875, 0.091796875
	# 0.021484375, 0.1767578125
	# (0.00390625, 0.060546875)
	# (0.0078125, 0.1220703125)
	
	# 3.779296875, 31.998046875
	# 5.7255859375, 94.6552734375

	# 2.685546875, 22.0048828125
	# 5.349609375, 44.7978515625

	# 1.4462890625, 24.47265625
	# 2.9951171875, 50.7529296875

	# 2.404296875, 27.4580078125
	# 5.125, 56.953125

	# rmin = frac * 2.404296875
	# rmax = frac * 27.4580078125
	# (resmin1, resmin) = getactualValue(n,tg, k, p, frac, f, rmin)
	# (resmax1, resmax) = getactualValue(n,tg, k, p, frac, f, rmax)

	print("start")
	
	(der, init) = getSteadyStateDerivative()
	defs = der(t_gap = tg, k_a = k, eps = p, pi=math.pi)
	initf = init(t_gap = tg, k_a = k, eps = p, pi=math.pi)

	rmin = 0.009765625
	rmax = 0.1181640625
	var('vd')
	eqn = eval('initf == vd')
	eqn1 = eqn(vd = rmin)
	eqn2 = eqn(vd = rmax)
	dmin = solve(eqn1, d0)
	dmax = solve(eqn2, d0)
	print(dmin[0].right())
	print(dmax[0].right())
	resMin = RR(dmin[0].right())
	resMax = RR(dmax[0].right())
		
	res = (resMin, resMax)
	print(res)

def getSteadyStateDerivative():
	d = eval('(d0/(1 - e**(-k_a * t_gap)))')
	pi = math.pi
	assume(t_gap>0)

	ssd = eval('(d * e**(-k_a * tm))')

	ssf = diff(ssd, tm)
	initssf = ssd(tm = 0)
	print(ssf, initssf)
	return (ssf, initssf)

def plotactualValue(nterms, tg, k, p, frac, f, optOff, din):
	
	print('plotactualValue', din)
	print('getactualValue ',nterms, tg, k, p, frac, din,f)

	ddash = eval('(dd0/(1 - e**(-k_a * t_gap)))')
	ssd = eval('(ddash * e**(-k_a * tm))')
	ssdV = ssd(t_gap = tg, k_a = k, pi=math.pi)

	(fn, f1, f2, a0) = getFourier(nterms)
	#dfn = getFourierDerivative(fn)

	pfr = fn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)
	#defs = dfn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = 0)

	#offset = getOffset(pfr, defs, nterms, tg, p)
	print('offset: ', optOff)

	offset = getOptimizedOffset(nterms, tg, k, p, f, din)
	print('plotactualValue-offset: ', offset, optOff)
	
	#offset = getOffsetFnc(pfr, tg, p, nterms)
	print('sageCal offset: ', offset, tg, p, nterms)

	#(fn1, f11, f21, a01) = getFourier(nterms)
	pfr1 = fn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta = f, off = optOff)
	
	ifn1 = definite_integral(pfr1, tm, 0, tg)
	issd = definite_integral(ssdV, tm, 0, tg)
	eqn = eval('issd == ifn1')
	#print(issd, ifn1)
	#print(eqn)
	av = solve(eqn, dd0)
	print(av[0].left(), "=", av[0].right())
	res = RR(av[0].right())
	print(res, res/frac)

	ssdvddo = ssdV(dd0 = res)
	
	#print('\n Steadystate :', ssdvddo)
	#print('\n Fourier :', pfr1)
	plt1 = plot(pfr1(tm), tm, 0, tg, color = 'green',  fill=True) 
	plt2 = plot(ssdvddo(tm), tm, 0, tg,  color = 'blue', fill=True) + line([(0, res), (tg, res)],  rgbcolor=(1, 0, 1),  legend_label='Steady state dose', linestyle='--' )
	plt = plt1 + plt2
	plt1.save(filename = 'images/actualfr.png')
	plt.save(filename = 'images/actualssd.png')

	# et = p #Nrip*Tg/(Nterms)
	# fet1 = math.floor(et)
	# (fet, chk) = getRange(et, nterms, tg)
	# print('chk: ',chk, fet, fet1)

	# pfr3 = fn(t_gap = tg, k_a = k, eps = et, d0 = din, pi=math.pi, delta = f, off = 0)
	# plt10 = plot(pfr3(tm), tm, 0, tg, color = 'brown')
	# (off, tm2) = getOffset(pfr3, tg- fet, chk)
	# plt12 = line([(tm2,0), (tm2,-off)], rgbcolor = (1, 0, 1)) 
	# plt122 = line([(tg-et,0), (tg-et,din)], rgbcolor = (1, 0, 1))
	# plt13 = plt10 + plt12 + plt122
	# plt13.save(filename = 'images/fourier22.png')

	return (res, res/frac)
	# else:
	# 	print(din, din/0.8)
	# 	return (din, din/0.8)

	# av1 = find_root(eqn, dd0, 0, tg)
	# print(av1)


if __name__ == "__main__": 
	main(sys.argv[1:])


# # ddash = eval('(dd0/(1 - e**(-k_a * t_gap)))')
# 	# ssd = eval('(ddash * e**(-k_a * tm))')
# 	ssd = eval('((dd0 * e**(-k_a * tm))/(1 - e**(-k_a * t_gap)))')
# 	ssdV = ssd(t_gap = tg, k_a = k, eps = p, pi=math.pi, delta =f)

# 	(fn, f1, f2, a0) = getFourier(nterms)
# 	dfn = getFourierDerivative(fn)

# 	pfr = fn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta =f, off = 0)
# 	defs = dfn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta =f, off = 0)

# 	#(offset, tm1) = getOffset(pfr, tg - p, (tg/N))
# 	offset = optOff
# 	print('offset: ', offset)

# 	#if(offset > 0.0):
# #	(fn1, f11, f21, a01) = getFourier(nterms)
# 	pfr1 = fn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, delta =f, off = offset)
		
# 	issd = definite_integral(ssdV, tm, 0, tg)
# 	ifn1 = definite_integral(pfr1, tm, 0, tg)
# 	eqn = eval('issd == ifn1')
# 	#print(issd, ifn1)
# 	#print(eqn)
# 	av = solve(eqn, dd0)
# 	#print(av[0].left(), "=", av[0].right())
# 	res = RR(av[0].right())
# 	print(res, res/frac)
# 	print(issd(dd0=res), ifn1)

# 	ssdvddo = ssdV(dd0 = res)
