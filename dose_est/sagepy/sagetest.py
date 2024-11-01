#!/usr/bin/env sage -python
from sage.all import *
from sage.symbolic.integration.integral import definite_integral
from sage.symbolic.integration.integral import indefinite_integral

var('d0, k_a, eps, t_gap, n, t, pg, pi, off, f, dd0')
d = eval('(d0/(1 - e**(-k_a * t_gap)))')
a = eval('(d * e**(-k_a*(t_gap-eps)))')
pg = eval('(2 *pi/t_gap)')
pi = math.pi
assume(t_gap - eps > 0)

#from sage.symbolic.integration.integral import definite_integral

def main(argv):

	tg = 10.05184
	N = 20
	din = 0.8 *3
	k = 0.18384693747612374
	p = 1
	f = 8

	(fn, f1, f2, a0) = getFourier(N)
	dfn = getFourierDerivative(fn)

	pfr = fn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi,f = f, off = 0)
	defs = dfn(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi,f = f, off = 0)
	pf1 = f1(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, f = f, off = 0)
	pf2 = f2(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, f = f, off = 0)

	offset = getOffset(pfr, defs, N, tg, p)
	print('offset: ', offset)


	(fn1, f11, f21, a01) = getFourier(N)
	pfr1 = fn1(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, f = f, off = offset)

	plt1 = plot(pfr(t), t, 0, 20)
	i = tg
	plt2 = plot(pf1(t), t, 0, tg -p, color = 'red') + plot(pf2(t), t, tg -p, tg, color = 'red')
	while(i <= 20):
		r0 = i
		r1 = i+tg -p
		r2 = i + tg
		plot2 = plt2+ plot(pf1(t), t, i, r1, color = 'red') + plot(pf2(t), t, r1, r2, color = 'red')
		i = i+tg

	plt3 = plot(pfr1(t), t, 0, 20, color = 'green')
	plt = plt1+plt2+plt3
	plt.save('fourier.png')
	#show(plt)
	print('a0')
	print(a01)

def getOffset(fn, dfn, N, tg, p):
	#print(fn)
	#print(dfn)
	
	res1 = bisection(dfn, tg-p, tg, 0.001)
#	print(res1)
	offset = fn(t = res1)
	#print(offset)
	if(offset > 0):
		t1 = (res1 + 0.001)
	#	print(t1)
		res2 = bisection(dfn, t1, tg, 0.001)
		#print(res2)
		offset2 = fn(t = res2)
		#print(offset2)
		if(offset2 < 0 ):
			#fr = fn - offset2;
			return -1 *offset2
		else:
			return 0
	else:
		return -1*offset

def getactualValue(N, tg, k, p, frac, din,f = 8):
	ddash = eval('(dd0/(1 - e**(-k_a * t_gap)))')
	ssd = eval('(ddash * e**(-k_a * t))')
	ssdV = ssd(t_gap = tg, k_a = k, eps = p, pi=math.pi, f = f)

	(fn, f1, f2, a0) = getFourier(N)
	dfn = getFourierDerivative(fn)

	pfr = fn(t_gap = tg, k_a = k, eps = p, d0 = frac*din, pi=math.pi, f = f, off = 0)
	defs = dfn(t_gap = tg, k_a = k, eps = p, d0 = frac* din, pi=math.pi, f = f, off = 0)
	pf1 = f1(t_gap = tg, k_a = k, eps = p, d0 = frac*din, pi=math.pi, f = f, off = 0)
	pf2 = f2(t_gap = tg, k_a = k, eps = p, d0 = frac*din, pi=math.pi, f = f, off = 0)

	offset = getOffset(pfr, defs, N, tg, p)
	print('offset: ', offset)

	(fn1, f11, f21, a01) = getFourier(N)
	pfr1 = fn1(t_gap = tg, k_a = k, eps = p, d0 = din, pi=math.pi, f = f, off = offset)
	
	ifn1 = definite_integral(pfr1, t, 0, tg)
	issd = definite_integral(ssdV, t, 0, tg)
	eqn = eval('issd == ifn1')
	print(issd, ifn1)
	print(eqn)
	av = solve(eqn, dd0)
	print(av[0].left(), "=", av[0].right())
	res = RR(av[0].right())
	print(res, res/0.8)
	return (res, res/0.8)

	# av1 = find_root(eqn, dd0, 0, tg)
	# print(av1)



def getFourier(N):

	s = eval('(t-t_gap+eps)')
	x = eval('((eps**2)/(4 *(d/f)))')

	f1 = eval('(d * e**(-k_a * t))')
	f2 = eval('(a + (s**2)/(4 * x))')
	# print(f1)
	# print(f2)

	a01 = definite_integral(f1,t,0,(t_gap-eps))
	a02 = definite_integral(f2,t,(t_gap-eps), t_gap)
	a0 = eval('(2 *(a01+a02)/t_gap)')
	#print(a0)

	f1c = eval('f1 * cos(pg * n * t)')
	f2c = eval('f2 * cos(pg * n * t)')
	an1 = definite_integral(f1c,t,0,(t_gap- eps))
	an2 = definite_integral(f2c,t,(t_gap- eps), t_gap)
	an = eval('(2 *(an1 + an2)/t_gap)')

	#print(an)

	f1s = eval('f1 * sin(pg * n * t)')
	f2s = eval('f2 * sin(pg * n * t)')
	bn1 = definite_integral(f1s,t,0,(t_gap- eps))
	bn2 = definite_integral(f2s,t,(t_gap- eps), t_gap)
	bn = eval('(2 *(bn1 + bn2)/t_gap)')
	#print(bn)

	# print('a0 = ', a0)
	# print('an = ', an)
	# print('bn = ', bn)
	# print('\n')

	four = eval('a0/2 + off')
	anc = eval('an * cos(pg * n * t)')
	bns = eval('bn * sin(pg * n * t)')

	for i in range(1,N):
		ansub = anc(n = i)
		bnsub = bns(n = i)
		# print(ansub)
		# print('+')
		# print(bnsub)
		# print('+')
		four = eval('four + ansub + bnsub')

	#print('\n fourier upto 10 terms: \n')
	#print(four)
	return (four, f1, f2, a0/2)

def getSteadyStateDerivative():
	ssd = eval('(d * e**(-k_a * t))')

	ssf = diff(ssd, t)
	initssf = ssd(t = 0)

	return (ssf, initssf)


def getFourierDerivative(f):

	der = diff(f, t)

	#print('\n derivative of fourier: \n')
	#print(der)
	return der

def getIntegral(f):
	it = indefinite_integral(f,t)
	print(it)
	return it

def bisection(f, a,b,tol):
	c = (a+b)/2.0
	while (b-a)/2.0 > tol:
		if f(t=c) == 0:
			return c
		elif f(t=a)*f(t=c) < 0:
			b = c
		else :
			a = c
		c = (a+b)/2.0
		
	return c

if __name__ == "__main__": 
	main(sys.argv[1:])

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

	# show(plot(pfr(t), t, 0, 40)+ plot(pf1(t), t, 0, 20 - 1, color='red'))

# fourdashn = ((pg * n) * ((bn * cos((pg * n)* t)) - (an * sin((pg * n)* t))))
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

