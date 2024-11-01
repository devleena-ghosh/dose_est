import os
import subprocess
import re
import sys, getopt
import math as math
from sympy import *
import numpy as np
import matplotlib.pyplot as plt 



ln_2 = 0.693
pi =3.1412
frac =0.8
#t_half = 1.25
#k_a = ln_2 / t_half
#t_gap =1.25 * 3 / 2;
#kt =( k_a * t_gap )
#pg =( 2 * pi / t_gap )

t_unit = 24
t_day =( 2 * pi / 5 )
td_hr =( t_day * t_unit )
freq =( 2 * pi / td_hr )
t4_h =( 7 * td_hr )
t_half =( td_hr / 12 )
k_el =( ln_2 / t_half )
k_a =( 2 * k_el )
t_gap =( td_hr / 3 )
kt =( k_a * t_gap )
d0 = 0.5 * frac
eps = 1
m = t_gap - eps
num = 10
n = 1
f1 = 0
f =0
tp = 0

def main(argv):
	global n1
	#f = calculateFourier(t)
	#der = getDerivative(n1)
	#print('derivative(fs, t) :', der)

	t = np.arange(0, 30, 0.001) 
	
	vfun = np.vectorize(fourier)   
	#n1 = math.floor(5/tg)
	f = vfun(num, t)

	# sigf1 = np.vectorize(smoothstep)
	# sf2 = sigf1(0.01, d, t)

	plt.plot(t, f, 'b-')
	plt.xlabel('time ->')
	plt.show()


def function(t):
	global n
	global f1
	global f
	global tp
	#print(f1, 'n =', n, t, tp, tg)
	if (n > 0):
		if (math.fabs(t - n*t_gap) <= 0.001) and (t > (n-1)*t_gap):
			f = f1+ d0
			n += 1
			print("dosage applied")
			#return f
		elif(t > (n-1)*t_gap):	
			tp =  t - (n-1)*t_gap
			f1 = f* math.exp(-k_a*tp)

		#f1 = d0 * (1 - exp(-n *k*tg)) / exp(-k*tg)
	return '{:.5f}'.format(f1)

def an0(n):
	return ((2 * d0 * t_gap)*(p(n) * ekm() * math.sin(m* p(n)) - k_a * (1+ ekm() * math.cos(m*p(n)))))/((1- exp(-k_a * t_gap))*(4 * pow(pi,2) * pow(n,2) + pow(k_a,2) * pow(t_gap,2)))

def an1(n):
	return (- d0 * ekm() * math.sin(m* p(n)))/((1- exp(-k_a * t_gap))*( pi * n))

def an2(n):
	return ((2 * d0 * pow(t_gap, 2)) *(1- ekm())*(4*pi*n - 2* p(n) * m * math.cos(m* p(n)) - (pow((p(n)*m),2)-2) * math.sin(m*p(n))))/((1- exp(-k_a * t_gap))*(8 * pow(pi,3) * pow(n,3) * pow(eps,2)))

def bn0(n):
	return ((2 * d0 * t_gap)*(p(n) - ekm() * (k_a * math.sin(m* p(n)) - p(n) * math.cos(m*p(n)))))/((1- exp(-k_a * t_gap))*(4 * pow(pi,2) * pow(n,2) + pow(k_a,2) * pow(t_gap,2)))

def bn1(n):
	return ( -d0 * ekm() * (1 - math.cos(m* p(n))))/((1- exp(-k_a * t_gap))*( pi * n))

def bn2(n):
	return ((2 * d0 * pow(t_gap, 2)) *(1- ekm()) * ((2 - 4*pow(pi,2)*pow(n,2)) - 2* p(n) * m * math.sin(m* p(n)) - (2 -pow((p(n)*m),2)) * math.cos(m*p(n))))/((1- exp(-k_a * t_gap))*(8 * pow(pi,3) * pow(n,3) * pow(eps,2)))

def ekm():
	return exp(-k_a * m)

def p(n):
	return (2 * pi * n)/t_gap

def str_p(n):
	return '(pg * '+str(n)+' )'
	
def approx(tm):
	#val = ( frac * d0 ) * ( ( 1 / ( 1 - exp(-kt) ) ) - ( ( 1 - exp(-kt) ) / ( kt ) ) );
	#der = (- pg) * (((2 * d0 * t_gap)/(4 * pow(pi, 2) *1+ pow(kt, 2))) *  k_a * (1 - exp(-kt))* 1* sin((pg * 1 ) * tm)+((2 * d0 * t_gap)/(4 * pow(pi, 2) *4+ pow(kt, 2))) *  k_a * (1 - exp(-kt))* 2* sin((pg * 2 ) * tm)+((2 * d0 * t_gap)/(4 * pow(pi, 2) *9+ pow(kt, 2))) *  k_a * (1 - exp(-kt))* 3* sin((pg * 3 ) * tm)+((2 * d0 * t_gap)/(4 * pow(pi, 2) *16+ pow(kt, 2))) *  k_a * (1 - exp(-kt))* 4* sin((pg * 4 ) * tm)+((2 * d0 * t_gap)/(4 * pow(pi, 2) *25+ pow(kt, 2))) *  k_a * (1 - exp(-kt))* 5* sin((pg * 5 ) * tm)-((2 * d0 * t_gap)/(4 * pow(pi, 2) *1+ pow(kt, 2))) * (pg * 1 )* (1 - exp(-kt))* 1* cos((pg * 1 ) * tm)-((2 * d0 * t_gap)/(4 * pow(pi, 2) *4+ pow(kt, 2))) * (pg * 2 )* (1 - exp(-kt))* 2* cos((pg * 2 ) * tm)-((2 * d0 * t_gap)/(4 * pow(pi, 2) *9+ pow(kt, 2))) * (pg * 3 )* (1 - exp(-kt))* 3* cos((pg * 3 ) * tm)-((2 * d0 * t_gap)/(4 * pow(pi, 2) *16+ pow(kt, 2))) * (pg * 4 )* (1 - exp(-kt))* 4* cos((pg * 4 ) * tm)-((2 * d0 * t_gap)/(4 * pow(pi, 2) *25+ pow(kt, 2))) * (pg * 5 )* (1 - exp(-kt))* 5* cos((pg * 5 ) * tm));
	#val += der * tm;

	val = d0 *((1/ (1 - exp(-kt))) + ((1 - exp(-kt))/kt))+	((2 * d0 * t_gap)/(4 * pow(pi,2) * 1 + pow(kt,2)) *  k_a * (1 - exp(-kt))* cos((pg * 1 ) * tm))+	((2 * d0 * t_gap)/(4 * pow(pi,2) * 4 + pow(kt,2)) *  k_a * (1 - exp(-kt))* cos((pg * 2 ) * tm))+((2 * d0 * t_gap)/(4 * pow(pi,2) * 9 + pow(kt,2)) *  k_a * (1 - exp(-kt))* cos((pg * 3 ) * tm))+((2 * d0 * t_gap)/(4 * pow(pi,2) * 16 + pow(kt,2)) *  k_a * (1 - exp(-kt))* cos((pg * 4 ) * tm))+((2 * d0 * t_gap)/(4 * pow(pi,2) * 25 + pow(kt,2)) *  k_a * (1 - exp(-kt))* cos((pg * 5 ) * tm))+((2 * d0 * t_gap)/(4 * pow(pi,2) * 1 + pow(kt,2)) * (pg * 1 )* (1 - exp(-kt))* sin((pg * 1 ) * tm))+((2 * d0 * t_gap)/(4 * pow(pi,2) * 4 + pow(kt,2)) * (pg * 2 )* (1 - exp(-kt))* sin((pg * 2 ) * tm))+((2 * d0 * t_gap)/(4 * pow(pi,2) * 9 + pow(kt,2)) * (pg * 3 )* (1 - exp(-kt))* sin((pg * 3 ) * tm))+((2 * d0 * t_gap)/(4 * pow(pi,2) * 16 + pow(kt,2)) * (pg * 4 )* (1 - exp(-kt))* sin((pg * 4 ) * tm))+((2 * d0 * t_gap)/(4 * pow(pi,2) * 25 + pow(kt,2)) * (pg * 5 )* (1 - exp(-kt))* sin((pg * 5 ) * tm));
	return val

def fourier(N, t):
	#term = (1 - math.exp(-k_a * t_gap))
	a0 = (d0 * (eps +(1/k_a + eps/3)*(1 - exp(-k_a *(t_gap - eps)))))/(t_gap* (1- exp(-k_a * t_gap)))
	an = 0
	bn = 0
	for i in range(1, N+1):
		an = an + (an0(i)+ an1(i)+ an2(i)) * math.cos(p(i) * t)
		bn = bn + (bn0(i)+ bn1(i)+ bn2(i)) * math.sin(p(i) * t)
	#print(der)
	return a0 + an + bn

def getDerivative(N):
	fs = ''
	#term = '(1 - exp(-kt))'
	a0 = '(d0/t_gap) * (eps +(1/k_a + eps/3)*(1 - exp(-k_t + keps)))'
	an = ''
	bn = ''
	adn = ''
	bdn = ''
	for i in range(1, N+1):
		an = an + '+\n' + str_c(i) + ' *  k_a ' + '* cos('+ str_p(i)+' * tm))'
		bn = bn + '+\n' + str_bn(i)  + '* sin('+ str_p(i)+' * tm))'

		if(i == 1):
			adn = str_bn(i) + '* '+ str_p(i) + '* cos('+ str_p(i)+' * tm))'
			bdn = bdn + '-\n' + str_bn(i) + '* k_a * sin('+ str_p(i)+' * tm))'
		else:
			adn = adn + '+\n' + str_bn(i) + '* '+ str_p(i) + '* cos('+ str_p(i)+' * tm))'
			bdn = bdn + '-\n' + str_bn(i) + '* k_a * sin('+ str_p(i)+' * tm))'

	fs = a0 + an+ bn
	der = '(' + adn + bdn + ')'
	print(fs + '\n')
	print(der)
	return der

def str_ano(n):
	return ((2 * d0 * t_gap)*(p(n) * ekm() * math.sin(m* p(n)) - k_a * (1+ ekm() * math.cos(m*p(n)))))/((1- exp(-k_a * t_gap))*(4 * pow(pi,2) * pow(n,2) + pow(k_a,2) * pow(t_gap,2)))

def str_an1(n):
	return (- d0 * ekm() * math.sin(m* p(n)))/((1- exp(-k_a * t_gap))*( pi * n))

def str_an2(n):
	return ((2 * d0 * pow(t_gap, 2)) *(1- ekm())*(4*pi*n - 2* p(n) * m * math.cos(m* p(n)) - (pow((p(n)*m),2)-2) * math.sin(m*p(n))))/((1- exp(-k_a * t_gap))*(8 * pow(pi,3) * pow(n,3) * pow(eps,2)))

def str_bno(n):
	return ((2 * d0 * t_gap)*(p(n) - ekm() * (k_a* math.sin(m* p(n)) - p(n) * math.cos(m*p(n)))))/((1- exp(-k_a * t_gap))*(4 * pow(pi,2) * pow(n,2) + pow(k_a,2) * pow(t_gap,2)))

def _strbn1(n):
	return ( -d0 * ekm() * (1 - math.cos(m* p(n))))/((1- exp(-k_a * t_gap))*( pi * n))

def str_bn2(n):
	return ((2 * d0 * pow(t_gap, 2)) *(1- ekm()) * ((2 - 4*pow(pi,2)*pow(n,2)) - 2* p(n) * m * math.sin(m* p(n)) - (2 -pow((p(n)*m),2)) * math.cos(m*p(n))))/((1- exp(-k_a * t_gap))*(8 * pow(pi,3) * pow(n,3) * pow(eps,2)))

def str_ekm():
	return exp(-k_a * m)

def newtonraphson(f, f_, x0, TOL=0.001, NMAX=100):
	"""
	Takes a function f, its derivative f_, initial value x0, tolerance value(optional) TOL and
	max number of iterations(optional) NMAX and returns the root of the equation
	using the newton-raphson method.
	"""
	n=1
	while n<=NMAX:
		x1 = x0 - (f(x0)/f_(x0))
		if abs(x1 - x0) < TOL:
			return x1
		else:
			x0 = x1
	return False

if __name__ == "__main__": 
   main(sys.argv[1:])

	def func(x):
		"""
		Function x^3 - x -2
		We will calculate the root of this function using different methods.
		"""
		return math.pow(x,3) - x -2

	def func_(x):
		"""
		Derivate of the function f(x) = x^3 - x -2
		This will be used in Newton-Rahson method.
		"""
		return 3*math.pow(x,2)-1

	#Invoking Newton Raphson Method
	res = newtonraphson(func,func_,1)
	print res
