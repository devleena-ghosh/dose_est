#!/usr/bin/env sage -python
from sage.all import *
from sage.symbolic.integration.integral import definite_integral
from sage.symbolic.integration.integral import indefinite_integral

from sageCal import *

def main(argv):

	nterms = 20
	
	flag = True

	case = 'hypo2_v2'
	(rmin, rmax) = (0.021484375, 16.835937)
	#(3.509765625, 33.6328125)
	#(5.49609375, 52.23828125)
	#(3.2041015625, 47.2607421875)
	#(5.7060546875, 48.5166015625)
	#(2.4111328125, 27.814453125)
	#(1.4501953125, 24.7900390625)
	#(3.5771484375, 39.3427734375)
	#(2.6923828125, 25.7021484375)
	#(0.00390625, 0.060546875)
	#(0.01171875, 0.091796875)
	#(0.005859375, 0.064453125)
	#(0.0078125, 0.1220703125)
	#(0.0107421875, 0.12890625)
	#(0.0107421875, 0.12890625)
	#(0.005859375, 0.064453125)
	#(0.01171875, 0.091796875)
	#0.00390625, 0.060546875)

	#(0.01171875, 0.091796875	)
	#(0.005859375, 0.064453125)
	#(1.4501953125, 24.7900390625)
	# k_a = (5 * ( 0.693 / ( 7 * ( ( 2 * 3.1412 / 5 ) * 24 ) ) ) )
	# t_g = ( ( 2 * 3.1412 / 5 ) * 24 )

	#hyper_v2:
	#k_a = ( 2 * ( 0.693 / ( ( ( 2 * 3.1412 / 5 ) * 24 ) / 4 ) ) )
	#t_g = ( 2 * ( ( 2 * 3.1412 / 5 ) * 24 ) / 3 )

	#hyper_v1:
	#k_a = ( 2 * ( 0.693 / ( ( ( 2 * 3.1412 / 5 ) * 24 ) / 4 ) ) )
	#t_g = ( ( ( 2 * 3.1412 / 5 ) * 24 ) / 3 )

	#hypo_v1:
	#k_a = ( 5 * ( 0.693 / ( 7 * ( ( 2 * 3.1412 / 5 ) * 24 ) ) ) )
	#t_g = ( ( 2 * 3.1412 / 5 ) * 24 )

	#hypo_v2:
	k_a = ( 5 * ( 0.693 / ( 7 * ( ( 2 * 3.1412 / 5 ) * 24 ) ) ) )
	t_g = ( 2 * ( ( 2 * 3.1412 / 5 ) * 24 ) )

	print(rmin, rmax)

	if(flag):
		(off, ffs, p1, f1, tm1) = optimizeSmootherfunction(nterms, t_g, k_a, rmin)
		print(p1, f1)
		(off, ffs, p2, f2, tm2) = optimizeSmootherfunction(nterms, t_g, k_a, rmax)
		print(p2, f2)
		(rmin1, resMin) = getactualValue(nterms, t_g, k_a, p1, 0.8, f1, 0.8*rmin)
		(rmax1, resMax) = getactualValue(nterms, t_g, k_a, p2, 0.8, f2, 0.8*rmax)
		res1 = (rmin1, rmax1)
		res = (resMin, resMax)
		print(case, 'result (actual/0.8, actual)', res, res1)

	else:
		(rmin1, resMin) = getSSDActualValue(nterms, t_g, k_a, 0.8, rmin)
		(rmax1, resMax) = getSSDActualValue(nterms, t_g, k_a, 0.8, rmax)
			#res = (resMin, resMax)
		res = (rmin1, rmax1)
		print('result ', res)


if __name__ == "__main__": 
	main(sys.argv[1:])





