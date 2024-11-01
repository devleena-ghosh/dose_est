#!/usr/bin/env sage -python
from sage.all import *
import sys
import numpy as np
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sagepy.sageCal import *
from sagepy.optimizeOffset import *
from sagepy.drugFourier import *
SIMPLE = True
nterms = 20
LOOP =  True 
#LOOP = False
'''

			d/dt[q_14]= -kdelay*q_14 + q_7;\
			d/dt[q_15]= kdelay*(q_14 - q_15);\
			d/dt[q_16]= kdelay*(q_15 - q_16); \
			d/dt[q_17]= kdelay*(q_16 - q_17);\
			d/dt[q_18]= kdelay*(q_17 - q_18);\
			d/dt[q_19]= kdelay*(q_18 - q_19);\

			q_14(t0)  = ic_14;\
			q_15(t0)  = ic_15;\
			q_16(t0)  = ic_16;\
			q_17(t0)  = ic_17;\
			q_18(t0)  = ic_18;\
			q_19(t0)  = ic_19;\
ic_0 = 0.0;\
			ic_1 = 0.322114215761171;\
			ic_2 = 0.201296960359917;\
			ic_3 = 0.638967411907560;\
			ic_4 = 0.00663104034826483;\
			ic_5 = 0.0112595761822961;\
			ic_6 = 0.0652960640300348;\
			ic_7 = 1.78829584764370;\
			ic_8 = 7.05727560072869;\
			ic_9 = 7.05714474742141;\
			ic_10 = 0.0;\
			ic_11 = 0.0;\
			ic_12 = 0.0;\
			ic_13 = 0.0;\
			ic_14 = 3.34289716182018;\
			ic_15 = 3.69277248068433;\
			ic_16 = 3.87942133769244;\
			ic_17 = 3.90061903207543;\
			ic_18 = 3.77875734283571;\
			ic_19 = 3.55364471589659;\
			\
			p_44 = d_2;\
			p_46 = d_4;\
			p_44 = d_2;\
			p_46 = d_4;\
			p_44 = 0.12*d_2;\
			p_46 = 0.12*d_4;\
			'''
basic_eqn = "Equations:\
			p_44 = 0.12*d_2;\
			p_46 = 0.12*d_4;\
			q_19 = q_7;\
			\
			q4F = (p_24 + p_25 * q_1 + p_26 * q_1**2 + p_27 * q_1**3) * q_4;\
			q1F = (p_7 + p_8 * q_1 + p_9 * q_1**2 + p_10 * q_1**3) * q_1;\
			SR3 = (p_19 * q_19) * d_1;\
			SR4 = (p_1 * q_19) * d_1;\
			fCIRC = 1 + (p_32 / (p_31 * exp(-q_9)) - 1)*(1/(1 + exp(10*q_9 - 55)));\
			SRTSH = (p_30 + p_31*fCIRC*sin(p_0*tm - p_33))*exp(-q_9);\
			fdegTSH = p_34 + p_35/(p_36+q_7);\
			fLAG = p_41 + 2*q_8**11/(p_42**11 + q_8**11);\
			f4 = p_37 + 5*p_37/(1+exp(2*q_8 - 7));\
			NL = p_13/(p_14 + q_2);\
			\
			T4conv  = 777.0/p_47;\
			T3conv  = 651.0/p_47;\
			TSHconv = 5.6/p_48;\
			\
			T4 = q_1*T4conv;\
			T3 = q_4*T3conv;\
			TSH = q_7*TSHconv;\
			FT3 = q4F*T3conv*0.5;\
			FT4 = q1F*T4conv*0.45;\
			FT3_1 = q4F*T3conv;\
			FT4_1 = q1F*T4conv;\
			eT4 = q_10*777.0;\
			eT3 = q_12*651.0;\
			edT4 = q_11*777.0;\
			edT3 = q_13*651.0;\
			\
			d/dt[tm] = 1.0;\
			d/dt[q_1] = SR4 + p_3 * q_2 + p_4 * q_3 - (p_5 + p_6) * q1F + p_11 * q_11 + u1;\
			d/dt[q_2] = p_6 * q1F - (p_3 + p_12 + NL) * q_2;\
			d/dt[q_3] = p_5 * q1F - (p_4 + p_15 / (p_16 + q_3) + p_17/(p_18 + q_3)) * q_3;\
			d/dt[q_4] = SR3 + p_20 * q_5 + p_21 * q_6 - (p_22 + p_23) * q4F + p_28 * q_13 + u4;\
			d/dt[q_5] = p_23 * q4F + NL * q_2 - (p_20 + p_29) * q_5; \
			d/dt[q_6] = p_22 * q4F + p_15 * q_3 / (p_16 + q_3) + p_17 * q_3/(p_18 + q_3) - (p_21)*q_6;\
			\
			d/dt[q_7] = SRTSH - fdegTSH * q_7;\
			d/dt[q_8] = f4 / p_38 * q_1 + p_37/p_39 * q_4 - p_40 * q_8;\
			d/dt[q_9] = fLAG * (q_8 - q_9);\
			d/dt[q_10] = {0};\
			d/dt[q_11] =  p_43 * q_10 - (p_44 + p_11) * q_11;\
			d/dt[q_12] = {1};\
			d/dt[q_13] =  p_45 * q_12 - (p_46 + p_28) * q_13;\
			\
			Inits:\
			\
			tm(t0) = ic_0;\
			q_1(t0) = ic_1;\
			q_2(t0)  = ic_2;\
			q_3(t0)  = ic_3;\
			q_4(t0)  = ic_4;\
			q_5(t0)  = ic_5;\
			q_6(t0)  = ic_6;\
			q_7(t0)  = ic_7;\
			q_8(t0)  = ic_8;\
			q_9(t0)  = ic_9;\
			q_10(t0)  = et4_init;\
			q_11(t0)  = ic_11;\
			q_12(t0)  = et3_init;\
			q_13(t0)  = ic_13;\
			\
			Constraints:\
			tm > 2.4;\
			tm < 3.0;\
			"
inputs = "Functions:\
			def ip(tm, x, tg, d):\
				if (tm % tg) == 0:\
					return x + d;"
'''
def getEquations(t_units):
	equations = {}
	equation = basic_eqn
	for t_unit in t_units:
		equations.update({t_unit:equation})
	return equations
	# return equation
'''
u1 = 0.0;
u4 = 0.0;
kdelay = 5.0/8;
d_0 = 0.0;
d_1 = 1.0;
d_2 = 0.88;
d_3 = 1.0;
d_4 = 0.88;

p_0 = 2*math.pi/24.0;
p_1 = 0.00174155;
p_2 = 8;
p_3 = 0.868;
p_4 = 0.108;
p_5 = 584;
p_6 = 1503;
p_7 = 0.000289;
p_8 = 0.000214;
p_9 = 0.000128;
p_10 = -8.83e-6;
p_11 = 0.88;
p_12 = 0.0189;
p_13 = 0.00998996;
p_14 = 2.85;
p_15 = 6.63e-4;
p_16 = 95;
p_17 = 0.00074619;
p_18 = 0.075;
p_19 = 3.3572e-4;
p_20 = 5.37;
p_21 = 0.0689;
p_22 = 127;
p_23 = 2043;
p_24 = 0.00395;
p_25 = 0.00185;
p_26 = 0.00061;
p_27 = -0.000505;
p_28 = 0.88;
p_29 = 0.207;
p_30 = 1166;
p_31 = 581;
p_32 = 2.37;
p_33 = -3.71;
p_34 = 0.53;
p_35 = 0.037;
p_36 = 23;
p_37 = 0.118;
p_38 = 0.29;
p_39 = 0.006;
p_40 = 0.037;
p_41 = 0.0034;
p_42 = 5;
p_43 = 1.3; 
p_44 = 0.12*d_2;
p_45 = 1.78;
p_46 = 0.12*d_4;
p_47 = 3.2;
p_48 = 5.2;

# parameter variations
default_var = 10.0


ic_0 = 0.0;
ic_1 = 0.322114215761171;
ic_2 = 0.201296960359917;
ic_3 = 0.638967411907560;
ic_4 = 0.00663104034826483;
ic_5 = 0.0112595761822961;
ic_6 = 0.0652960640300348;
ic_7 = 1.78829584764370;
ic_8 = 7.05727560072869;
ic_9 = 7.05714474742141;
ic_10 = 0.0;
ic_11 = 0.0;
ic_12 = 0.0;
ic_13 = 0.0;
ic_14 = 3.34289716182018;
ic_15 = 3.69277248068433;
ic_16 = 3.87942133769244;
ic_17 = 3.90061903207543;
ic_18 = 3.77875734283571;
ic_19 = 3.55364471589659;


tg_t4 = 24.0
tg_t3 = 24.0
dose_t4 = 0.0 #2
dose_t3 = 0.0 #1
off_t4 = 0.0
off_t3 = 0.0
eps_t4 = 1/nterms
delta_t4 = 1.2
eps_t3 = 1/nterms
delta_t3 = 1.2
et4_init = dose_t4
et3_init = dose_t3

ic_10 = dose_t4
ic_12 = dose_t3

t4_ode = 0.0
t3_ode = 0.0

##########################
MAX_tt4=110;
MIN_tt4=45;
AVG_tt4 = (MAX_tt4+MIN_tt4)/2;

#ug/L
MAX_t4=0.017;
MIN_t4=0.008;
AVG_t4 = (MAX_t4+MIN_t4)/2;
    
MAX_tsh=3.0;
MIN_tsh=0.5;
AVG_tsh = (MAX_tsh+MIN_tsh)/2;
    
MAX_tt3=1.8;
MIN_tt3=0.6;
AVG_tt3 = (MAX_tt3+MIN_tt3)/2;

MAX_t3=0.0045;
MIN_t3=0.0022;
AVG_t3 = (MAX_t3+MIN_t3)/2;

MAX_crt_t4=0.1;

MAX_crt_tsh=10.0;
MAX_bl_tsh=4.0;
MIN_bl_tsh=0.5;
MIN_crt_tsh=0.3;

TD = 50;
TU = 24;
TT = 50;
sampHr = 1.0 if SIMPLE else 2.0
PR1 = (TD/2.0)*TU
PR2 = (TD - 1)*TU

print('Sample per hr :', sampHr)

# ng/dL -- *12.87 --> pmol/L
def FT4conv2pmol(x):
	# # ng/dL to ug/L
	# x1 = x* 0.01
	# # ug/L to uMol/L
	# x2 = x1/777.0
	# # umol/L to pmol/L
	# x3 = x2*1e6
	return x*12.87

# pmol/L -- /12.87 --> ng/dL -- 0.01 --> ug/L
def FT4conv2ug(x):
	# # ng/dL to ug/L
	# x1 = x* 0.01
	# # ug/L to uMol/L
	# x2 = x1/777.0
	# # umol/L to pmol/L
	# x3 = x2*1e6
	return 0.01*x/12.87
	
#########################
'''
rowNo = 1 #eu
classId = 0 #eu

if rowNo == 1:
	params = [0.02889856807557092, 0.7493677168556132, 0.9500572363310411, 0.08052356392413862, 0.37946065575241567, 0.04799187265983223, 1599.1568978564417, 1980.5213844857644]
	#params = [0.027804490528089868, 0.8773261380711301, 0.03979709875761355, 0.8046265701075439, 0.08141745889071453, 0.3517071124779993, 0.05082630537474972, 1527.046287442562, 2254.2787804785103]
elif rowNo == 2:	
	params = [0.00400775284830391, 0.8270300261826281, 0.8849509433961563, 0.12392171885939515, 0.22075145879809693, 0.032582016664440765, 1709.2769876440807, 1430.6966711231178]
	# params = [0.044441311448401605, 0.7150632876650352, 0.00022785061020673362, 0.9014097153216267, 0.1346852642609331, 0.2999999242972856, 0.03909706119124789, 1221.8294507924388, 2771.218679858527]
elif rowNo == 3:	
	params = [0.016090655587322554, 0.7056056647645642, 0.8532353919902227, 0.12487407540005219, 0.24371596294626613, 0.030931495675422187, 1934.1012965853483, 1436.0518030407234]
	#params = [0.0015266478748991032, 0.8786742118617643, 0.016294341287458707, 0.6495030718039532, 0.13966496649886267, 0.21304494973879118, 0.024424872652661963, 1472.4386290789344, 2014.9086725099194]
elif rowNo == 4:	
	params = [0.007571741113799129, 0.8754020228040587, 0.7534468517002062, 0.09090086724737123, 0.35081987156450717, 0.045143023434125296, 1380.9019293651977, 2604.074177933131]
	#params = [0.1836336225029186, 1.0553637335190478, 0.35592689964765367, 0.8250856387625735, 0.08550683382662697, 0.24250209924001376, 0.04606560862239511, 1739.9935885987102, 1927.2560732690185]
elif rowNo == 5:	
	params = [0.001619843052019778, 0.7522756038630971, 0.8857756116085994, 0.08785362899960446, 0.3720649463723616, 0.03842506462344339, 1555.9739157998117, 1382.942305584017]
	#params = [0.007797563733278835, 0.8431342295793149, 0.002440720234283645, 0.7698627423444713, 0.09139675660704515, 0.37280723295176776, 0.043945250399950356, 1387.5424557514348, 2143.9775644789165]	
elif rowNo == 6:
	params = [0.004152740897013057, 0.8601013333212502, 0.832396895903404, 0.09722785066355807, 0.27176799165616383, 0.04547291209904932, 1497.484261854042, 1404.4604105027527]
	#params = [0.0012172132682129553, 0.647829321613019, 0.006254091543945441, 0.6349720905823741, 0.07998912659417735, 0.3386193317240299, 0.030142584472773473, 1578.16254111129, 2365.3443151122474]
else:
	params = [0.1* d_1, d_2, d_4, p_37, p_38, p_40, p_6, p_23]
	rowNo = 0
#params = [0.027804490528089868, 0.8773261380711301, 0.03979709875761355, 0.8046265701075439, 0.08141745889071453, 0.3517071124779993, 0.05082630537474972, 1527.046287442562, 2254.2787804785103];
'''
'''
d_1 = params[0];
d_2 = params[1];
d_3 = params[2];
d_4 = params[3];
p_37 = params[4];
p_38 = params[5];
p_40 = params[6];
p_6 = params[7];
p_23 = params[8];
##########
d_1 = params[0]
d_2 = params[1]
d_4 = params[2]
p_37 = params[3]
p_38 = params[4]
p_40 = params[5]
p_6 = params[6]
p_23 = params[7]
'''

ind_u1 = 1
ind_u4 = 2
ind_kdelay = 3
ind_d_0 = 4
ind_d_1 = 5
ind_d_2 = 6
ind_d_3 = 7
ind_d_4 = 8
ind_p_0 = 9
ind_p_1 = 10
ind_p_2 = 11
ind_p_3 = 12
ind_p_4 = 13
ind_p_5 = 14
ind_p_6 = 15
ind_p_7 = 16
ind_p_8 = 17
ind_p_9 = 18
ind_p_10 = 19
ind_p_11 = 20
ind_p_12 = 21
ind_p_13 = 22
ind_p_14 = 23
ind_p_15 = 24
ind_p_16 = 25
ind_p_17 = 26
ind_p_18 = 27
ind_p_19 = 28
ind_p_20 = 29
ind_p_21 = 30
ind_p_22 = 31
ind_p_23 = 32
ind_p_24 = 33
ind_p_25 = 34
ind_p_26 = 35
ind_p_27 = 36
ind_p_28 = 37
ind_p_29 = 38
ind_p_30 = 39
ind_p_31 = 40
ind_p_32 = 41
ind_p_33 = 42
ind_p_34 = 43
ind_p_35 = 44
ind_p_36 = 45
ind_p_37 = 46
ind_p_38 = 47
ind_p_39 = 48
ind_p_40 = 49
ind_p_41 = 50
ind_p_42 = 51
ind_p_43 = 52 
ind_p_44 = 53
ind_p_45 = 54
ind_p_46 = 55
ind_p_47 = 56
ind_p_48 = 57

ind_ic_0 = 58
ind_ic_1 = 59
ind_ic_2 = 60
ind_ic_3 = 61
ind_ic_4 = 62
ind_ic_5 = 63
ind_ic_6 = 64
ind_ic_7 = 65
ind_ic_8 = 66
ind_ic_9 = 67
ind_ic_10 = 68
ind_ic_11 = 69
ind_ic_12 = 70
ind_ic_13 = 71
ind_ic_14 = 72
ind_ic_15 = 73
ind_ic_16 = 74
ind_ic_17 = 75
ind_ic_18 = 76
ind_ic_19 = 77

ind_tg_t4 = 78
ind_tg_t3 = 79
ind_dose_t4 = 80
ind_dose_t3 = 81
ind_off_t4 = 82
ind_off_t3 = 83
ind_eps_t4 = 84
ind_delta_t4 = 85
ind_eps_t3 = 86
ind_delta_t3 = 87
ind_et4_init = 88
ind_et3_init = 89
ind_t4_ode = 90
ind_t3_ode = 91


# ind_u1, ind_u4, ind_kdelay, ind_d_0, ind_d_1, ind_d_2, ind_d_3, ind_d_4, ind_p_0, ind_p_1, ind_p_2, ind_p_3, ind_p_4, ind_p_5, ind_p_6, ind_p_7, ind_p_8, ind_p_9, ind_p_10, ind_p_11, ind_p_12, ind_p_13, ind_p_14, ind_p_15, ind_p_16, ind_p_17, ind_p_18, ind_p_19, ind_p_20, ind_p_21, ind_p_22, ind_p_23, ind_p_24, ind_p_25, ind_p_26, ind_p_27, ind_p_28, ind_p_29, ind_p_30, ind_p_31, ind_p_32, ind_p_33, ind_p_34, ind_p_35, ind_p_36, ind_p_37, ind_p_38, ind_p_39, ind_p_40, ind_p_41, ind_p_42, ind_p_43, ind_p_44, ind_p_45, ind_p_46, ind_p_47, ind_p_48

# u1, u4, kdelay, d_0, d_1, d_2, d_3, d_4, p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9, p_10, p_11, p_12, p_13, p_14, p_15, p_16, p_17, p_18, p_19, p_20, p_21, p_22, p_23, p_24, p_25, p_26, p_27, p_28, p_29, p_30, p_31, p_32, p_33, p_34, p_35, p_36, p_37, p_38, p_39, p_40, p_41, p_42, p_43, p_44, p_45, p_46, p_47, p_48

ParamNames_all = ['u1', 'u4', 'kdelay', 'd_0', 'd_1', 'd_2', 'd_3', 'd_4', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7', 'p_8', 'p_9', 'p_10', 'p_11', 'p_12', 'p_13', 'p_14', 'p_15', 'p_16', 'p_17', 'p_18', 'p_19', 'p_20', 'p_21', 'p_22', 'p_23', 'p_24', 'p_25', 'p_26', 'p_27', 'p_28', 'p_29', 'p_30', 'p_31', 'p_32', 'p_33', 'p_34', 'p_35', 'p_36', 'p_37', 'p_38', 'p_39', 'p_40', 'p_41', 'p_42', 'p_43', 'p_44', 'p_45', 'p_46', 'p_47', 'p_48', 'ic_0', 'ic_1', 'ic_2', 'ic_3', 'ic_4', 'ic_5', 'ic_6', 'ic_7', 'ic_8', 'ic_9', 'ic_10', 'ic_11', 'ic_12', 'ic_13', 'ic_14', 'ic_15', 'ic_16', 'ic_17', 'ic_18', 'ic_19', 'tg_t4', 'tg_t3', 'dose_t4', 'dose_t3', 'off_t4', 'off_t3', 'eps_t4', 'delta_t4', 'eps_t3', 'delta_t3', 'et4_init','et3_init', 't4_ode', 't3_ode']

AllIndices = [ind_u1, ind_u4, ind_kdelay, ind_d_0, ind_d_1, ind_d_2, ind_d_3, ind_d_4, ind_p_0, ind_p_1, ind_p_2, ind_p_3, ind_p_4, ind_p_5, ind_p_6, ind_p_7, ind_p_8, ind_p_9, ind_p_10, ind_p_11, ind_p_12, ind_p_13, ind_p_14, ind_p_15, ind_p_16, ind_p_17, ind_p_18, ind_p_19, ind_p_20, ind_p_21, ind_p_22, ind_p_23, ind_p_24, ind_p_25, ind_p_26, ind_p_27, ind_p_28, ind_p_29, ind_p_30, ind_p_31, ind_p_32, ind_p_33, ind_p_34, ind_p_35, ind_p_36, ind_p_37, ind_p_38, ind_p_39, ind_p_40, ind_p_41, ind_p_42, ind_p_43, ind_p_44, ind_p_45, ind_p_46, ind_p_47, ind_p_48, ind_ic_0, ind_ic_1, ind_ic_2, ind_ic_3, ind_ic_4, ind_ic_5, ind_ic_6, ind_ic_7, ind_ic_8, ind_ic_9, ind_ic_10, ind_ic_11, ind_ic_12, ind_ic_13, ind_ic_14, ind_ic_15, ind_ic_16, ind_ic_17, ind_ic_18, ind_ic_19, ind_tg_t4, ind_tg_t3, ind_dose_t4, ind_dose_t3, ind_off_t4, ind_off_t3, ind_eps_t4, ind_delta_t4, ind_eps_t3, ind_delta_t3, ind_et4_init, ind_et3_init, ind_t4_ode, ind_t3_ode] #, ind_tg_t4, ind_tg_t3, ind_dose_t4, ind_dose_t3, ind_off_t4, ind_off_t3, ind_eps_t4, ind_delta_t4, ind_eps_t3, ind_delta_t3, ind_et4_init, ind_et3_init, ind_t4_ode, ind_t3_ode]

Default_Values = [u1, u4, kdelay, d_0, d_1, d_2, d_3, d_4, p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9, p_10, p_11, p_12, p_13, p_14, p_15, p_16, p_17, p_18, p_19, p_20, p_21, p_22, p_23, p_24, p_25, p_26, p_27, p_28, p_29, p_30, p_31, p_32, p_33, p_34, p_35, p_36, p_37, p_38, p_39, p_40, p_41, p_42, p_43, p_44, p_45, p_46, p_47, p_48, ic_0, ic_1, ic_2, ic_3, ic_4, ic_5, ic_6, ic_7, ic_8, ic_9, ic_10, ic_11, ic_12, ic_13, ic_14, ic_15, ic_16, ic_17, ic_18, ic_19, tg_t4, tg_t3, dose_t4, dose_t3, off_t4, off_t3, eps_t4, delta_t4, eps_t3, delta_t3, et4_init, et3_init, t4_ode, t3_ode, ]

Default_Inits = [ic_0, ic_1, ic_2, ic_3, ic_4, ic_5, ic_6, ic_7, ic_8, ic_9, ic_10, ic_11, ic_12, ic_13, ic_14, ic_15, ic_16, ic_17, ic_18, ic_19]

All_IC = [ind_ic_0, ind_ic_1, ind_ic_2, ind_ic_3, ind_ic_4, ind_ic_5, ind_ic_6, ind_ic_7, ind_ic_8, ind_ic_9, ind_ic_10, ind_ic_11, ind_ic_12, ind_ic_13, ind_ic_14, ind_ic_15, ind_ic_16, ind_ic_17, ind_ic_18, ind_ic_19]

All_IC_names = ['t4', 't4_fast', 't4_slow', 't3p', 't3_fast', 't3_slow', 'tsh_p', 't3b', 't3b_lag', 't4_pill', 't4_gut', 't3_pill', 't3_gut', 'q1', 'q2', 'q3', 'q4', 'q5', 'tsh_ds']

Default_Variations = {}
Default_Variations.update({ind_d_1:(0.00001, 1.25)})
Default_Variations.update({ind_d_2:(0.7, 0.88)})
Default_Variations.update({ind_d_3:(0.00001, 1.25)})
Default_Variations.update({ind_d_4:(0.7, 0.88)})
Default_Variations.update({ind_p_30:(0.95*p_30, 1.05*p_30)})
Default_Variations.update({ind_p_31:(0.95*p_31, 1.05*p_31)})
Default_Variations.update({ind_p_32:(0.95*p_32, 1.05*p_32)})
Default_Variations.update({ind_p_37:(0.75*p_37, 1.25*p_37)})
Default_Variations.update({ind_p_38:(0.75*p_38, 1.25*p_38)})
Default_Variations.update({ind_p_40:(0.75*p_40, 1.25*p_40)})

Default_Variations.update({ind_tg_t4:(0.9*tg_t4, 1.1*tg_t4)})
Default_Variations.update({ind_tg_t3:(0.4*tg_t3, 1.1*tg_t3)})
Default_Variations.update({ind_dose_t4:(70.0/777, 150.0/777)})
Default_Variations.update({ind_dose_t3:(5.0/651, 15.0/651)})

################################################################

def toStr(expr):
	exprString = str(expr).replace('e^', 'exp').replace('^', '**')
	return exprString
	
def getFourierFunction(LOOP = False):
	N = nterms
	# print(p_45, p_43)
	var('tg_t4', 'tg_t3','dose_t4', 'dose_t3', 'eps_t4', 'eps_t3', 'delta_t4', 'delta_t3', 'off_t3', 'off_t4')
		#'p_43',  'p_45', 
	if SIMPLE:
		(fn, f1, f2, a0) = getFourierSimple(N)
		#a0 = eval('a0+off')
	else:
		(fn, f1, f2, a0) = getFourier(N)
		#a0 = eval('a0+off_t3')
	
	#dfn = getFourierDerivative(fn)
	pfr_t4 = fn(t_gap = tg_t4, k_a = p_43, pi=math.pi, off = off_t4, d0 = dose_t4, eps = eps_t4, delta = delta_t4)
	defs_t4 = getFourierDerivative(pfr_t4)	
	#defs2 = dfn(t_gap = tg, k_a = ka, eps = p, pi=math.pi, delta= f, off = 0)
	inita_t4 = a0(t_gap = tg_t4, k_a = p_43, pi=math.pi, off = off_t4, d0 = dose_t4, eps = eps_t4, delta = delta_t4)
	initf_t4 = pfr_t4(tm = 0)
	#initf_t4 = (dose_t4/(p_43*tg_t4))
	
	#print("four ", pfr, "\n derivative: ",toString(defs),"\n der: ", toString(defs2), "\n init: ", toString(inita) , "\n initf: ", toString(initf))
	#print("four ", pfr_t4, "\n derivative: ",toStr(defs_t4),"\n initf: ", toStr(initf_t4))
	
	# print('\n T4 derivative of fourier: \n')
	# print(defs_t4)
	# print(inita_t4, initf_t4)
	
	pfr_t3 = fn(t_gap = tg_t3, k_a = p_45, pi=math.pi, off = off_t3, d0 = dose_t3, eps = eps_t3, delta = delta_t3)
	defs_t3 = getFourierDerivative(pfr_t3)
	#defs2 = dfn(t_gap = tg, k_a = ka, eps = p, pi=math.pi, delta= f, off = 0)
	inita_t3 = a0(t_gap = tg_t3, k_a = p_45, pi=math.pi, off = off_t3, d0 = dose_t3, eps = eps_t3, delta = delta_t3)
	initf_t3 = pfr_t3(tm = 0)
	#initf_t3 = (dose_t3/(p_45*tg_t3))
	#print("four ", pfr, "\n derivative: ",toString(defs),"\n der: ", toString(defs2), "\n init: ", toString(inita) , "\n initf: ", toString(initf))
	#print("four ", pfr_t3, "\n derivative: ",toStr(defs_t3),"\n initf: ", toStr(initf_t3))
	
	# print('\n T3 derivative of fourier: \n')
	# print(defs_t3)
	# print(inita_t3, initf_t3)
	# if LOOP:
	# 	defs_t4, defs_t3 = '-p_43*q_10', '-p_45*q_12'
	return fn, pfr_t4, pfr_t3, defs_t4, defs_t3, initf_t4, initf_t3

fn, pfr_t4, pfr_t3, defs_t4, defs_t3, initf_t4, initf_t3 = getFourierFunction()
if LOOP:
	equation = basic_eqn.format(toStr('-p_43*q_10'), toStr('-p_45*q_12'))	
else:
	equation = basic_eqn.format(toStr(defs_t4), toStr(defs_t3)) #, toStr(initf_t4), toStr(initf_t3))

def getActualDosages(params, hashmap, LOOP = True):
	N = nterms
	values = setParams(params, hashmap)

	p_43 = values[ind_p_43]
	p_45 = values[ind_p_45]

	tg_t4, p_43, dose_t4, p1, f1 = values[ind_tg_t4], values[ind_p_43], values[ind_dose_t4], values[ind_eps_t4], values[ind_delta_t4]
	tg_t3, p_45, dose_t3, p2, f2 = values[ind_tg_t3], values[ind_p_45], values[ind_dose_t3], values[ind_eps_t3], values[ind_delta_t3]
	#(off1, ffs1, p1, f1, tm1) = optimizeSmootherfunction(N, tg_t4, p_43, dose_t4)
	# print(p1, f1)
	#(off2, ffs2, p2, f2, tm2) = optimizeSmootherfunction(N, tg_t3, p_45, dose_t3)
	# print(p2, f2)
	if not LOOP:
		dose_t4_1 = dose_t4#*(1- exp(-p_43*tg_t4))
		dose_t3_1 = dose_t3#*(1- exp(-p_45*tg_t3))

		p_43_1 = p_43 #+ 2*0.693/(7*24)
		p_45_1 = p_45 #+ 0.693/(24)

		(r_t4, res_t4) = getactualValue(N, tg_t4, p_43_1, p1, 1.0, f1, dose_t4_1, SIMPLE)
		(r_t3, res_t3) = getactualValue(N, tg_t3, p_45_1, p2, 1.0, f2, dose_t3_1, SIMPLE)

	else:
		dose_t4_1 = dose_t4
		dose_t3_1 = dose_t3
		
		p_43_1 = p_43
		p_45_1 = p_45

		res_t4 = dose_t4_1
		res_t3 = dose_t3_1

	# (r_t4, res_t4) = getSSDActualValue(N, tg_t4, p_43, p1, 1.0, f1, dose_t4, SIMPLE)
	# (r_t3, res_t3) = getSSDActualValue(N, tg_t3, p_45, p2, 1.0, f2, dose_t3, SIMPLE)

	res = (res_t4, dose_t4, res_t3, dose_t3)
	return res

def getRounded(tg):
	rt1 = tg/24.0
	if rt1 < 0.5:
		rt1 = 1/3.0
	elif rt1 == 0.5:
		rt1 = 1/2.0
	else:
		rt1 = round(rt1)
	tg1 = rt1*24.0
	#print('getRounded', tg1, rt1)
	return tg1

def getPF(tg1):
	if tg1 == 24.0:
		p = 1.2
		f = 1.2957
	elif tg1 == 12.0:
		p = 0.6
		f = 1.201996
	elif tg1 == 8.0:
		p = 0.4
		f = 1.1729
	else:
		p = 1.0/nterms #2*tg_t4/N
		f = 4.0
	return p, f
	
def setOffsetParams(params, dose_selectedIndices, selectedIndices, hashmap, LOOP=True):
	hypo_params = []
	for i in range(len(dose_selectedIndices), len(params)):
		hypo_params.append(params[i])

	#var( 'off_t3', 'off_t4')
	#print('In getOffsetParams')
	N = nterms
	values = setParams(params, hashmap)
	tg_t4, p_43, dose_t4_1 = values[ind_tg_t4], values[ind_p_43], values[ind_dose_t4] #*(1-exp(-p_43*tg_t4))
	tg_t3, p_45, dose_t3_1 = values[ind_tg_t3], values[ind_p_45], values[ind_dose_t3] #*(1-exp(-p_45*tg_t3))
	
	# dose_t4_1 = values[ind_dose_t4]
	# dose_t3_1 = values[ind_dose_t3]
	if not LOOP:
		p_43_1 = p_43 #+ 2*0.693/(7*24)
		p_45_1 = p_45 #+ 0.693/(24)

		dose_t4 = dose_t4_1 #*(1- exp(-p_43_1*tg_t4))
		dose_t3 = dose_t3_1 #*(1- exp(-p_45_1*tg_t3))

		# print((1- exp(-p_43_1*tg_t4)))
		# print('setOffsetParams', p_43, p_43_1, p_45, p_45_1, dose_t4, dose_t4_1, dose_t4*777, dose_t4_1*777)
	else:
		p_43_1 = p_43
		p_45_1 = p_45

		dose_t4 = values[ind_dose_t4]
		dose_t3 = values[ind_dose_t3]
		# dose_t4 = dose_t4_1 #*(1- exp(-p_43_1*tg_t4))
		# dose_t3 = dose_t3_1 #*(1- exp(-p_45_1*tg_t3))

	tg_t4 = getRounded(tg_t4)
	tg_t3 = getRounded(tg_t3)
	
	if not SIMPLE:
		# fn1 = fn(t_gap = tg_t4, k_a = p_43, d0 = dose_t4, pi=math.pi, off = 0, eps = p1, delat = f1)
		# fn2 = fn(t_gap = tg_t3, k_a = p_45, d0 = dose_t3, pi=math.pi, off = 0)
		# print('###########')
		# print(fn1)
		# print('###########')
		#(offset1, ffs1, p1, f1, tm1) = optimizeSmootherfunction(N, tg_t4, p_43, dose_t4, fn1)
		#print(p1, f1)
		#(offset2, ffs2, p2, f2, tm2) = optimizeSmootherfunction(N, tg_t3, p_45, dose_t3, fn2)
		#print(p2, f2)
		
		p1, f1 = getPF(tg_t4)
		p2, f2 = getPF(tg_t3)

		# (fet1, chk1) = getRange(p1, N, tg_t4)
		# (fet2, chk2) = getRange(p2, N, tg_t3)	

		# fn1 = fn(t_gap = tg_t4, k_a = p_43, eps = p1, d0 = dose_t4, pi=math.pi, delta = f1, off = 0)
		# fn2 = fn(t_gap = tg_t3, k_a = p_45, eps = p2, d0 = dose_t3, pi=math.pi, delta = f2, off = 0)
		# (offset1, tm1) = getOffset(fn1, tg_t4 - fet1, chk1)
		# (offset2, tm2) = getOffset(fn2, tg_t3 - fet2, chk2)

	else:		
		p1 = 1.0/N #2*tg_t4/N
		f1 = 4.0
		p2 = 1.0/N #2*tg_t3/N
		f2 = 4.0	

	(fet1, chk1) = getRange(p1, N, tg_t4)
	(fet2, chk2) = getRange(p2, N, tg_t3)	
	fn1 = fn(t_gap = tg_t4, k_a = p_43_1, eps = p1, d0 = dose_t4, pi=math.pi, delta = f1, off = 0)
	fn2 = fn(t_gap = tg_t3, k_a = p_45_1, eps = p2, d0 = dose_t3, pi=math.pi, delta = f2, off = 0)
	
	#print(fn, fn1)
	#sys.stdout.flush()

	(offset1, tm1) = getOffset(fn1, tg_t4 - fet1, chk1)
	(offset2, tm2) = getOffset(fn2, tg_t3 - fet2, chk2)
		
	# stable_time_tosimulate = TD*TU
	# stable_sampHr = 2 #24/5.0
	# stable_samples = int(stable_time_tosimulate*stable_sampHr)

	# tm = np.linspace(0, stable_time_tosimulate, stable_samples)
	# #print(fn1)

	# y = [fn1(tm=i) for i in tm]
	# off1 = -1*np.min(y)
	# print(off1)

	#print('chk: ',p1, p2, chk1, chk2, fet1, fet2, p1 * N/tg_t4, p2*N/tg_t3)
	#sys.stdout.flush()
	#
	#pfr_1 = pfr1(d0 = dose_t4, eps = p1, delta = f1)
	#pfr_2 = pfr2(d0 = dose_t3, eps = p2, delta = f2)
	#defs = dfn(d0 = dval, eps = p, delta = f)
	#fn1 = fn(pi=math.pi, off = 0.0)
	#fn2 = fn(pi=math.pi, off = 0.0)
	offset1 = getOptimizedOffset(N, tg_t4, p_43, p1, f1, dose_t4, fn1)
	offset2 = getOptimizedOffset(N, tg_t3, p_45, p2, f2, dose_t3, fn2)
	#ss_t4 = (dose_t4/(1 - exp(-p_43 * tg_t4)))
	# print('offset: ', offset1, offset2)#, ss_t4)
	# offset1, offset2 = 0.0, 0.0

	#(rmin1, resMin) = getactualValue(N, tg_t4, p_43, p1, 1.0, f1, dose_t4, SIMPLE)	
	#print('actual value', rmin1, resMin)
	init_et4 = initf_t4(tg_t4 = tg_t4, p_43 = p_43_1, t_gap = tg_t4, k_a = p_43, eps_t4 = p1, dose_t4 = dose_t4, pi=math.pi, delta_t4 = f1, off_t4 = offset1)
	init_et3 = initf_t3(tg_t3 = tg_t3, p_45 = p_45_1, t_gap = tg_t3, k_a = p_45, eps_t3 = p2, dose_t3 = dose_t3, pi=math.pi, delta_t3 = f2, off_t3 = offset2)
	#print('Init: ', toStr(init_et4), toStr(init_et3))

	#de_et4 = pfr_t4(tg_t4 = tg_t4, p_43 = p_43, eps_t4 = p1, dose_t4 = dose_t4, pi=math.pi, delta_t4 = f1, off_t4 = offset1)
	#de_et3 = pfr_t3(tg_t3 = tg_t3, p_45 = p_45, eps_t3 = p2, dose_t3 = dose_t3, pi=math.pi, delta_t3 = f2, off_t3 = offset2)
	#print('ode : ', de_et4, de_et3)
	#sys.stdout.flush()
	#return offset1, offset2 
	#return float(toStr(init_et4)), float(toStr(init_et3)), p1, f1, p2, f2 #, float(toStr(de_et4)), float(toStr(de_et3)) 
	# if not LOOP:
	et4_init, et3_init = float(toStr(init_et4)), float(toStr(init_et3)) 
	# else:
		# et4_init, et3_init = 0, 0
	# print('setOffsetParams', tg_t4, tg_t3, dose_t4, dose_t3, et4_init, et3_init, p1, f1, p2, f2)
	# params1 = [tg_t4, tg_t3, dose_t4, dose_t3, et4_init, et3_init, p1, f1, p2, f2] + hypo_params
	#params1 = [dose_t4, dose_t3, et4_init, et3_init, p1, f1, p2, f2] + hypo_params
	# params1 = [tg_t4, tg_t3, dose_t4, dose_t3, et4_init, et3_init]
	new_values = {}
	new_values.update({ind_dose_t4:dose_t4})
	new_values.update({ind_dose_t3:dose_t3})
	new_values.update({ind_et4_init:et4_init})
	new_values.update({ind_et3_init:et3_init})
	new_values.update({ind_delta_t4:p1})
	new_values.update({ind_eps_t4:f1})
	new_values.update({ind_delta_t3:p2})
	new_values.update({ind_eps_t3:f2})
	
	params1 = [0.0 for i in dose_selectedIndices]
	for ind in dose_selectedIndices:
		params1[hashmap[ind]] = new_values[ind]
		
	return params1 + hypo_params

#def getEquations():
	#equations = {}
#	return equation
	# return equation

# hr_scale_num = [ind_kdelay, ind_tau]

# hr_scale_den = [ind_d_1, ind_d_2, ind_d_3, ind_d_4,  ind_k12, ind_k13, ind_k31free, ind_k21free, ind_kT4absorp, ind_k02, ind_VmaxD1fast, ind_VmaxD1slow, ind_VmaxD2slow, ind_k45, ind_k46, ind_k64free, ind_k54free, ind_k3absorb, ind_k05, ind_B0, ind_A0, ind_Amax, ind_VmaxTSH,  ind_k3, ind_KdegT3B, ind_KLAG_HYPO, ind_k4dissolve, ind_k4excrete, ind_k3dissolve, ind_k3excrete]

# mol_scale_num = [ind_VmaxD1fast, ind_VmaxD1slow, ind_VmaxD2slow, ind_KmD1fast, ind_KmD1slow, ind_KmD2slow, ind_K50TSH, ind_VmaxTSH, ind_KLAG, ind_B0, ind_A0, ind_Amax, ind_w, ind_t4p_ss, ind_t3p_ss, ind_t4_init, ind_t4_fast_init, ind_t4_slow_init, ind_t3p_init , ind_t3_fast_init , ind_t3_slow_init, ind_tsh_p_init, ind_t3b_init ,ind_t3b_lag_init , ind_t4_pill_init, ind_t4_gut_init , ind_t3_pill_init, ind_t3_gut_init ,ind_q1_init, ind_q2_init , ind_q3_init , ind_q4_init , ind_q5_init , ind_q6_init]

# mol_scale_den = [ind_S4, ind_B, ind_C, ind_D, ind_S3, ind_b, ind_c, ind_d ]
# Default_Variations.update({ind_dose_t3:(0.0, 0.0)})

def getParam_indices():
	indices = {}
	for i in range(len(ParamNames_all)):
		name = ParamNames_all[i]
		index = AllIndices[i]
		indices.update({name:index})
	return indices

def getParam_all():	
	param_all = {}
	for i in range(len(ParamNames_all)):
		name = ParamNames_all[i]
		index = AllIndices[i]
		param_all.update({index:name})
	return param_all


def findIndices(selectedIndices):
	hashmap = {}
	for ind in AllIndices:
		if ind in selectedIndices:
			i = selectedIndices.index(ind)
			hashmap.update({ind: i})
	return hashmap

def getdefaultParams():
	values = {}
	for i in range(len(AllIndices)):
		val = Default_Values[i]
		ind = AllIndices[i]
		values.update({ind:val})
	return values
	#print(values[ind_gt])

def getdefaultVariations():
	values = {}
	for i in range(len(AllIndices)):
		val = default_var
		ind = AllIndices[i]
		values.update({ind:val})
	return values

def getRanges():
	range_all = {}
	for i in range(len(AllIndices)):
		ind = AllIndices[i]
		val = Default_Values[i]
		var = default_var # [-default_var, default_var]
		range_ind = [val*(1-var/100.0), val*(1+var/100.0)]
		if ind in Default_Variations:
			range_ind = [Default_Variations[ind][0], Default_Variations[ind][1]]		
		range_all.update({ind:range_ind})
	return range_all


def setParams(params, hashmap):
	# print('setParams', params, hashmap)
	values = getdefaultParams()

	# print('before', values.keys())
	for ind in AllIndices:
		if ind in hashmap and len(params) > hashmap[ind]:
			val = params[hashmap[ind]]
			values.update({ind:val})
			#print('updating')
	# print('after', values[ind_ic_0], values[ind_off_t3])
	return values

def getParams(selectedIndices, hashmap, inputs = []):
	params = []
	#print('getParams', inputs)
	values = setParams(inputs, hashmap)
	for ind in selectedIndices:
		params.append(values[ind])
		#print('updated', values[ind])
	#print(values[ind_gt])
	return params
	
def setInits(values, inits):
	# hashmap1 = findIndices(All_IC)
	# values = getdefaultParams()
	if len(inits) == 0:
		inits = Default_Inits 
	for i in range(len(All_IC)):
		ind = All_IC[i]
		val = inits[i]
		values.update({ind:val})
	#print(values[ind_gt])
	return values

# def printParams(scale = 'hr'):
# 	values = getdefaultParams()
# 	params = getParam_all()
# 	print('#################################')
# 	for i in values:
# 		val = values[i]
# 		pre_unit = ''
# 		unit = ''
# 		if i in mol_scale_num:
# 			unit = 'umol'
# 			pre_unit = 'umol'
# 		if i in mol_scale_den:
# 			unit = 'umol^-1'
# 			pre_unit = 'umol^-1'
# 		if scale == 'sec':
# 			if i in hr_scale_num:
# 				values[i] = values[i]*3600
# 				unit += ' sec'
# 				pre_unit += ' hr'
# 			if i in hr_scale_den:
# 				values[i] = values[i]/3600
# 				unit += ' sec^-1'
# 				pre_unit += ' hr^-1'
# 		else:
# 			if i in hr_scale_num:
# 				unit += ' hr'
# 				pre_unit += ' hr'
# 			if i in hr_scale_den:
# 				values[i] = values[i]/3600
# 				unit += ' hr^-1'
# 				pre_unit += ' hr^-1'
# 	# if scale[1] == 
	
# 	# for i in values:
# 		print('{0}: {1:1.2e} {3}, {2:1.2e} {4}'.format(params[i], values[i], val, unit, pre_unit))

# 	print('#################################')

