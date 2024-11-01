#!/bin/bash
#PBS -N onemax_mp
#PBS -V
CUR_DIR=/dose_est
# User Directives
cd $CUR_DIR
pwd

# ODE model of the pathway (formatted for dReach tool)
fname='eisen_1s_3p_fix_tsc_et4macro.drh'

# observedfile with estimated hypothyroid parameters
ff='eisen_sskm_opt_data_07423'

# date-time stamp for generating logs and results
k=110

# maximum parameter variation (0.05 ~ 5%)
MHPV=0.05

# precision parameter for dReach tool 
dd=0.001

j=1
declare -a arr1=("1" "2" "6" "7" "10" "13" "14" "16" "17" "19" "8" "12" "18" "20" "11" "15")

RAT="1.0"
for i in "${arr1[@]}"
do
	echo 'starting row $i'
	#for RAT in "${ratio[@]}"
	#do
		echo 'starting ratio ${RAT}'
		cat $CUR_DIR/estimation/get_dose_bin_fourier_mp_et4_macro.py | 
			sed "s/index *= *[0-9][0-9]*/index = $i /" |
			sed "s/Rows *= *[0-9][0-9]*/Rows = $j /" |
			sed "s/LOOP *= *[0-9][0-9]*/LOOP = 0 /" |
			sed "s/MS *= *[0-9][0-9]*/MS = 0 /" |
			sed "s/SCALE *= *[0-9][0-9]*/SCALE = 120 /" |
			sed "s/FIX *= *[0-9][0-9]*/FIX = 0 /" |
			sed "s/RATIO *= *[0-9][.][0-9][0-9]*/RATIO = ${RAT} /" |
			sed "s/TF *= *[0-9][0-9]*/TF = ${k} /" |
			sed "s/IND *= *[0-9][0-9]*/IND = 0 /" |
			sed "s/EXTRA *= *[0-9][0-9]*/EXTRA = 0 /" |
			sed "s/MAX_HPV *= *[0-9][.][0-9][0-9]*/MAX_HPV = ${MHPV} /" |
			sed "s/MIN_DELTA *= *[0-9][.][0-9][0-9]*/MIN_DELTA = ${dd} /" |
			sed "s/import simulateHA_1 as sim*/import simulateHA_${i} as sim /" > $CUR_DIR/estimation/get_hpr_nd_1s_mp${i}.py 
		sage --python $CUR_DIR/estimation/get_hpr_nd_1s_mp${i}.py -i eisen/model_hp/${fname} -o eisen/model_hp/eisen_model_Out.txt -d eisen/model_hp/eisen_model_1state_dose_der_tscale.csv -s eisen/model_hp/${ff}.csv > logs/eis_log_hpr_nd_1s_mp_${k}_${i}_${RAT}.txt 

		wait
	#done
done

declare -a arr=("3" "4" "5" "9") #, "8" "11" "12" "15" "18" "20")

RAT=0.01

echo 'starting ratio ${RAT}'
for i in "${arr[@]}"
do
	echo 'starting row $i'
	#for RAT in "${ratio[@]}"
	#do
		cat $CUR_DIR/estimation/get_dose_bin_fourier_mp_et4_macro.py | 
			sed "s/index *= *[0-9][0-9]*/index = $i /" |
			sed "s/Rows *= *[0-9][0-9]*/Rows = $j /" |
			sed "s/LOOP *= *[0-9][0-9]*/LOOP = 0 /" |
			sed "s/MS *= *[0-9][0-9]*/MS = 0 /" |
			sed "s/SCALE *= *[0-9][0-9]*/SCALE = 120 /" |
			sed "s/FIX *= *[0-9][0-9]*/FIX = 1 /" |
			sed "s/RATIO *= *[0-9][.][0-9][0-9]*/RATIO = ${RAT} /" |
			sed "s/TF *= *[0-9][0-9]*/TF = ${k} /" |
			sed "s/IND *= *[0-9][0-9]*/IND = 0 /" |
			sed "s/EXTRA *= *[0-9][0-9]*/EXTRA = 0 /" |
			sed "s/MAX_HPV *= *[0-9][.][0-9][0-9]*/MAX_HPV = ${MHPV} /" |
			sed "s/MIN_DELTA *= *[0-9][.][0-9][0-9]*/MIN_DELTA = ${dd} /" |
			sed "s/import simulateHA_1 as sim*/import simulateHA_${i} as sim /" > $CUR_DIR/estimation/get_hpr_nd_1s_mp${i}.py 
		sage --python $CUR_DIR/estimation/get_hpr_nd_1s_mp${i}.py -i eisen/model_hp/${fname} -o eisen/model_hp/eisen_model_Out.txt -d eisen/model_hp/eisen_model_1state_dose_der_tscale.csv -s eisen/model_hp/${ff}.csv > logs/eis_log_hpr_nd_1s_mp_${k}_${i}_${RAT}.txt 

		wait
	#done
done

RAT=0.25

echo 'starting ratio ${RAT}'
for i in "${arr[@]}"
do
	echo 'starting row $i'
	#for RAT in "${ratio[@]}"
	#do
		cat $CUR_DIR/estimation/get_dose_bin_fourier_mp_et4_macro.py | 
			sed "s/index *= *[0-9][0-9]*/index = $i /" |
			sed "s/Rows *= *[0-9][0-9]*/Rows = $j /" |
			sed "s/LOOP *= *[0-9][0-9]*/LOOP = 0 /" |
			sed "s/MS *= *[0-9][0-9]*/MS = 0 /" |
			sed "s/SCALE *= *[0-9][0-9]*/SCALE = 120 /" |
			sed "s/FIX *= *[0-9][0-9]*/FIX = 1 /" |
			sed "s/RATIO *= *[0-9][.][0-9][0-9]*/RATIO = ${RAT} /" |
			sed "s/TF *= *[0-9][0-9]*/TF = ${k} /" |
			sed "s/IND *= *[0-9][0-9]*/IND = 0 /" |
			sed "s/EXTRA *= *[0-9][0-9]*/EXTRA = 0 /" |
			sed "s/MAX_HPV *= *[0-9][.][0-9][0-9]*/MAX_HPV = ${MHPV} /" |
			sed "s/MIN_DELTA *= *[0-9][.][0-9][0-9]*/MIN_DELTA = ${dd} /" |
			sed "s/import simulateHA_1 as sim*/import simulateHA_${i} as sim /" > $CUR_DIR/estimation/get_hpr_nd_1s_mp${i}.py 
		sage --python $CUR_DIR/estimation/get_hpr_nd_1s_mp${i}.py -i eisen/model_hp/${fname} -o eisen/model_hp/eisen_model_Out.txt -d eisen/model_hp/eisen_model_1state_dose_der_tscale.csv -s eisen/model_hp/${ff}.csv > logs/eis_log_hpr_nd_1s_mp_${k}_${i}_${RAT}.txt 

		wait
	#done
done

RAT=0.5

echo 'starting ratio ${RAT}'
for i in "${arr[@]}"
do
	echo 'starting row $i'
	#for RAT in "${ratio[@]}"
	#do
		cat $CUR_DIR/estimation/get_dose_bin_fourier_mp_et4_macro.py | 
			sed "s/index *= *[0-9][0-9]*/index = $i /" |
			sed "s/Rows *= *[0-9][0-9]*/Rows = $j /" |
			sed "s/LOOP *= *[0-9][0-9]*/LOOP = 0 /" |
			sed "s/MS *= *[0-9][0-9]*/MS = 0 /" |
			sed "s/SCALE *= *[0-9][0-9]*/SCALE = 120 /" |
			sed "s/FIX *= *[0-9][0-9]*/FIX = 1 /" |
			sed "s/RATIO *= *[0-9][.][0-9][0-9]*/RATIO = ${RAT} /" |
			sed "s/TF *= *[0-9][0-9]*/TF = ${k} /" |
			sed "s/IND *= *[0-9][0-9]*/IND = 0 /" |
			sed "s/EXTRA *= *[0-9][0-9]*/EXTRA = 0 /" |
			sed "s/MAX_HPV *= *[0-9][.][0-9][0-9]*/MAX_HPV = ${MHPV} /" |
			sed "s/MIN_DELTA *= *[0-9][.][0-9][0-9]*/MIN_DELTA = ${dd} /" |
			sed "s/import simulateHA_1 as sim*/import simulateHA_${i} as sim /" > $CUR_DIR/estimation/get_hpr_nd_1s_mp${i}.py 
		sage --python $CUR_DIR/estimation/get_hpr_nd_1s_mp${i}.py -i eisen/model_hp/${fname} -o eisen/model_hp/eisen_model_Out.txt -d eisen/model_hp/eisen_model_1state_dose_der_tscale.csv -s eisen/model_hp/${ff}.csv > logs/eis_log_hpr_nd_1s_mp_${k}_${i}_${RAT}.txt 

		wait
	#done
done

RAT=0.75

echo 'starting ratio ${RAT}'
for i in "${arr[@]}"
do
	echo 'starting row $i'
	#for RAT in "${ratio[@]}"
	#do
		cat $CUR_DIR/estimation/get_dose_bin_fourier_mp_et4_macro.py | 
			sed "s/index *= *[0-9][0-9]*/index = $i /" |
			sed "s/Rows *= *[0-9][0-9]*/Rows = $j /" |
			sed "s/LOOP *= *[0-9][0-9]*/LOOP = 0 /" |
			sed "s/MS *= *[0-9][0-9]*/MS = 0 /" |
			sed "s/SCALE *= *[0-9][0-9]*/SCALE = 120 /" |
			sed "s/FIX *= *[0-9][0-9]*/FIX = 1 /" |
			sed "s/RATIO *= *[0-9][.][0-9][0-9]*/RATIO = ${RAT} /" |
			sed "s/TF *= *[0-9][0-9]*/TF = ${k} /" |
			sed "s/IND *= *[0-9][0-9]*/IND = 0 /" |
			sed "s/EXTRA *= *[0-9][0-9]*/EXTRA = 0 /" |
			sed "s/MAX_HPV *= *[0-9][.][0-9][0-9]*/MAX_HPV = ${MHPV} /" |
			sed "s/MIN_DELTA *= *[0-9][.][0-9][0-9]*/MIN_DELTA = ${dd} /" |
			sed "s/import simulateHA_1 as sim*/import simulateHA_${i} as sim /" > $CUR_DIR/estimation/get_hpr_nd_1s_mp${i}.py 
		sage --python $CUR_DIR/estimation/get_hpr_nd_1s_mp${i}.py -i eisen/model_hp/${fname} -o eisen/model_hp/eisen_model_Out.txt -d eisen/model_hp/eisen_model_1state_dose_der_tscale.csv -s eisen/model_hp/${ff}.csv > logs/eis_log_hpr_nd_1s_mp_${k}_${i}_${RAT}.txt 

		wait
	#done
done

RAT=0.99

echo 'starting ratio ${RAT}'
for i in "${arr[@]}"
do
	echo 'starting row $i'
	#for RAT in "${ratio[@]}"
	#do
		cat $CUR_DIR/estimation/get_dose_bin_fourier_mp_et4_macro.py | 
			sed "s/index *= *[0-9][0-9]*/index = $i /" |
			sed "s/Rows *= *[0-9][0-9]*/Rows = $j /" |
			sed "s/LOOP *= *[0-9][0-9]*/LOOP = 0 /" |
			sed "s/MS *= *[0-9][0-9]*/MS = 0 /" |
			sed "s/SCALE *= *[0-9][0-9]*/SCALE = 120 /" |
			sed "s/FIX *= *[0-9][0-9]*/FIX = 1 /" |
			sed "s/RATIO *= *[0-9][.][0-9][0-9]*/RATIO = ${RAT} /" |
			sed "s/TF *= *[0-9][0-9]*/TF = ${k} /" |
			sed "s/IND *= *[0-9][0-9]*/IND = 0 /" |
			sed "s/EXTRA *= *[0-9][0-9]*/EXTRA = 0 /" |
			sed "s/MAX_HPV *= *[0-9][.][0-9][0-9]*/MAX_HPV = ${MHPV} /" |
			sed "s/MIN_DELTA *= *[0-9][.][0-9][0-9]*/MIN_DELTA = ${dd} /" |
			sed "s/import simulateHA_1 as sim*/import simulateHA_${i} as sim /" > $CUR_DIR/estimation/get_hpr_nd_1s_mp${i}.py 
		sage --python $CUR_DIR/estimation/get_hpr_nd_1s_mp${i}.py -i eisen/model_hp/${fname} -o eisen/model_hp/eisen_model_Out.txt -d eisen/model_hp/eisen_model_1state_dose_der_tscale.csv -s eisen/model_hp/${ff}.csv > logs/eis_log_hpr_nd_1s_mp_${k}_${i}_${RAT}.txt 

		wait
	#done
done


echo 'script ended'