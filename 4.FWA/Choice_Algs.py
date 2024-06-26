
import numpy as np
from Alg import HHO,HHO_SMA,HHO_SMA_Chaos,HHO_SMA_DE,ADCHA,NSGAII,Random,HHO_Chaos,HHO_DE,Chaos,DE

import random
import pickle
import parse_args
Max_power = 1000 #(mW)

def get_Base_ue_ThroughtpusTable(Power_array,check_conectional,BS_FWA_distance):
    transmit_bandwidth = 4e+8
    # print(NP_three_D_X[1])
    log_arg = 1 + (check_conectional * BS_FWA_distance * (Power_array*0.001))
    if np.any(log_arg) <= 0:
        pass
    else:
        Base_ue_ThroughtpusTable = transmit_bandwidth * np.log2(log_arg)
        return Base_ue_ThroughtpusTable

def get_violation(Base_ue_ThroughtpusTable,SBSlink,FWAlink,overpower):
    #計算違規值
    # SBSlink =  np.maximum( np.abs(1-np.sum(check_conectional,axis=0)) ,0)
    # FWAlink = np.maximum( (1-np.sum(check_conectional,axis=1)) ,0)
    requir_thr = np.maximum( ( Min_Rate - np.sum(Base_ue_ThroughtpusTable,axis=0)), 0)
    # overpower=np.maximum( np.sum(NP_three_D_X[1]*check_conectional,axis=1) - Max_power, 0 )
    constrained_violation = (    
        np.sum(SBSlink*10000) +   
        np.sum(FWAlink*10000) +
        np.sum( requir_thr/ (10**6) )+
        np.sum(overpower*100)
        )
    return requir_thr,overpower,constrained_violation

def find_MaxEE(X,BS_FWA_distance,real_distance):
    
    if BS_FWA_distance is False:
        print(f'BS_FWA_distance is False')
        exit(1)
    base_numbers=BS_FWA_distance.shape[0]
    ue_numbers = BS_FWA_distance.shape[1]
    
    NP_three_D_X=np.reshape(X,(2,base_numbers,ue_numbers))
    
    max_indices = np.argmax(NP_three_D_X[0]/real_distance, axis=0)
    Power_array=NP_three_D_X[1].copy()
    #創造一個二維陣列，用來檢查是否有連線
    check_conectional = np.zeros((base_numbers, ue_numbers), dtype=bool)
    #將最大值的位置設為True
    check_conectional[max_indices, np.arange(ue_numbers)] = True
    overpower=np.maximum( np.sum(NP_three_D_X[1]*check_conectional,axis=1) - Max_power, 0 )
    SBSlink =  np.maximum( np.abs(1-np.sum(check_conectional,axis=0)) ,0)
    FWAlink = np.maximum( (1-np.sum(check_conectional,axis=1)) ,0)
    #check
    for i,over in enumerate(overpower):
        if over > 0:
            Power_array[i] = (Power_array[i]*(Max_power/(over+Max_power))).astype(int)
        # Energy_efficient = np.sum(Base_ue_ThroughtpusTable) / (np.sum((Power_array)*check_conectional) + 10**-10)
        # print(requir_thr)
        # return (False,Energy_efficient,constrained_violation)

    #計算每個基站的傳輸速率
    Base_ue_ThroughtpusTable= get_Base_ue_ThroughtpusTable(Power_array,check_conectional,BS_FWA_distance)

    #計算違規值
    requir_thr,overpower,constrained_violation=get_violation(Base_ue_ThroughtpusTable,SBSlink,FWAlink,overpower)
    #如果有違規值，回傳False
    # print(Base_ue_ThroughtpusTable)
    total_power = np.sum((Power_array*check_conectional))
    if total_power == 0:
        Energy_efficient = 0
    else:
        Energy_efficient = np.sum(Base_ue_ThroughtpusTable /  total_power)
    # print(NP_three_D_X[1]*check_conectional)
    # print('=============')
    # print((Power_array)*check_conectional)
    
    #如果分母為0，回傳False
    if np.sum(NP_three_D_X[1]*check_conectional) == 0:
        return (False,0,constrained_violation)
    
    if constrained_violation>0:
        return (False,Energy_efficient,constrained_violation)
    return (True,Energy_efficient,constrained_violation)

def compare_Best(news,olds):
        '''Compare the best solution with the new solution'''
        if news[0]==True and olds[0] == True and news[1]>olds[1]:
            return -1
        if news[0]==True and olds[0] == False:
            return -1
        if news[0]==False and olds[0] == False and news[2] < olds[2]:
            return -1
        return 1

def compare_Best_bool(news,olds):
        """Compare the best solution with the new solution"""
        if news[0]==True and olds[0] == True and news[1]>olds[1]:
            # print(True)
            return True
        if news[0]==True and olds[0] == False:
            # print(True)
            return True
        if news[0]==False and olds[0] == False and news[2] < olds[2]:
            # print(True)
            return True
        # print(False)
        return False
    

def run1(index=0, replaceCommandStr=''):
    # 這裡要改成讀取參數# Declare Min_Rate as a global variable
    global Min_Rate

    # Parse the command line arguments
    if replaceCommandStr == '':
        arges = parse_args.argsParser()
    else:
        arges = parse_args.argsParser(replaceCommandStr)
    # If the unit is Kbps and GoalThr is specified, calculate Min_Rate in bps
    if arges.unit == 'Kbps' and arges.GoalThr:
        Min_Rate = arges.GoalThr * (10**7)
    # If the unit is Mbps and GoalThr is specified, calculate Min_Rate in bps
    elif arges.unit == 'Mbps' and arges.GoalThr:
        Min_Rate = arges.GoalThr * (10**8)
    # If the unit is Gbps and GoalThr is specified, calculate Min_Rate in bps
    elif arges.unit == 'Gbps' and arges.GoalThr:
        Min_Rate = arges.GoalThr * (10**9)
    # If the unit is not recognized or GoalThr is not specified, print an error message and exit
    else:
        print("Unknown unit or number (Minmun Rate)")
        exit(1)
   
    # Set the random seed
    np.random.seed(000)
    random.seed(000)
    # Load the data
    with open(f'{arges.NumberOfBS}BS_{arges.NumberOfFWA}FWA_coords.pkl','rb') as file:
        loadad_data = pickle.load(file)
    # Load the data
    BS_FWA_distance=loadad_data['FWA_BS_distance']
    # Get the number of base stations and user equipments
    base_numbers=BS_FWA_distance.shape[0]
    ue_numbers=BS_FWA_distance.shape[1]
    # Set the parameters for the HHO algorithm
    function=find_MaxEE
    dimension=2*ue_numbers*base_numbers
    iteration=arges.iteration
    problem_size=100
    lb=0
    ub=1000
    compare_func = compare_Best
    Start_run_times=arges.Start_time
    End_run_times=arges.End_time
    # Set the random seed list
    random_seed_list = random.sample(range(1, 200000), 100000)
    Mu = 4
    CR =0.3
    # Antenna_gain is the gain of the antenna in dBi (decibels relative to isotropic)
    Antenna_gain=24 
    # LOS_shadow is the shadowing effect in Line-Of-Sight (LOS) propagation in dBi
    LOS_shadow = 4 
    # NLOS_shadow is the shadowing effect in Non-Line-Of-Sight (NLOS) propagation in dBi
    NLOS_shadow = 18 
    # transmit_bandwidth is the bandwidth over which the signal is transmitted, in Hz
    transmit_bandwidth = 4e+8
    # carrier_frequency is the frequency of the carrier signal, in GHz
    carrier_frequency = 28 
    # N0 is the noise power spectral density in dBm (decibels relative to 1 milliwatt)
    N0=-87.81

    # HHO_DE = HHO_DE_Chaos(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool,Mu,CR)
    # HHO_DE.save_fitness_params(BS_FWA_distance,carrier_frequency,Antenna_gain,LOS_shadow,NLOS_shadow,N0,transmit_bandwidth)
    # return HHO_DE.request_run(index,random_seed_list)
    #執行HHO演算法
    type = arges.type
    #選擇演算法
    
    if type == 'HHO':
        Alg1 = HHO(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool)
    elif type == 'HHO_SMA':
        Alg1 = HHO_SMA(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool)
    elif type == 'ADCHA':
        Alg1 = ADCHA(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool,Mu,CR)
    elif type == 'NSGA':
        parents_portion = 0.3
        mutation_prob = 0.0125
        crossover_prob = 0.9
        Alg1 = NSGAII(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool,parents_portion,mutation_prob,crossover_prob)
    elif type == 'Random':
        Alg1 = Random(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool)
    elif type == 'HHO_Chaos':
        Alg1 = HHO_Chaos(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool,Mu,CR)
    elif type == 'HHO_DE':
        Alg1 = HHO_DE(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool,Mu,CR)
    elif type == 'Chaos':
        Alg1 = Chaos(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool,Mu,CR)
    elif type == 'DE':
        Alg1 = DE(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool,Mu,CR)
    else:
        print('Unknown type')
        exit(1)
    
    Alg1.save_fitness_params(BS_FWA_distance,carrier_frequency,Antenna_gain,LOS_shadow
        ,NLOS_shadow,N0,transmit_bandwidth)
    # Alg1.init_pop()
    Alg1.mutil_run(Start_run_times,End_run_times,random_seed_list,f'{arges.NumberOfBS}BS_{arges.NumberOfFWA}FWA_GoalThr{arges.GoalThr}{arges.unit}')
 
    # return Alg1.request_run(index,random_seed_list)

if __name__ == '__main__':
    run1()
    # run1(replaceCommandStr='-type ADCHA -BS 5 -FWA 50 -Thr 1 -unit Gbps -iter 5 -start 0 -end 1')

    
    
