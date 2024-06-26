import matplotlib.pyplot as plt
import scienceplots
import pickle
import numpy as np
import pandas as pd

def load_csv(Folder_Path):
    convergence=np.loadtxt(f'{Folder_Path}/convergence.csv')
    constrained_violation_curve=np.loadtxt(f'{Folder_Path}/constrained_violation_curve.csv')
    bestIndividual=np.loadtxt(f'{Folder_Path}/bestIndividual.csv')
    return convergence,constrained_violation_curve,bestIndividual

def z_score_normalize(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    normalized_data = (data - mean) / std_dev
    return normalized_data

if __name__ == '__main__':
    # FWAs = [ '2.0Gbps' , '2.5Gbps' , '3.0Gbps']
    # FWAs = ['1.0Gbps']
    # FWAs = ['2.0Gbps']
    # FWAs = [10,20,30,40,50]
    FWAs = ['4.0Mbps' ,'6.0Mbps' , '8.0Mbps' , '10.0Mbps' , '12.0Mbps']
    
    iteration =5000
    for FWA in FWAs:
        #my_list = list(range(85)) + list(range(100, 144))+list(range(150, 291)) +list(range(300, 343)) +list(range(350, 499))
        StartTime = 0
        EndTime = 500
        current_date='2024-04-26'
        # Alg_list =  ['ADCHA']
        Alg_list =  ['ADCHA','HHO','HHO_SMA','NSGAII']
        # Alg_list =  ['ADCHA','HHO','DE','Chaos','HHO_DE','HHO_Chaos']
        # Alg_list =  ['ADCHA','ADCHA_or','HHO_Chaos','HHO_DE','HHO_DE_new','HHO_DE_or']
        # Deal with simulation results to list of results
        scenes = f'5BS_30FWA_GoalThr{FWA}'
        coord = '5BS_30FWA_coords'
        # scenes = f'5BS_{FWA}FWA_GoalThr6.0Mbps'
        # coord = f'5BS_{FWA}FWA_coords'
        ## Read coordinates data for each base station and Fixed access points
        Init_Path = f'./output/{scenes}/{current_date}'
        with open(f'./{coord}.pkl','rb') as file:
            loadad_data = pickle.load(file)
        
        ## Save BS and FWA between distances and amount
        BS_FWA_distance=loadad_data['FWA_BS_distance']
        base_numbers=BS_FWA_distance.shape[0]
        ue_numbers=BS_FWA_distance.shape[1]

        # Alg_list =  ['Propose HHO','HHO','SMA','HHO_SMA','GA']
        
        ## set All data in list 
        
        ## record final energy efficiency results
        # final_EE = []

        ## record energy efficiency in difference
        

        ## record final violation value
        # final_violation = []

        ## record final violation value in difference
        

        ## record difference Algorithm Iteration result (violation value)
        Diffs_EE = {}

        ## record difference Algorithm Iteration result (energy efficiency)
        Diffs_violation = {}

        

        ## record  each time final EE by difference Algorithm
        Diffs_each_final_EE = {}

        Diffs_each_final_violation = {}

        for index,Alg in enumerate(Alg_list):
            ## record  each time final EE
            
            each_final_EE = []
            each_final_violation = []
            iteration_EE = []
            iteration_violation = []
            print(Alg)
            # print(my_list)
            for Time in range(StartTime,EndTime):
                if Time == StartTime:
                    avg_convergence,avg_constrained_violation_curve,bestIndividual=load_csv(f'{Init_Path}/{Alg}/{Time}')
                    avg_convergence = avg_convergence[:iteration]
                    avg_constrained_violation_curve = avg_constrained_violation_curve[:iteration]
                    iteration_EE = avg_convergence
                    iteration_violation = avg_constrained_violation_curve
                    each_final_EE.append(avg_convergence[-1])
                    each_final_violation.append(avg_constrained_violation_curve[-1])
                    if avg_constrained_violation_curve[-1] < 0:
                        print(f'{Alg}:{avg_constrained_violation_curve[-1]}')
                        print('<0')
                        exit(1)
                else:
                    avg_convergence,avg_constrained_violation_curve,bestIndividual=load_csv(f'{Init_Path}/{Alg}/{Time}')
                    avg_convergence = avg_convergence[:iteration]
                    avg_constrained_violation_curve = avg_constrained_violation_curve[:iteration]
                    iteration_EE = (iteration_EE + avg_convergence)/2
                    iteration_violation = (iteration_violation + avg_constrained_violation_curve)/2
                    each_final_EE.append(avg_convergence[-1])
                    each_final_violation.append(avg_constrained_violation_curve[-1])
                    if avg_constrained_violation_curve[-1] < 0:
                        print(f'{Alg}:{avg_constrained_violation_curve[-1]}')
                        print('<0')
                        exit(1)

            Diffs_EE[Alg] = iteration_EE
            # print(f'{Alg}5000:{Diffs_EE[Alg][4999]}')
            # print(f'{Alg}5000:{Diffs_EE[Alg][49999]}')
            Diffs_violation[Alg] = iteration_violation
            Diffs_each_final_EE[Alg] = each_final_EE
            Diffs_each_final_violation[Alg] = each_final_violation


        #####################ploting the results############################
        plt.style.use(['science','ieee','no-latex'])
        #### ploting energy efficiency of iteration
        fig_iteration_EE,ax_iteration_EE = plt.subplots(figsize=(8,4))
        #### ploting energy efficiency of final
        fig_final_EE,ax_final_EE = plt.subplots(figsize=(8,4))
        final_Alg_list_EE = []
        #### ploting violation value of iteration
        fig_iteration_violation,ax_iteration_violation = plt.subplots(figsize=(8,4))
        #### ploting violation value of final
        fig_final_violation,ax_final_violation = plt.subplots(figsize=(8,4))
        final_Alg_list_violation = []

        #### ploting energy efficency of final (error bars)
        final_Alg_list_EE_errorbar = []
        final_Alg_list_violation_errorbar = []
        

        # colors = ['k','r','b','g','m']
        # line_sltyes = ['-',':','--','-.',((0,(1,3)))]

        colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']
        line_sltyes = ['-', ':', '--', '-.', ((0, (1, 3))), (0, (5, 10)), (0, (3, 1, 1, 1))]
        for index,Alg in enumerate(Alg_list):
            ax_iteration_EE.plot(Diffs_EE[Alg],label=f'{Alg}',color=colors[index],linestyle=line_sltyes[index])
            ax_iteration_violation.plot(Diffs_violation[Alg],label=f'{Alg}',
                                        color=colors[index],linestyle=line_sltyes[index])
            final_Alg_list_EE.append(Diffs_EE[Alg][-1])
            final_Alg_list_violation.append(Diffs_violation[Alg][-1])
            final_Alg_list_EE_errorbar.append(np.std(np.array(Diffs_each_final_EE[Alg])))
            final_Alg_list_violation_errorbar.append(np.std(np.array(Diffs_each_final_violation[Alg])))
            # print(f'{Alg} {int(np.std(np.array(Diffs_each_final_EE[Alg])))}')
            print(f'{Alg} vio-std:{np.std(np.array(Diffs_each_final_violation[Alg]))}')
            print(f'{Alg} mean:{np.mean(np.array(Diffs_each_final_EE[Alg]))}')
            print(f'{Alg} std:{np.std(np.array(Diffs_each_final_EE[Alg]))}')
        


        
        # #### ploting energy efficiency of final
        # ax_final_EE.bar([i*10 for i in range(len(final_Alg_list_EE))],final_Alg_list_EE,yerr=final_Alg_list_EE_errorbar,color='none',edgecolor='black',tick_label=Alg_list,width=2.4)
        # ax_final_EE.set_ylabel('Energy efficiency (bit/s)/mW',fontsize=15)
        # ax_final_EE.grid()
        # fig_final_EE.savefig(f'{Init_Path}/fig_finial_EE.jpg')

        # #### ploting energy efficiency of iteration
        # ax_iteration_EE.set_ylabel('Energy efficiency (bit/s)/mW',fontsize=15)
        # ax_iteration_EE.set_xlabel('Iteration',fontsize=15)
        # fig_iteration_EE.legend(loc='upper center', bbox_to_anchor=(1, 1), framealpha=0.5)
        # fig_iteration_EE.savefig(f'{Init_Path}/fig_iteration_EE.jpg')

        ## ploting violation value  of final
        ax_final_violation.bar([i*10 for i in range(len(final_Alg_list_violation))],final_Alg_list_violation,yerr=final_Alg_list_violation_errorbar,color='none',edgecolor='black',tick_label=Alg_list,width=2.4)
        ax_final_violation.set_ylabel('Violation value',fontsize=15)
        ax_final_violation.set_ylim(bottom=0)
        fig_final_violation.savefig(f'{Init_Path}/fig_finial_violation.jpg')


        #### ploting violation value of iteration
        ax_iteration_violation.set_ylabel('Violation value',fontsize=15)
        ax_iteration_violation.set_xlabel('Iteration',fontsize=15)
        
        fig_iteration_violation.legend(loc='upper center', bbox_to_anchor=(1,1), framealpha=0.5)
        fig_iteration_violation.savefig(f'{Init_Path}/fig_iteration_violation.jpg')
        print(Init_Path)
        print(f'ploting finish {FWA}FWA')


        

        # 计算相对差异百分比
        relative_diff = {}
        for alg1 in Alg_list:
            relative_diff[alg1] = {}
            for alg2 in Alg_list:
                if alg1 != alg2:
                    diff_percentage = ((final_Alg_list_EE[Alg_list.index(alg1)] - final_Alg_list_EE[Alg_list.index(alg2)]) / final_Alg_list_EE[Alg_list.index(alg2)]) * 100
                    relative_diff[alg1][alg2] = f"{diff_percentage:.2f}%"
                else:
                    relative_diff[alg1][alg2] = "N/A"
        print(FWA)
        print('===============================')
        print(relative_diff)

        # 创建表格
        df_relative_diff = pd.DataFrame(relative_diff)

        # 保存表格为 CSV 文件
        df_relative_diff.to_csv(f'{Init_Path}/relative_diff.csv', index=True)
