import matplotlib.pyplot as plt
import scienceplots
import pickle
import numpy as np

def load_csv(Folder_Path):
    convergence=np.loadtxt(f'{Folder_Path}/convergence.csv')
    constrained_violation_curve=np.loadtxt(f'{Folder_Path}/constrained_violation_curve.csv')
    bestIndividual=np.loadtxt(f'{Folder_Path}/bestIndividual.csv')
    return convergence,constrained_violation_curve,bestIndividual

def load_csv_excution_times(Folder_Path):
    ExecutionTime=np.loadtxt(f'{Folder_Path}/ExecutionTime.csv')
    return ExecutionTime


if __name__ == '__main__':
    Alg_list =  ['ADCHA']
    for Alg in Alg_list:
        # not_use_case = [(10,'6.0M'),(10,'2.0G'),(10,'2.5G'),(10,'3.0G')\
        #                 ,(30,'6.0M'),(30,'2.0G'),(30,'2.5G'),(30,'3.0G')\
        #                 ,(40,'6.0M'),(40,'2.0G'),(40,'2.5G'),(40,'3.0G')\
        #                 ,(50,'6.0M'),(50,'2.0G'),(50,'2.5G'),(50,'3.0G')]
        not_use_case = []
        StartTime = 0
        EndTime = 500
        # my_list = list(range(200)) + list(range(300, 359))
        # Alg = 'ADCHA'
        current_date='2024-03-20'
        ## Deal with simulation results to list of results
        FWAs = [10,20,30,40,50]
        # FWAs=[30]
        requirements_throughput = ['6.0M']
        All_results ={}
        All_results_vio ={}
        excution_times={}
        ## Read coordinates data for each base station and Fixed access points
        for FWA in FWAs:
            for thr in requirements_throughput:
                buf_EE = []
                buf_vio = []
                if (FWA,thr) not in not_use_case:
                    Init_Path = f'./output/5BS_{FWA}FWA_GoalThr{thr}bps/{current_date}'
                    print(Init_Path)
                    with open(f'./5BS_{FWA}FWA_coords.pkl','rb') as file:
                        loadad_data = pickle.load(file)
                    for Time in range(StartTime,EndTime):
                        if Time == StartTime:
                            avg_convergence,avg_constrained_violation_curve,bestIndividual=load_csv(f'{Init_Path}/{Alg}/{Time}')
                            buf_EE.append(avg_convergence[-1])
                            buf_vio.append(avg_constrained_violation_curve[-1])
                        else:
                            avg_convergence,avg_constrained_violation_curve,bestIndividual=load_csv(f'{Init_Path}/{Alg}/{Time}')
                            buf_EE.append(avg_convergence[-1])
                            buf_vio.append(avg_constrained_violation_curve[-1])
                    excution_times[(FWA,thr)]=load_csv_excution_times(f'{Init_Path}/{Alg}')
                    print(excution_times[(FWA,thr)])
                    All_results[(FWA,thr)] = buf_EE
                    All_results_vio[(FWA,thr)] = buf_vio
                    

        #####################ploting the results############################
        plt.style.use(['science','ieee','no-latex'])
        #### ploting FWA
        fig_FWA_EE,ax_FWA_EE = plt.subplots(figsize=(8,4))



        #### ploting throughput
        fig_thr_EE,ax_thr_EE = plt.subplots(figsize=(8,4))
        FWA_mean_list = []
        FWA_std_list = []
        FWA_list = []
        FWA_excution_times = []
        for FWA in FWAs:
            FWA_std_list.append(np.mean(np.array(All_results[(FWA,'6.0M')])))
            FWA_mean_list.append(np.mean(np.array(All_results[(FWA,'6.0M')])))
            FWA_list.append(All_results[(FWA,'6.0M')])
            
        # print(FWA_mean_list)
        ax_FWA_EE.boxplot(FWA_list,labels=FWAs,showfliers=False)
        # ax_FWA_EE.bar([i*10 for i in range(len(FWA_mean_list))],FWA_mean_list,yerr=FWA_std_list,color='none',edgecolor='black',tick_label=FWAs,width=2.4)
        ax_FWA_EE.set_ylabel('Energy efficiency (bit/s)/mW',fontsize=15)
        ax_FWA_EE.grid()
        fig_FWA_EE.savefig(f'./{Alg}_compare_diff_FWA.jpg')

        

        thr_mean_list = []
        thr_std_list = []
        thr_list = []
        for thr in requirements_throughput:
            thr_std_list.append(np.mean(np.array(All_results[(30,thr)])))
            thr_mean_list.append(np.mean(np.array(All_results[(30,thr)])))
            thr_list.append(All_results[(30,thr)])
            FWA_excution_times.append(np.sum(np.array(excution_times[(30,thr)])))
        # print(thr_mean_list)
        # labels=requirements_throughput

        ax_thr_EE.boxplot(thr_list,labels=requirements_throughput,showfliers=False)
        # ax_thr_EE.bar([i*10 for i in range(len(thr_mean_list))],thr_mean_list,yerr=thr_std_list,color='none',edgecolor='black',tick_label=requirements_throughput,width=2.4)
        ax_thr_EE.set_ylabel('Energy efficiency (bit/s)/mW',fontsize=15)
        ax_thr_EE.grid()
        fig_thr_EE.savefig(f'./{Alg}_compare_diff_thr.jpg')

        fig_FWA_excution_times,ax_FWA_excution_times = plt.subplots(figsize=(8,4))
        ax_FWA_excution_times.bar(requirements_throughput,FWA_excution_times)
        ax_FWA_excution_times.set_ylabel('Execution time (s)',fontsize=15)
        ax_FWA_excution_times.grid()
        fig_FWA_excution_times.savefig(f'./{Alg}_compare_diff_FWA_excution_time.jpg')
