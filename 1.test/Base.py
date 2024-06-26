import abc
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import shutil
import csv

class Root(abc.ABC):
    """ This is root of all Algorithms """

    ID_MIN_PROB = 0         # min problem
    ID_MAX_PROB = -1        # max problem

    ID_POS = 0              # Position
    ID_FIT = 1              # Fitness
    ID_COMPARE = 3
    EPSILON = 10E-10

    def __init__(self,function,dimension,iteration,problem_size,lb,ub):
        """
        Parameters
        ----------
        obj_func : function
        lb : list
        ub : list
        problem_size : int, optional
        batch_size: int, optional
        verbose : bool, optional
        """
        # assert (type(variable_type_mixed).__module__=='numpy'),\
        #     "\n variable_type must be numpy array"
        
        #############################################################
        # check all parameters
        assert(type(problem_size) is int),"Problem size must be an int number"
        assert(problem_size > 0),"Problem size must > 0"
        assert((ub is not None) and (lb is not None)),"You must seeting your low bound and upper bound."
        assert(dimension > 0),"dimension must > 0"
        assert(function),"pls setting your obj_fuction"
        assert(iteration > 0),"problem_size must > 0"
        #############################################################


        #############################################################
        # update all parameters to self
        self.dimension = dimension
        self.problem_size = problem_size ## SearchAgents_no
        self.iteration = iteration
        self.obj_func = function
        self.ub =ub
        self.lb =lb
        self.init_set=False
        #############################################################
        
        self.init_params()

    def save_fitness_params(self, BS_FWA_distance,carrier_frequency,Antenna_gain,LOS_shadow,NLOS_shadow,N0,transmit_bandwidth):
        self.BS_FWA_distance = BS_FWA_distance
        BS_FWA_distance = BS_FWA_distance
        BS_FWA_LOS=32.4+21*np.log10(BS_FWA_distance)+20*np.log10(carrier_frequency) #
        BS_FWA_NLOS=32.4+31.9*np.log10(BS_FWA_distance)+20*np.log10(carrier_frequency)

        BS_FWA_Gain_LOS_shdow = Antenna_gain - BS_FWA_LOS - LOS_shadow
        BS_FWA_Gain_NLOS_shdow = Antenna_gain - BS_FWA_NLOS - NLOS_shadow

        BS_FWA_Gain_LOS_shdow_multi =10**((BS_FWA_Gain_LOS_shdow-30)/10) #9 becuz noise power is to small that can't use in float

        BS_FWA_Gain_NLOS_shdow_multi =10**((BS_FWA_Gain_NLOS_shdow-30)/10) #9 becuz noise power is to small that can't use in float
        N0_W = 10**((N0-30)/10)

        BS_FWA_Gain_NLOS_shdow_N0_multi = BS_FWA_Gain_NLOS_shdow_multi / N0_W
        
        BS_FWA_Gain_LOS_shdow_N0_multi = BS_FWA_Gain_LOS_shdow_multi / N0_W

        self.BS_FWA_Gain_NLOS_shdow_N0_multi = BS_FWA_Gain_NLOS_shdow_N0_multi
        self.BS_FWA_Gain_LOS_shdow_N0_multi = BS_FWA_Gain_LOS_shdow_N0_multi
        # thr_NLOS=transmit_bandwidth * np.log2(1+BS_FWA_Gain_NLOS_shdow_N0_multi*100)
        # thr_LOS=transmit_bandwidth * np.log2(1+BS_FWA_Gain_LOS_shdow_N0_multi*100)


    def init_pop(self):
        self.init_set = True
        init_pop_same = [self.create_solution() for _ in range(self.problem_size)]
        self.init_pop_same =  np.array(init_pop_same,dtype=object)
        # print(self.init_pop_same)
        # exit(1)


    
    def create_solution(self, minmax=0):
        # if not isinstance(self.lb, list):
        #     self.lb = [self.lb for _ in range(self.dimension)]
        #     self.ub = [self.ub for _ in range(self.dimension)]
        # self.lb = np.asarray(self.lb)
        # self.ub = np.asarray(self.ub)
        pos = np.asarray([(x*(self.ub-self.lb)+self.lb) for x in np.random.uniform(0,1,self.dimension)]).astype(int)
        
        fit = self.get_fitness(pos)
        weight = np.zeros(self.problem_size)
        
        return (pos, fit)
    
    def get_fitness(self, position=None, minmax=1):
        """     Assumption that objective function always return the original value
        :param position: 1-D numpy array
        :param minmax: 0- min problem, 1 - max problem
        :return:
        """
        # print(1.0 / (self.obj_func(position) + self.EPSILON))
        # return self.obj_func(position) if minmax == 0 else 1.0 / (self.obj_func(position) + self.EPSILON)
        # position=self.amend_position(self, position)
        
        position=np.clip(position, self.lb, self.ub)
        
        return self.obj_func(position.astype(int),self.BS_FWA_Gain_LOS_shdow_N0_multi,self.BS_FWA_distance)
        # ans=(ans[0],ans[1],ans[2],ans[3],ans[4],ans[5])

    def get_sorted_pop_and_global_best_solution(self, pop=None, id_fit=None, id_best=None, sort_func=None):
        """ Sort population and return the sorted population and the best position """
        # print(pop[:, 1])
        # cmp_func=functools.cmp_to_key(sort_func)
        # np.argsort(pop[:, 1], order=sort_func)
        dtype = np.dtype([
            ('bool_val', bool),
            ('float_val', float),
            ('float_val2', float),
        ])


        sort_pop = np.array(
            pop[:,self.ID_FIT],dtype=dtype
            )
        Stats = sort_pop['bool_val']
        EE = sort_pop['float_val']
        vio = sort_pop['float_val2']
        # 为了实现优先级的逻辑，我们需要将 True 设置为较大的数，False 设置为较小的数
        # 这里选择使用一个大于 float 的数表示 True，一个小于 float 的数表示 False
        Stats_numeric = np.where(Stats, 1e10, -1e10)


        sorted_indices = np.lexsort((Stats_numeric, -EE, vio))
        sorted_pop = pop[sorted_indices]
        # print(sorted_data)
        # sorted_indices = np.argsort(pop[:,self.ID_FIT], kind='quicksort', axis=0, cmp_to_key=lambda x: sort_func(pop[x[0]], pop[x[1]]))
        # print(sorted_pop)

        # print(sorted_indices)
        # exit(1)
        # print(pop[:,self.ID_FIT])
        # exit(1)
        # np_cmp_func =np.vectorize(cmp_func)
        # sort_data=np_cmp_func(pop[:,self.ID_FIT])
        # ans_index=np.argsort(sort_data)
        # sorted_pop=combin_pop[ans_index]
        # sorted_indices = np.argsort(pop[:, 1], order=sort_func)
        
        # # 根据排序后的索引重新排列 pop
        # sorted_pop = pop[sorted_indices]
        
        # 返回排序后的 pop 和最佳位置
        return sorted_pop, sorted_pop[id_best].copy()
    
    def update_sorted_population_and_global_best_solution(self, pop=None, id_best=None, g_best=None,sort_func=None,compare_func_bool=None,original_pop=None):
        """ Sort the population and update the current best position. Return the sorted population and the new current best position """
        if isinstance(original_pop, np.ndarray):
            combin_pop=np.append(pop,original_pop,axis=0)
        else:
            combin_pop = pop

        dtype = np.dtype([
            ('bool_val', bool),
            ('float_val', float),
            ('float_val2', float),
        ])
        # sorted_pop = np.array(
        #     combin_pop[:,self.ID_FIT],dtype=dtype
        #     )

        try:
            sorted_pop = np.array(
                combin_pop[:,self.ID_FIT],dtype=dtype
                )
        except:
            print(combin_pop[:,self.ID_FIT])
            # print(combin_pop[:,self.ID_FIT])
            exit(1)
        Stats = sorted_pop['bool_val']
        EE = sorted_pop['float_val']
        vio = sorted_pop['float_val2']
        # 为了实现优先级的逻辑，我们需要将 True 设置为较大的数，False 设置为较小的数
        # 这里选择使用一个大于 float 的数表示 True，一个小于 float 的数表示 False
        Stats_numeric = np.where(Stats, 1e10, -1e10)
        # cmp_func=functools.cmp_to_key(sort_func)
        # np_cmp_func =np.vectorize(cmp_func)
        # sort_data=np_cmp_func(combin_pop[:,self.ID_FIT])
        # ans_index=np.argsort(sort_data)
        # sorted_pop=combin_pop[ans_index]
        # sorted_pop=sorted_pop[:self.problem_size]
        sorted_indices = np.lexsort((Stats_numeric, -EE, vio))
        combin_pop = combin_pop[sorted_indices]
        combin_pop = combin_pop[:self.problem_size]
        # print(sorted_data)
        # sorted_indices = np.argsort(pop[:,self.ID_FIT], kind='quicksort', axis=0, cmp_to_key=lambda x: sort_func(pop[x[0]], pop[x[1]]))
        # print(sorted_pop)
        
        #############
        current_best = combin_pop[id_best]
        if compare_func_bool(current_best[self.ID_FIT],g_best[self.ID_FIT]):
            g_best = current_best.copy()
            return combin_pop, g_best
        return combin_pop, g_best
    
    def amend_position(self, position=None):
        ans=np.clip(position, self.lb, self.ub)
        if np.any(ans < self.lb) or np.any(ans >  self.ub):
            print(ans)
            print('error')
            exit(1)

        return np.clip(position, self.lb, self.ub)

    def amend_position_random(self, position=None):
        return np.where(np.logical_and(self.lb <= position, position <= self.ub), position, np.random.uniform(self.lb, self.ub))
    
    # def compare_Best_bool(self,news,olds):
    #     if news[0]==True and olds[0] == True and news[2]>olds[2]:
    #         return True
    #     if news[0]==True and olds[0] == False:
    #         return True
    #     if news[0]==False and olds[0] == False and news[3] < olds[3]:
    #         return True
    #     return False

    @abc.abstractmethod
    def run(self):
        'Return algorithms result'
        return NotImplemented
    
    def request_run(self,index_times,seed_list):
            
        
        # self.mutil_bestIndividual = []

        np.random.seed(seed_list[index_times])
        random.seed(seed_list[index_times])
        self.run()
        Data = {'convergence':self.convergence, 'constrained_violation_curve':self.constrained_violation_curve, 'bestIndividual':self.bestIndividual}
        return Data
    
    def mutil_run(self, Start_run_times,End_run_times,seed_list,Scenes,folder_extra_name=None):
        self.mutil_convergence = []
        self.mutil_constrained_violation_curve = []
        self.mutil_dimension = []
        
        if folder_extra_name==None:
            Folder_Path = f'./output/{Scenes}/{datetime.date.today()}/{self.__class__.__name__}'
        else:
            Folder_Path = f'./output/{Scenes}/{datetime.date.today()}/{self.__class__.__name__}/{folder_extra_name}'
        if not os.path.isdir(Folder_Path):
            os.makedirs(Folder_Path,mode=0o777)
            
        for time in range(Start_run_times,End_run_times):
            
            np.random.seed(seed_list[time])
            random.seed(seed_list[time])
            Folder_Path_time= f'{Folder_Path}/{time}'
            if not os.path.isdir(Folder_Path_time):
                os.makedirs(Folder_Path_time,mode=0o777)
            else:
                shutil.rmtree(Folder_Path_time)
                os.makedirs(Folder_Path_time,mode=0o777)
            self.init_pop()
            self.run()
            
            self.SaveTimes(Folder_Path)
            
            self.save_data(Folder_Path_time)

            # self.mutil_convergence.append(self.convergence)
            # self.mutil_constrained_violation_curve.append(self.constrained_violation_curve)
            # self.mutil_dimension.append(self.dimension)
            # if time == 0:
            #     self.avg_convergence =self.convergence
            #     self.avg_constrained_violation_curve =self.constrained_violation_curve
            #     self.avg_dimension = self.dimension
            #     self.avg_bestIndividual = self.bestIndividual
            # else:
            #     self.avg_convergence = (self.avg_convergence+ self.convergence)/2
            #     self.avg_constrained_violation_curve = (self.avg_constrained_violation_curve + self.constrained_violation_curve)/2
            #     self.avg_dimension =(self.avg_dimension + self.dimension)/2
            #     self.avg_bestIndividual =  (self.avg_bestIndividual + self.bestIndividual )/2
        

    def save_data(self,Folder_Path_time):
        # with open(f'{Folder_Path_time}/convergence.csv','w',newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(self.convergence)
        np.savetxt(f'{Folder_Path_time}/convergence.csv',self.convergence,delimiter=',')

        # with open(f'{Folder_Path_time}/constrained_violation_curve.csv','w',newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(self.constrained_violation_curve)
        np.savetxt(f'{Folder_Path_time}/constrained_violation_curve.csv',
                   self.constrained_violation_curve,delimiter=',')
        
        # with open(f'{Folder_Path_time}/dimension.csv','w',newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(self.dimension)
        # np.savetxt(f'{Folder_Path_time}/dimension.csv',self.dimension,delimiter=',')

        # with open(f'{Folder_Path_time}/bestIndividual.csv','w',newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(self.bestIndividual)
        np.savetxt(f'{Folder_Path_time}/bestIndividual.csv',self.bestIndividual,delimiter=',')



    def init_params(self):
        self.startTime=None 
        self.endTime=None
        self.executionTime=None ## algorithm execution time
        self.convertTime=None
        self.optimizer=None
        self.objfname=None
        self.best=None
        self.bestIndividual=None
        self.ALLData=None
        self.posRecord=[]

    def SaveTimes(self,Folder_Path):
        # print(self.executionTime)
        with open(f'{Folder_Path}/ExecutionTime.csv', 'a') as file:
            np.savetxt(file, [self.executionTime], delimiter=',')

    def show_time(self):
        print(self.executionTime)

    def get_best_individual(self):
        pass

    def progress(self, count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()
