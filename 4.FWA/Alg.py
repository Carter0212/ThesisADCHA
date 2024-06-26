from Base import Root
import numpy as np
import random
import math
import time

# def get_DE_Strategy_U_POS(problem_size, ID_POS, dimension, pop, i, CR):
    


class HHO_SMA_ch(Root):

    ID_MESK = 5
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func,ue_numbers,base_numbers):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.ue_numbers = ue_numbers
        self.base_numbers = base_numbers


    def run(self):
        timerStart=time.time() 
        self.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        t = 0
        convergence_curve=np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            original_pop = pop.copy()
            
            # self.progress(t,self.iteration,status="HHO is running...")
            E1=2*(1-(t/self.iteration)) # factor to show the decreaing energy of rabbit 
            p = 1-(t/self.iteration)
            for i in range(self.problem_size):
                if not pop[i][self.ID_FIT][0]:
                    ALL_MASK=pop[i,self.ID_FIT][self.ID_MESK]
                    Over_Power_mask = ALL_MASK[3]
                    rate_inadeuate_mask = ALL_MASK[2]
                    Zero_mask_connect_UE = ALL_MASK[1]
                    Zero_mask_connect_BS = ALL_MASK[0]
                    if(i==0):
                        print(pop[i,self.ID_POS])
                        print(self.dimension)
                        print(rate_inadeuate_mask,Over_Power_mask)
                        
                    for dim in range(self.dimension):
                        rate_check=np.isin(rate_inadeuate_mask,abs(dim-self.dimension//2)//self.base_numbers)
                        Zero_mask_connect_UE_Check = np.isin(Zero_mask_connect_UE,abs(dim-self.dimension//2)//self.base_numbers)
                        Zero_mask_connect_BS_Check = np.isin(Zero_mask_connect_BS,abs(dim-self.dimension//2)%self.base_numbers)
                        Power_check=np.isin(Over_Power_mask,abs(dim-self.dimension//2)%self.base_numbers)
                        test_mask=[]
                        change_connectORpower = random.choice([True, False])
                        
                        if dim < self.dimension//2:
                            if True in Zero_mask_connect_UE_Check:  
                                if pop[i,self.ID_POS][dim] < 500:
                                    pop[i,self.ID_POS][dim]*=0.8
                                    test_mask.append(dim)
                            if True in Zero_mask_connect_BS_Check:  
                                if pop[i,self.ID_POS][dim] < 500:
                                    pop[i,self.ID_POS][dim]*=0.8
                                    test_mask.append(dim)
                            if change_connectORpower and rate_check and pop[i,self.ID_POS][dim] <500:
                                pop[i,self.ID_POS][dim]*=1.2

                            
                        else:
                            # check Power
                            rate_check=np.isin(rate_inadeuate_mask,(dim//self.base_numbers))
                            Zero_mask_connect_UE_Check = np.isin(Zero_mask_connect_UE,dim//self.base_numbers)
                            if True in rate_check and Power_check in False:
                                if i == 0:
                                    print(rate_check,dim)   
                                if pop[i,self.ID_POS][dim] < 500:
                                    if i == 0:
                                        print(Power_check,dim)
                                    pop[i,self.ID_POS][dim]*=1.2
                                    test_mask.append(dim)
                            if True in Power_check and pop[i,self.ID_POS][dim] < 600 and not change_connectORpower:
                                pop[i,self.ID_POS][dim]*=1.2
                                test_mask.append(dim)
                    if(i==0):
                        print(pop[i,self.ID_POS])
                        
                        print(test_mask)
                    
                    
                    
                r8 = np.random.normal(0,1)
                j = 1-(i/self.problem_size)
                if j <= 0.3:
                    a= np.random.randint(self.lb, self.ub, size=self.dimension)
                    pop[i,self.ID_POS] = a
                elif r8 < p:
                    E0=2*random.random()-1  # -1<E0<1
                    Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper


                    # -------- Exploration phase Eq. (1) in paper -------------------#

                    if abs(Escaping_Energy)>=1:
                        q = random.random()
                        rand_Hawk_index = math.floor(self.problem_size*random.random())
                        pop_rand = pop[rand_Hawk_index,0]
                        if q<0.5:
                        # perch based on other family members
                            pop[i,self.ID_POS]=pop_rand-random.random()*abs(pop_rand-2*random.random()*pop[i,self.ID_POS])
                            
                        elif q>=0.5:
                            #perch on a random tall tree (random site inside group's home range)
                            pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS] - pop[:,self.ID_POS].mean(0))-random.random()*((self.ub-self.lb)*random.random()+self.lb)
                    elif abs(Escaping_Energy)<1:
                        #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                        #phase 1: ----- surprise pounce (seven kills) ----------
                        #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                        r=random.random() # probablity of each event
                        if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                            pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        
                        if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                            Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                            pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])

                        
                        #phase 2: --------performing team rapid dives (leapfrog movements)----------

                        if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                            #rabbit try to escape by many zigzag deceptive motions
                            Jump_strength=2*(1-random.random())
                            X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                            X1 = np.clip(X1, self.lb, self.ub)
                            X1_fitness = self.obj_func(X1)
                            if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                                pop[i] = (X1.copy(),X1_fitness)
                            else: # hawks perform levy-based short rapid dives around the rabbit
                                X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                                X2 = np.clip(X2, self.lb, self.ub)
                                X2_fitness = self.obj_func(X2)
                                if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                                    pop[i] = (X2.copy(),X2_fitness) 
                        
                        if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                            Jump_strength=2*(1-random.random())
                            X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))
                            X1 = np.clip(X1, self.lb, self.ub)
                            X1_fitness = self.obj_func(X1)
                            if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                                pop[i] = (X1.copy(),X1_fitness)
                            else: # Perform levy-based short rapid dives around the rabbit
                                X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                                X2 = np.clip(X2, self.lb, self.ub)
                                X2_fitness = self.obj_func(X2)
                                if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]):
                                    pop[i] = (X2.copy(),X2_fitness)
                elif r8 >= p:
                    pop[i,self.ID_POS] = pop[i,self.ID_POS]*np.random.normal(0,1,size=self.dimension)
                pop[i,self.ID_POS] = self.amend_position(pop[i,self.ID_POS])
                pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,original_pop)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][2]
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=(t+1)
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.convergence=convergence_curve
        self.best=best_Rabbit[self.ID_FIT][2] 
        self.bestIndividual = best_Rabbit[self.ID_POS]
        self.convergence=convergence_curve
    
    def Levy(self,dim):
        beta=1.5
        sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
        u= 0.01*np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v),(1/beta))
        step = np.divide(u,zz)
        return step
    
    def Weight(self,pop,problem):
        pass
        # POS=np.reshape(pop[problem,self.ID_POS],(2,base_numbers,ue_numbers)).copy()
        # if pop[problem,self.ID_FIT][0]:
        #     pass
        # else:
        #     check_conectional = (POS[0] >= 500)
        #     if not pop[problem,self.ID_FIT][1][0]:
        #         pass
        #     if not pop[problem,self.ID_FIT][1][1]:
        #         pass
        #     if not pop[problem,self.ID_FIT][1][2]:
        #         Zero_mask_connect_BS=np.where((np.sum(check_conectional,axis=1) <= 0))
        #     if not pop[problem,self.ID_FIT][1][3]:
        #         Zero_mask_connect_UE=np.where((np.sum(check_conectional,axis=1) <= 0))
        #     if not pop[problem,self.ID_FIT][1][4]:
        #         np.sum(Base_ue_ThroughtpusTable,axis=0) >= Min_Rate
        #     if not pop[problem,self.ID_FIT][1][5]:
        #         # happen Power Over Maximum constraint
        #         # check Which
        #         BS_power=np.sum(POS[1]*check_conectional,axis=1)
        #         Power_mask=(BS_power > 1000)
        #         whichBSOverPower=np.where(Power_mask)
        #         POS[1,whichBSOverPower,:]=

class Random(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func


    def run(self):
        timerStart=time.time() 
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        # pop = [self.create_solution() for _ in range(self.problem_size)]
        # pop=np.array(pop)
        # pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        # best_Rabbit = None
        while t < self.iteration:
            # self.progress(t,self.iteration,status="Random is running...")
            # original_pop,best_Rabbit=self.update_sorted_population_and_global_best_solution(original_pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,chaos_pop)
            
            for i in range(self.problem_size):
                buffer_people=self.create_solution()
                if t == 0 and i == 0:
                    best_Rabbit = buffer_people
                if self.compare_bool_func(buffer_people[self.ID_FIT],best_Rabbit[self.ID_FIT]):
                    # best_Rabbit[self.ID_POS] = buffer_people[self.ID_POS]
                    # best_Rabbit[self.ID_FIT] = buffer_people[self.ID_FIT]
                    best_Rabbit = buffer_people
            # pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][1]
            constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][2]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=(t+1)
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT][1] 
        self.bestIndividual = best_Rabbit[self.ID_POS]
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve

class ADCHA(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func,Mu,CR):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.Mu = Mu
        self.CR = CR


    def run(self):
        timerStart=time.time() 
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        if self.init_set == False:
            pop = [self.create_solution() for _ in range(self.problem_size)]
            pop=np.array(pop,dtype=object)
        else:
            pop = self.init_pop_same.copy()
        # self.chaos_var = random.random()
        self.chaos_var = random.random()
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            original_pop = pop.copy()
            
            # original_pop,best_Rabbit=self.update_sorted_population_and_global_best_solution(original_pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,chaos_pop)
            E1 = np.log(np.power((t+1)/self.iteration,1/3))
            # E1=2*(1-((t)/self.iteration))
            # p = 1-((t+1)/self.iteration)
            for i in range(self.problem_size):

                E0=2*random.random()-1  # -1<E0<1
                Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper


                # -------- Exploration phase Eq. (1) in paper -------------------#

                if abs(Escaping_Energy)>=1:
                    q = random.random()
                    rand_Hawk_index = math.floor(self.problem_size*random.random())
                    pop_rand = pop[rand_Hawk_index,0]
                    if q<0.5:
                    # perch based on other family members
                        pop[i,self.ID_POS]=(pop_rand-random.random()*abs(pop_rand-2*random.random()*
                                                                        pop[i,self.ID_POS]))
                        
                        
                    elif q>=0.5:
                        #perch on a random tall tree (random site inside group's home range)
                        pop[i,self.ID_POS]=((best_Rabbit[self.ID_POS] - pop[:,self.ID_POS].mean(0))-
                                            random.random()*((self.ub-self.lb)*random.random()+self.lb))
                elif abs(Escaping_Energy)<1:
                    #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
                    
                        # self.check_fuc(pop[i,self.ID_POS],'E>=0.5,r>=0.5')
                    #phase 1: ----- surprise pounce (seven kills) ----------
                    #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                    r=random.random() # probablity of each event
                    if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                        pop[i,self.ID_POS]=((best_Rabbit[self.ID_POS])-Escaping_Energy*
                                            abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS]))
                    
                    if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                        Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                        pop[i,self.ID_POS]=((best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])-Escaping_Energy*
                                            abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS]))
                    
                    #phase 2: --------performing team rapid dives (leapfrog movements)----------

                    if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                        #rabbit try to escape by many zigzag deceptive motions
                        Jump_strength=2*(1-random.random())
                        X1=(best_Rabbit[self.ID_POS]-Escaping_Energy*
                            abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])).astype(int)
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.get_fitness(X1)
                        if self.compare_bool_func(X1_fitness,pop[i,self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # hawks perform levy-based short rapid dives around the rabbit
                            X2=(best_Rabbit[self.ID_POS]-Escaping_Energy*
                                abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])+
                                np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension)))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.get_fitness(X2)
                            if self.compare_bool_func(X2_fitness,pop[i,self.ID_FIT]): # improved move?
                                pop[i] = (X2.copy(),X2_fitness) 
                    
                    if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                        Jump_strength=2*(1-random.random())
                        X1=(best_Rabbit[self.ID_POS]-Escaping_Energy*
                            abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0)))
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.get_fitness(X1)
                        if self.compare_bool_func(X1_fitness,pop[i,self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # Perform levy-based short rapid dives around the rabbit
                            X2=(best_Rabbit[self.ID_POS]-Escaping_Energy*
                                abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))+
                                np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension)))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.get_fitness(X2)
                            if self.compare_bool_func(X2_fitness,pop[i,self.ID_FIT]):
                                pop[i] = (X2.copy(),X2_fitness)

                # pop[i,self.ID_POS] = self.amend_position(pop[i,self.ID_POS])
                pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])
                pop[i]=self.DE_Strategy(pop,i)
            chaos_pop=self.chaos(pop.copy())
            combin_original_pop=np.append(original_pop,chaos_pop,axis=0)
            # pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,original_pop)
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,combin_original_pop)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][1]
            
            constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][2]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=(t+1)
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT][2] 
        self.bestIndividual = best_Rabbit[self.ID_POS]
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve

    # def chaos(self):
    #     chaos = random.random()
    #     pop_chaos = [self.create_solution() for _ in range(self.problem_size)]
    #     # pop_chaos = np.asarray([(chaos*(self.ub-self.lb)+self.lb) for x in range(self.dimension)]).astype(int)
    #     pop_chaos=np.array(pop_chaos,dtype=object)
        
    #     for ch in range(self.problem_size):
    #         chaos = self.Mu * chaos * (1-chaos)
    #         buffer=pop_chaos[ch,self.ID_POS]*chaos
    #         pop_chaos[ch,self.ID_POS]= buffer
    #         pop_chaos[ch,self.ID_FIT] = self.get_fitness(pop_chaos[ch,self.ID_POS])    
    #     return pop_chaos

    def chaos(self,original_pop):
        

        for ch in range(self.problem_size):
            buffer=original_pop[ch,self.ID_POS]*self.chaos_var
            original_pop[ch,self.ID_POS]= buffer
            original_pop[ch,self.ID_FIT] = self.get_fitness(original_pop[ch,self.ID_POS])
        self.chaos_var = self.Mu * self.chaos_var * (1-self.chaos_var) 
        return original_pop
    
    # def chaos(self):
        
    #     # pop_chaos = [self.create_solution() for _ in range(self.problem_size)]
    #     pop_chaos = np.asarray([(chaos*(self.ub-self.lb)+self.lb) for x in range(self.dimension)]).astype(int)
    #     pop_chaos=np.array(pop_chaos,dtype=object)
        
    #     for ch in range(self.problem_size):
    #         # chaos = self.Mu * self.chaos * (1-self.chaos)
    #         buffer=pop_chaos[ch,self.ID_POS]*
    #         pop_chaos[ch,self.ID_POS]= buffer
    #         pop_chaos[ch,self.ID_FIT] = self.get_fitness(pop_chaos[ch,self.ID_POS])    
    #     return pop_chaos

    def DE_Strategy(self,pop,i):
        # U_POS = get_DE_Strategy_U_POS(self.problem_size, self.ID_POS, self.dimension, pop, i, self.CR)
        choice_three_pop=random.sample(range(0,self.problem_size),3)
        V_POS=(pop[choice_three_pop[0]][self.ID_POS] + 
                (np.random.uniform(0,2)*
                (pop[choice_three_pop[1]][self.ID_POS]-pop[choice_three_pop[2]][self.ID_POS])))
        # U_POS=np.zeros(self.dimension)

        CR_compare =np.random.uniform(size=self.dimension) <= self.CR
        U_POS = (V_POS*CR_compare + pop[i][self.ID_POS]*(~CR_compare))

        U_FIT=self.get_fitness(U_POS)
        if self.compare_bool_func(U_FIT,pop[i][self.ID_FIT]):
            pop[i]=(U_POS.copy(),U_FIT)
        return pop[i]

    
    def Levy(self,dim):
        beta=1.5
        sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
        u= 0.01*np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v),(1/beta))
        step = np.divide(u,zz)
        return step

class HHO_Chaos(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func,Mu,CR):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.Mu = Mu
        self.CR = CR


    def run(self):
        timerStart=time.time() 
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop,dtype=object)
        self.chaos_var = random.random()
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            original_pop = pop.copy()
            # chaos_pop=self.chaos(original_pop.copy())
            # original_pop,best_Rabbit=self.update_sorted_population_and_global_best_solution(original_pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,chaos_pop)
            E1 = np.log(np.power((t+1)/self.iteration,1/3))
            
            p = 1-((t+1)/self.iteration)
            for i in range(self.problem_size):

                E0=2*random.random()-1  # -1<E0<1
                Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper


                # -------- Exploration phase Eq. (1) in paper -------------------#

                if abs(Escaping_Energy)>=1:
                    q = random.random()
                    rand_Hawk_index = math.floor(self.problem_size*random.random())
                    pop_rand = pop[rand_Hawk_index,0]
                    if q<0.5:
                    # perch based on other family members
                        pop[i,self.ID_POS]=(pop_rand-random.random()*abs(pop_rand-2*random.random()*
                                                                        pop[i,self.ID_POS]))
                        
                        
                    elif q>=0.5:
                        #perch on a random tall tree (random site inside group's home range)
                        pop[i,self.ID_POS]=((best_Rabbit[self.ID_POS] - pop[:,self.ID_POS].mean(0))-
                                            random.random()*((self.ub-self.lb)*random.random()+self.lb))
                elif abs(Escaping_Energy)<1:
                    #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
                    
                        # self.check_fuc(pop[i,self.ID_POS],'E>=0.5,r>=0.5')
                    #phase 1: ----- surprise pounce (seven kills) ----------
                    #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                    r=random.random() # probablity of each event
                    if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                        pop[i,self.ID_POS]=((best_Rabbit[self.ID_POS])-Escaping_Energy*
                                            abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS]))
                    
                    if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                        Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                        pop[i,self.ID_POS]=((best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])-Escaping_Energy*
                                            abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS]))
                    
                    #phase 2: --------performing team rapid dives (leapfrog movements)----------

                    if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                        #rabbit try to escape by many zigzag deceptive motions
                        Jump_strength=2*(1-random.random())
                        X1=(best_Rabbit[self.ID_POS]-Escaping_Energy*
                            abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])).astype(int)
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.get_fitness(X1)
                        if self.compare_bool_func(X1_fitness,pop[i,self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # hawks perform levy-based short rapid dives around the rabbit
                            X2=(best_Rabbit[self.ID_POS]-Escaping_Energy*
                                abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])+
                                np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension)))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.get_fitness(X2)
                            if self.compare_bool_func(X2_fitness,pop[i,self.ID_FIT]): # improved move?
                                pop[i] = (X2.copy(),X2_fitness) 
                    
                    if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                        Jump_strength=2*(1-random.random())
                        X1=(best_Rabbit[self.ID_POS]-Escaping_Energy*
                            abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0)))
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.get_fitness(X1)
                        if self.compare_bool_func(X1_fitness,pop[i,self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # Perform levy-based short rapid dives around the rabbit
                            X2=(best_Rabbit[self.ID_POS]-Escaping_Energy*
                                abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))+
                                np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension)))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.get_fitness(X2)
                            if self.compare_bool_func(X2_fitness,pop[i,self.ID_FIT]):
                                pop[i] = (X2.copy(),X2_fitness)

                pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])
                # pop[i]=self.DE_Strategy(pop,i)
            chaos_pop=self.chaos(pop.copy())
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,original_pop)
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,original_pop)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][1]
            
            constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][2]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=(t+1)
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT][2] 
        self.bestIndividual = best_Rabbit[self.ID_POS]
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve

    # def chaos(self,original_pop):
    #     chaos = random.random()

    #     for ch in range(self.problem_size):
    #         chaos = self.Mu * chaos * (1-chaos)
    #         buffer=original_pop[ch,self.ID_POS]*chaos
    #         original_pop[ch,self.ID_POS]= buffer
    #         original_pop[ch,self.ID_FIT] = self.get_fitness(original_pop[ch,self.ID_POS])    
    #     return original_pop

    # def chaos(self):
    #     chaos = random.random()
    #     pop_chaos = [self.create_solution() for _ in range(self.problem_size)]
    #     # pop_chaos = np.asarray([(chaos*(self.ub-self.lb)+self.lb) for x in range(self.dimension)]).astype(int)
    #     pop_chaos=np.array(pop_chaos,dtype=object)
        
    #     for ch in range(self.problem_size):
    #         chaos = self.Mu * chaos * (1-chaos)
    #         buffer=pop_chaos[ch,self.ID_POS]*chaos
    #         pop_chaos[ch,self.ID_POS]= buffer
    #         pop_chaos[ch,self.ID_FIT] = self.get_fitness(pop_chaos[ch,self.ID_POS])    
    #     return pop_chaos
    def chaos(self,original_pop):
        

        for ch in range(self.problem_size):
            
            # self.chaos_var = self.Mu * self.chaos_var * (1-self.chaos_var) 
            buffer=original_pop[ch,self.ID_POS]*self.chaos_var
            original_pop[ch,self.ID_POS]= buffer
            original_pop[ch,self.ID_FIT] = self.get_fitness(original_pop[ch,self.ID_POS])
        self.chaos_var = self.Mu * self.chaos_var * (1-self.chaos_var) 
        return original_pop

    
    def Levy(self,dim):
        beta=1.5
        sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
        u= 0.01*np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v),(1/beta))
        step = np.divide(u,zz)
        return step

class HHO_DE(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func,Mu,CR):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.Mu = Mu
        self.CR = CR


    def run(self):
        timerStart=time.time() 
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        if self.init_set == False:
            pop = [self.create_solution() for _ in range(self.problem_size)]
            pop=np.array(pop,dtype=object)
        else:
            pop = self.init_pop_same.copy()
                        
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            original_pop = pop.copy()
            # original_pop,best_Rabbit=self.update_sorted_population_and_global_best_solution(original_pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func)
            # E1 = np.log(np.power((t+1)/self.iteration,1/3))
            E1=2*(1-((t)/self.iteration))
            p = 1-((t+1)/self.iteration)
            for i in range(self.problem_size):

                E0=2*random.random()-1  # -1<E0<1
                Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper


                # -------- Exploration phase Eq. (1) in paper -------------------#

                if abs(Escaping_Energy)>=1:
                    q = random.random()
                    rand_Hawk_index = math.floor(self.problem_size*random.random())
                    pop_rand = pop[rand_Hawk_index,0]
                    if q<0.5:
                    # perch based on other family members
                        pop[i,self.ID_POS]=(pop_rand-random.random()*abs(pop_rand-2*random.random()*
                                                                        pop[i,self.ID_POS]))
                        
                        
                    elif q>=0.5:
                        #perch on a random tall tree (random site inside group's home range)
                        pop[i,self.ID_POS]=((best_Rabbit[self.ID_POS] - pop[:,self.ID_POS].mean(0))-
                                            random.random()*((self.ub-self.lb)*random.random()+self.lb))
                elif abs(Escaping_Energy)<1:
                    #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
                    
                        # self.check_fuc(pop[i,self.ID_POS],'E>=0.5,r>=0.5')
                    #phase 1: ----- surprise pounce (seven kills) ----------
                    #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                    r=random.random() # probablity of each event
                    if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                        pop[i,self.ID_POS]=((best_Rabbit[self.ID_POS])-Escaping_Energy*
                                            abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS]))
                    
                    if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                        Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                        pop[i,self.ID_POS]=((best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])-Escaping_Energy*
                                            abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS]))
                    
                    #phase 2: --------performing team rapid dives (leapfrog movements)----------

                    if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                        #rabbit try to escape by many zigzag deceptive motions
                        Jump_strength=2*(1-random.random())
                        X1=(best_Rabbit[self.ID_POS]-Escaping_Energy*
                            abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])).astype(int)
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.get_fitness(X1)
                        if self.compare_bool_func(X1_fitness,pop[i,self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # hawks perform levy-based short rapid dives around the rabbit
                            X2=(best_Rabbit[self.ID_POS]-Escaping_Energy*
                                abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])+
                                np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension)))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.get_fitness(X2)
                            if self.compare_bool_func(X2_fitness,pop[i,self.ID_FIT]): # improved move?
                                pop[i] = (X2.copy(),X2_fitness) 
                    
                    if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                        Jump_strength=2*(1-random.random())
                        X1=(best_Rabbit[self.ID_POS]-Escaping_Energy*
                            abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0)))
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.get_fitness(X1)
                        if self.compare_bool_func(X1_fitness,pop[i,self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # Perform levy-based short rapid dives around the rabbit
                            X2=(best_Rabbit[self.ID_POS]-Escaping_Energy*
                                abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))+
                                np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension)))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.get_fitness(X2)
                            if self.compare_bool_func(X2_fitness,pop[i,self.ID_FIT]):
                                pop[i] = (X2.copy(),X2_fitness)

                pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])
                pop[i]=self.DE_Strategy(pop,i)
            
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,original_pop)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][1]
            
            constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][2]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=(t+1)
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT][2] 
        self.bestIndividual = best_Rabbit[self.ID_POS]
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve

    def DE_Strategy(self,pop,i):
        # U_POS = get_DE_Strategy_U_POS(self.problem_size, self.ID_POS, self.dimension, pop, i, self.CR)
        choice_three_pop=random.sample(range(0,self.problem_size),3)
        V_POS=(pop[choice_three_pop[0]][self.ID_POS] + 
                (np.random.uniform(0,2)*
                (pop[choice_three_pop[1]][self.ID_POS]-pop[choice_three_pop[2]][self.ID_POS])))
        # U_POS=np.zeros(self.dimension)

        CR_compare =np.random.uniform(size=self.dimension) <= self.CR
        U_POS = (V_POS*CR_compare + pop[i][self.ID_POS]*(~CR_compare))

        U_FIT=self.get_fitness(U_POS)
        if self.compare_bool_func(U_FIT,pop[i][self.ID_FIT]):
            pop[i]=(U_POS.copy(),U_FIT)
        return pop[i]

    
    def Levy(self,dim):
        beta=1.5
        sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
        u= 0.01*np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v),(1/beta))
        step = np.divide(u,zz)
        return step

class Chaos(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func,Mu,CR):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.Mu = Mu
        self.CR = CR


    def run(self):
        timerStart=time.time() 
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop,dtype=object)
        self.chaos_var = random.random()
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            # original_pop = pop.copy()
            pop=self.chaos(pop.copy())
            pop,buffer_best_Rabbit=self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func)
            if self.compare_bool_func(buffer_best_Rabbit[self.ID_FIT],best_Rabbit[self.ID_FIT]):
                best_Rabbit = buffer_best_Rabbit
            convergence_curve[t]=best_Rabbit[self.ID_FIT][1]
            constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][2]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=(t+1)
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT][2] 
        self.bestIndividual = best_Rabbit[self.ID_POS]
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve

    # def chaos(self,original_pop):
    #     chaos = random.random()

    #     for ch in range(self.problem_size):
    #         chaos = self.Mu * chaos * (1-chaos)
    #         buffer=original_pop[ch,self.ID_POS]*chaos
    #         original_pop[ch,self.ID_POS]= buffer
    #         original_pop[ch,self.ID_FIT] = self.get_fitness(original_pop[ch,self.ID_POS])    
    #     return original_pop

    def chaos(self,original_pop):
        

        for ch in range(self.problem_size):
            
            # self.chaos_var = self.Mu * self.chaos_var * (1-self.chaos_var) 
            buffer=original_pop[ch,self.ID_POS]*self.chaos_var
            original_pop[ch,self.ID_POS]= buffer
            original_pop[ch,self.ID_FIT] = self.get_fitness(original_pop[ch,self.ID_POS])
        self.chaos_var = self.Mu * self.chaos_var * (1-self.chaos_var) 
        return original_pop

    
    def Levy(self,dim):
        beta=1.5
        sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
        u= 0.01*np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v),(1/beta))
        step = np.divide(u,zz)
        return step

class DE(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func,Mu,CR):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.Mu = Mu
        self.CR = CR


    def run(self):
        timerStart=time.time() 
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop,dtype=object)
                        
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            original_pop = pop.copy()
            original_pop,best_Rabbit=self.update_sorted_population_and_global_best_solution(original_pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func)
            
            for i in range(self.problem_size):
                # pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])
                pop[i]=self.DE_Strategy(pop,i)
            
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,original_pop)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][1]
            
            constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][2]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=(t+1)
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT][2] 
        self.bestIndividual = best_Rabbit[self.ID_POS]
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve

    def DE_Strategy(self,pop,i):
        # U_POS = get_DE_Strategy_U_POS(self.problem_size, self.ID_POS, self.dimension, pop, i, self.CR)
        choice_three_pop=random.sample(range(0,self.problem_size),3)
        V_POS=(pop[choice_three_pop[0]][self.ID_POS] + 
                (np.random.uniform(0,2)*
                (pop[choice_three_pop[1]][self.ID_POS]-pop[choice_three_pop[2]][self.ID_POS])))
        # U_POS=np.zeros(self.dimension)

        CR_compare =np.random.uniform(size=self.dimension) <= self.CR
        U_POS = (V_POS*CR_compare + pop[i][self.ID_POS]*(~CR_compare))

        U_FIT=self.get_fitness(U_POS)
        if self.compare_bool_func(U_FIT,pop[i][self.ID_FIT]):
            pop[i]=(U_POS.copy(),U_FIT)
        return pop[i]

class HHO_SMA_Chaos(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.Mu = 4


    def run(self):
        timerStart=time.time() 
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            original_pop = pop.copy()
            
            chaos_pop=self.chaos(original_pop.copy())
            original_pop,best_Rabbit=self.update_sorted_population_and_global_best_solution(original_pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,chaos_pop)
            # self.progress(t,self.iteration,status="HHO is running...")
            # E1=2*(1-(t/self.iteration)) # factor to show the decreaing energy of rabbit 
            E1 = np.log(np.power(t/self.iteration,1/3))
            p = 1-(t/self.iteration)
            for i in range(self.problem_size):
                r8 = np.random.normal(0,1)
                r7 = np.random.normal(0,1)
                # if np.random.normal(0,1) <0.03:
                #     a= np.random.randint(self.lb, self.ub, size=self.dimension)
                #     pop[i,self.ID_POS] = a
                # elif r8 < p:
                E0=2*random.random()-1  # -1<E0<1
                Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper


                # -------- Exploration phase Eq. (1) in paper -------------------#

                if abs(Escaping_Energy)>=1:
                    q = random.random()
                    rand_Hawk_index = math.floor(self.problem_size*random.random())
                    pop_rand = pop[rand_Hawk_index,0]
                    if q<0.5:
                    # perch based on other family members
                        pop[i,self.ID_POS]=int(pop_rand-random.random()*abs(pop_rand-2*random.random()*pop[i,self.ID_POS]))
                        
                    elif q>=0.5:
                        #perch on a random tall tree (random site inside group's home range)
                        pop[i,self.ID_POS]=int((best_Rabbit[self.ID_POS] - pop[:,self.ID_POS].mean(0))-random.random()*((self.ub-self.lb)*random.random()+self.lb))
                elif abs(Escaping_Energy)<1:
                    #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                    #phase 1: ----- surprise pounce (seven kills) ----------
                    #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                    r=random.random() # probablity of each event
                    if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                        X3 = int(best_Rabbit[self.ID_POS]+r7*(pop[:,self.ID_POS].mean(0)-pop[i,self.ID_POS]))
                        X3_1=int((best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS]))
                        X3_fitness=self.obj_func(X3)
                        X3_1_fitness=self.obj_func(X3_1)
                        if self.compare_bool_func(X3_fitness,X3_1_fitness):
                            pop[i]=(X3.copy(),X3_fitness)
                        else:
                            pop[i]=(X3_1.copy(),X3_1_fitness)
                    
                    if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                        Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                        X4=int((best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS]))
                        X4_1=int((best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS]))
                        X4_fitness=self.obj_func(X4)
                        X4_1_fitness=self.obj_func(X4_1)
                        if self.compare_bool_func(X4_fitness,X4_1_fitness):
                            pop[i]=(X4.copy(),X4_fitness)
                        else:
                            pop[i]=(X4_1.copy(),X4_1_fitness)
                    
                    #phase 2: --------performing team rapid dives (leapfrog movements)----------

                    if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                        #rabbit try to escape by many zigzag deceptive motions
                        Jump_strength=2*(1-random.random())
                        X1=int(best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS]))
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.obj_func(X1)
                        if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # hawks perform levy-based short rapid dives around the rabbit
                            X2=int(best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension)))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.obj_func(X2)
                            if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                                pop[i] = (X2.copy(),X2_fitness) 
                    
                    if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                        Jump_strength=2*(1-random.random())
                        X1=int(best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0)))
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.obj_func(X1)
                        if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # Perform levy-based short rapid dives around the rabbit
                            X2=int(best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension)))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.obj_func(X2)
                            if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]):
                                pop[i] = (X2.copy(),X2_fitness)
                # elif r8 >= p:
                #     pop[i,self.ID_POS] = pop[i,self.ID_POS]*np.random.normal(0,1,size=self.dimension)
                pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])

            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,original_pop)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][2]
            
            constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][3]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=(t+1)
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT][2] 
        self.bestIndividual = best_Rabbit[self.ID_POS]
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve

    def chaos(self,original_pop):
        chaos = random.random()

        for ch in range(self.problem_size):
            chaos = self.Mu * chaos * (1-chaos)
            buffer=original_pop[ch,self.ID_POS]*chaos
            original_pop[ch,self.ID_POS]= buffer
            original_pop[ch,self.ID_FIT] = self.get_fitness(original_pop[ch,self.ID_POS])    
        return original_pop

    
    def Levy(self,dim):
        beta=1.5
        sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
        u= 0.01*np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v),(1/beta))
        step = np.divide(u,zz)
        return step

class HHO_SMA_DE(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.CR=0.2

    def run(self):
        timerStart=time.time() 
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop,dtype=object)
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            original_pop = pop.copy()
            # self.progress(t,self.iteration,status="HHO is running...")
            # E1=2*(1-(t/self.iteration)) # factor to show the decreaing energy of rabbit 
            E1 = np.log(np.power((t+1)/self.iteration,1/3))
            p = 1-((t+1)/self.iteration)
            for i in range(self.problem_size):
                r8 = np.random.normal(0,1)
                r7 = np.random.normal(0,1)
                # if np.random.normal(0,1) <0.03:
                #     a= np.random.randint(self.lb, self.ub, size=self.dimension)
                #     pop[i,self.ID_POS] = a
                # elif r8 < p:
                E0=2*random.random()-1  # -1<E0<1
                Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper


                # -------- Exploration phase Eq. (1) in paper -------------------#

                if abs(Escaping_Energy)>=1:
                    q = random.random()
                    rand_Hawk_index = math.floor(self.problem_size*random.random())
                    pop_rand = pop[rand_Hawk_index,0]
                    if q<0.5:
                    # perch based on other family members
                        pop[i,self.ID_POS]=pop_rand-random.random()*abs(pop_rand-2*random.random()*pop[i,self.ID_POS])
                        
                    elif q>=0.5:
                        #perch on a random tall tree (random site inside group's home range)
                        pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS] - pop[:,self.ID_POS].mean(0))-random.random()*((self.ub-self.lb)*random.random()+self.lb)
                elif abs(Escaping_Energy)<1:
                    #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                    #phase 1: ----- surprise pounce (seven kills) ----------
                    #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                    r=random.random() # probablity of each event
                    if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                        X3 = best_Rabbit[self.ID_POS]+r7*(pop[:,self.ID_POS].mean(0)-pop[i,self.ID_POS])
                        X3_1=(best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X3_fitness=self.obj_func(X3)
                        X3_1_fitness=self.obj_func(X3_1)
                        if self.compare_bool_func(X3_fitness,X3_1_fitness):
                            pop[i]=(X3.copy(),X3_fitness)
                        else:
                            pop[i]=(X3_1.copy(),X3_1_fitness)
                    
                    if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                        Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                        X4=(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X4_1=(best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X4_fitness=self.obj_func(X4)
                        X4_1_fitness=self.obj_func(X4_1)
                        if self.compare_bool_func(X4_fitness,X4_1_fitness):
                            pop[i]=(X4.copy(),X4_fitness)
                        else:
                            pop[i]=(X4_1.copy(),X4_1_fitness)
                    
                    #phase 2: --------performing team rapid dives (leapfrog movements)----------

                    if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                        #rabbit try to escape by many zigzag deceptive motions
                        Jump_strength=2*(1-random.random())
                        X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.obj_func(X1)
                        if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # hawks perform levy-based short rapid dives around the rabbit
                            X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.obj_func(X2)
                            if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                                pop[i] = (X2.copy(),X2_fitness) 
                    
                    if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                        Jump_strength=2*(1-random.random())
                        X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.obj_func(X1)
                        if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # Perform levy-based short rapid dives around the rabbit
                            X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.obj_func(X2)
                            if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]):
                                pop[i] = (X2.copy(),X2_fitness)
                # elif r8 >= p:
                #     pop[i,self.ID_POS] = pop[i,self.ID_POS]*np.random.normal(0,1,size=self.dimension)
                pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])
                self.DE_Strategy(pop,i)
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,original_pop)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][2]
            constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][3]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=(t+1)
        
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT][2] 
        self.bestIndividual = best_Rabbit[self.ID_POS]

    
    def Levy(self,dim):
        beta=1.5
        sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
        u= 0.01*np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v),(1/beta))
        step = np.divide(u,zz)
        return step
    
    
    def DE_Strategy(self,pop,i):
        choice_three_pop=random.sample(range(0,self.problem_size),3)
        while i in choice_three_pop:
            choice_three_pop=random.sample(range(0,self.problem_size),3)
        V_POS=pop[choice_three_pop[0]][self.ID_POS] + (np.random.uniform(0.2,0.8)*(pop[choice_three_pop[1]][self.ID_POS]-pop[choice_three_pop[2]][self.ID_POS]))
        U_POS=np.zeros(self.dimension)

        CR_compare =np.random.uniform(size=self.dimension) <= self.CR
        U_POS = V_POS*CR_compare + pop[i][self.ID_POS]*(~CR_compare)


        ##Crossover operation
        # for dim in range(self.dimension):
        #     if np.random.random() <= self.CR:
        #         U_POS[dim] = V_POS[dim]
        #     else:
        #         U_POS[dim] = pop[i][self.ID_POS][dim]

        ## Selection operation
        U_FIT=self.get_fitness(U_POS)
        if self.compare_bool_func(U_FIT,pop[i][self.ID_FIT]):
            pop[i]=(U_POS.copy(),U_FIT)


class HHO_SMA(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        


    def run(self):
        timerStart=time.time() 
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        # pop = [self.create_solution() for _ in range(self.problem_size)]
        # pop=np.array(pop,dtype=object)
        if self.init_set == False:
            pop = [self.create_solution() for _ in range(self.problem_size)]
            pop=np.array(pop,dtype=object)
        else:
            pop = self.init_pop_same.copy()
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            # original_pop = pop.copy()
            # self.progress(t,self.iteration,status="HHO is running...")
            E1=2*(1-(t/self.iteration)) # factor to show the decreaing energy of rabbit 
            # E1 = np.log(np.power((t+1)/self.iteration,1/3))
            p = 1-((t+1)/self.iteration)
            for i in range(self.problem_size):
                r8 = np.random.normal(0,1)
                r7 = np.random.normal(0,1)
                # r9 = np.random.normal(0,1)
                # r6 = np.random.normal(0,1)
                
                
                if r7 < 0.03:
                    pop[i,self.ID_POS] = np.asarray([(r6*(self.ub-self.lb))+self.lb for r6 in np.random.uniform(0,1,self.dimension)])
                    # np.asarray([(x*(self.ub-self.lb)+self.lb) for x in np.random.uniform(0,1,self.dimension)]).astype(int)
                elif r8 < p:
                    E0=2*random.random()-1  # -1<E0<1
                    Escaping_Energy=(E0)*E1
                # -------- Exploration phase Eq. (1) in paper -------------------#
                    if abs(Escaping_Energy)>=1:
                        q = random.random()
                        rand_Hawk_index = math.floor(self.problem_size*random.random())
                        pop_rand = pop[rand_Hawk_index,0]
                        if q>=0.5:
                        # perch based on other family members
                            # pop[i,self.ID_POS]=int(pop_rand-random.random()*np.abs(pop_rand-2*random.random()*pop[i,self.ID_POS]))
                            pop[i,self.ID_POS]=pop_rand-np.random.random()*np.abs(pop_rand-2*random.random()*pop[i,self.ID_POS])
                        elif q<0.5:
                            #perch on a random tall tree (random site inside group's home range)
                            pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS] - pop[:,self.ID_POS].mean(0))-random.random()*((self.ub-self.lb)*random.random()+self.lb)
                    elif abs(Escaping_Energy)<1:
                        #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                        #phase 1: ----- surprise pounce (seven kills) ----------
                        #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                        r=random.random() # probablity of each event
                        if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                            X3 = best_Rabbit[self.ID_POS]+Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                            # X3_1=(best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                            X3_fitness=self.get_fitness(X3)
                            # X3_1_fitness=self.get_fitness(X3_1)
                            # if self.compare_bool_func(X3_fitness,X3_1_fitness):
                            pop[i]=(X3,X3_fitness)
                            # else:
                            #     pop[i]=(X3_1.copy(),X3_1_fitness)
                        
                        if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                            Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                            X4=(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                            # X4_1=(best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                            X4_fitness=self.get_fitness(X4)
                            # X4_1_fitness=self.get_fitness(X4_1)
                            # if self.compare_bool_func(X4_fitness,X4_1_fitness):
                            pop[i]=(X4,X4_fitness)
                            # else:
                                # pop[i]=(X4_1.copy(),X4_1_fitness)
                        
                        #phase 2: --------performing team rapid dives (leapfrog movements)----------

                        if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                            #rabbit try to escape by many zigzag deceptive motions
                            Jump_strength=2*(1-random.random())
                            X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                            X1 = np.clip(X1, self.lb, self.ub)
                            X1_fitness = self.get_fitness(X1)
                            if self.compare_bool_func(X1_fitness,pop[i,self.ID_FIT]): # improved move?
                                pop[i] = (X1,X1_fitness)
                            else: # hawks perform levy-based short rapid dives around the rabbit
                                X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                                X2 = np.clip(X2, self.lb, self.ub)
                                X2_fitness = self.get_fitness(X2)
                                if self.compare_bool_func(X2_fitness,pop[i,self.ID_FIT]): # improved move?
                                    pop[i] = (X2,X2_fitness)
                                
                        ''''
                        X2=(best_Rabbit[self.ID_POS]-Escaping_Energy*
                                abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))+
                                np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension)))
                            X2 = np.clip(X2, self.lb, self.ub)
                        '''
                        if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                            Jump_strength=2*(1-random.random())
                            X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))
                            X1 = np.clip(X1, self.lb, self.ub)
                            X1_fitness = self.get_fitness(X1)
                            if self.compare_bool_func(X1_fitness,pop[i,self.ID_FIT]): # improved move?
                                pop[i] = (X1,X1_fitness)
                            else: # Perform levy-based short rapid dives around the rabbit
                                X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                                X2 = np.clip(X2, self.lb, self.ub)
                                X2_fitness = self.get_fitness(X2)
                                if self.compare_bool_func(X2_fitness,pop[i,self.ID_FIT]):
                                    pop[i] = (X2,X2_fitness)
                            
                elif r8 >= p:
                    pop[i,self.ID_POS] = pop[i,self.ID_POS]*np.random.normal(0,1,size=self.dimension)
                pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][1]
            constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][2]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=(t+1)
        
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT][2] 
        self.bestIndividual = best_Rabbit[self.ID_POS]

    
    def Levy(self,dim):
        beta=1.5
        sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
        u= 0.01*np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v),(1/beta))
        step = np.divide(u,zz)
        return step

class HHO(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func

    def check_fuc(self,value,printWord):
        if np.any(value<0):
            print(printWord)
            print(value)
            exit(1)
        else:
            pass


    def run(self):
        timerStart=time.time() 
        self.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        # pop = [self.create_solution() for _ in range(self.problem_size)]
        # pop=np.array(pop,dtype=object)
        if self.init_set == False:
            pop = [self.create_solution() for _ in range(self.problem_size)]
            pop=np.array(pop,dtype=object)
        else:
            pop = self.init_pop_same.copy()
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            E1=2*(1-((t)/self.iteration)) # factor to show the decreaing energy of rabbit 

            for i in range(self.problem_size):
                
                E0=2*random.random()-1  # -1<E0<1
                Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper


                # -------- Exploration phase Eq. (1) in paper -------------------#

                if abs(Escaping_Energy)>=1:
                    q = random.random()
                    rand_Hawk_index = math.floor(self.problem_size*random.random())
                    pop_rand = pop[rand_Hawk_index,0]
                    if q<0.5:
                    # perch based on other family members
                        pop[i,self.ID_POS]=pop_rand-random.random()*abs(pop_rand-2*random.random()*pop[i,self.ID_POS])
                        
                        # self.check_fuc(pop[i,self.ID_POS],'E>=1,q<0.5')
                    elif q>=0.5:
                        #perch on a random tall tree (random site inside group's home range)
                        
                        pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS] - pop[:,self.ID_POS].mean(0))-random.random()*((self.ub-self.lb)*random.random()+self.lb)
                        # self.check_fuc(pop[i,self.ID_POS],'E>=1,q>=0.5')
                elif abs(Escaping_Energy)<1:
                    #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                    #phase 1: ----- surprise pounce (seven kills) ----------
                    #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                    r=random.random() # probablity of each event
                    if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                        pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        # self.check_fuc(pop[i,self.ID_POS],'E<0.5,r>=0.5')
                    if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                        Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                        pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        # self.check_fuc(pop[i,self.ID_POS],'E>=0.5,r>=0.5')
                    
                    #phase 2: --------performing team rapid dives (leapfrog movements)----------

                    if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                        #rabbit try to escape by many zigzag deceptive motions
                        Jump_strength=2*(1-random.random())
                        X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.get_fitness(X1)
                        if self.compare_bool_func(X1_fitness,pop[i,self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # hawks perform levy-based short rapid dives around the rabbit
                            X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.get_fitness(X2)
                            if self.compare_bool_func(X2_fitness,pop[i,self.ID_FIT]): # improved move?
                                pop[i] = (X2.copy(),X2_fitness)
                        # self.check_fuc(pop[i,self.ID_POS],'E>=0.5,q<0.5') 
                    
                    if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                        Jump_strength=2*(1-random.random())
                        X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.get_fitness(X1)
                        if self.compare_bool_func(X1_fitness,pop[i,self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # Perform levy-based short rapid dives around the rabbit
                            X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.get_fitness(X2)
                            if self.compare_bool_func(X2_fitness,pop[i,self.ID_FIT]):
                                pop[i] = (X2.copy(),X2_fitness)
                        # self.check_fuc(pop[i,self.ID_POS],'E>0.5,q<0.5')
                pop[i,self.ID_POS] = self.amend_position(pop[i,self.ID_POS])
                pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][1]
            constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][2]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=(t+1)
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT][2] 
        self.bestIndividual = best_Rabbit[self.ID_POS]
        
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve

    
    def Levy(self,dim):
        beta=1.5
        sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
        u= 0.01*np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v),(1/beta))
        step = np.divide(u,zz)
        return step

class BaseSMA(Root):
    """
    Modified version of: Slime Mould Algorithm (SMA)
            (Slime Mould Algorithm: A New Method for Stochastic Optimization)
    Notes:
            + Selected 2 unique and random solution to create new solution (not to create variable) --> remove third loop in original version
            + Check bound and update fitness after each individual move instead of after the whole population move in the original version
    """

    ID_WEI = 2

    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func,z=0.03):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.z = z

    def create_solution(self, minmax=0):
        # if not isinstance(self.lb, list):
        #     self.lb = [self.lb for _ in range(self.dimension)]
        #     self.ub = [self.ub for _ in range(self.dimension)]
        # self.lb = np.asarray(self.lb)
        # self.ub = np.asarray(self.ub)
        pos = np.asarray([x*(self.ub-self.lb)+self.lb for x in np.random.uniform(0,1,self.dimension)])
        fit = self.get_fitness(pos)
        weight = np.zeros(self.dimension)
        return (pos, fit,weight)
    
    def run(self):
        timerStart=time.time() 
        self.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        t = 0
        convergence_curve=np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        
        
        pop,g_best =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            self.progress(t,self.iteration,status="BaseSMA is running...")
            # print(pop[:,self.ID_FIT])
            s = pop[0][self.ID_FIT][2] - pop[-1][self.ID_FIT][2] + self.EPSILON
            
             # calculate the fitness weight of each slime mold
            for i in range(0, self.problem_size):
                # Eq.(2.5)
                if i <= int(self.problem_size / 2):
                    pop[i][self.ID_WEI] = 1 + np.random.uniform(0, 1, self.dimension) * np.log10(abs((pop[0][self.ID_FIT][2] - pop[i][self.ID_FIT][2])) / s + 1)
                else:
                    pop[i][self.ID_WEI] = 1 - np.random.uniform(0, 1, self.dimension) * np.log10(abs((pop[0][self.ID_FIT][2] - pop[i][self.ID_FIT][2])) / s + 1)

            a = np.arctanh(-((t + 1) / self.iteration) + 1)                        # Eq.(2.4)
            b = 1 - (t + 1) / self.iteration

            # Update the Position of search agents
            for i in range(0, self.problem_size):
                
                if np.random.uniform() < self.z:  # Eq.(2.7)
                    pos_new = np.random.uniform(self.lb, self.ub,self.dimension)
                else:
                    p = np.tanh(abs(pop[i][self.ID_FIT][2] - g_best[self.ID_FIT][2]))    # Eq.(2.2)
                    vb = np.random.uniform(-a, a, self.dimension)                      # Eq.(2.3)
                    vc = np.random.uniform(-b, b, self.dimension)

                    # two positions randomly selected from population, apply for the whole problem size instead of 1 variable
                    id_a, id_b = np.random.choice(list(set(range(0, self.problem_size)) - {i}), 2, replace=False)
                    pos_1 = g_best[self.ID_POS] + vb * (pop[i][self.ID_WEI] * pop[id_a][self.ID_POS] - pop[id_b][self.ID_POS])
                    pos_2 = vc * pop[i][self.ID_POS]
                    pos_new = np.where(np.random.uniform(0, 1, self.dimension) < p, pos_1, pos_2)

                # Check bound and re-calculate fitness after each individual move
                pos_new = self.amend_position(pos_new)
                fit_new = self.get_fitness(pos_new)
                pop[i][self.ID_POS] = pos_new
                pop[i][self.ID_FIT] = fit_new
                
            
            # Sorted population and update the global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,g_best,self.compare_func,self.compare_bool_func)   
            convergence_curve[t]=g_best[self.ID_FIT][2]
            # if (t%1==0):
            #         print(['At iteration '+ str(t)+ ' the best fitness is '+ str(g_best[self.ID_FIT])])
            t=(t+1)
        self.convergence=convergence_curve
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=g_best[self.ID_FIT][2] 
        self.bestIndividual = g_best[self.ID_POS]
        
        

class OriginalSMA(Root):
    """
        The original version of: Slime Mould Algorithm (SMA)
            (Slime Mould Algorithm: A New Method for Stochastic Optimization)
        Link:
            https://doi.org/10.1016/j.future.2020.03.055
    """

    ID_WEI = 2

    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func,z=0.03):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.z = z

    def create_solution(self, minmax=0):
        # if not isinstance(self.lb, list):
        #     self.lb = [self.lb for _ in range(self.dimension)]
        #     self.ub = [self.ub for _ in range(self.dimension)]
        # self.lb = np.asarray(self.lb)
        # self.ub = np.asarray(self.ub)
        pos = np.asarray([x*(self.ub-self.lb)+self.lb for x in np.random.uniform(0,1,self.dimension)])
        fit = self.get_fitness(pos)
        weight = np.zeros(self.dimension)
        return (pos, fit,weight)
    
    def run(self):
        timerStart=time.time() 
        self.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        
        # print(np.shape(pop[0, self.ID_WEI]))
        # combined_weights = np.vstack(pop[:, self.ID_WEI])
        # print(combined_weights)
        # print(np.shape(combined_weights))  # 输出 (100, 200)
        
        pop,g_best =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            # self.progress(t,self.iteration,status="BaseSMA is running...")
            s = abs(pop[0][self.ID_FIT][2] - pop[-1][self.ID_FIT][2] + self.EPSILON)
            
             # calculate the fitness weight of each slime mold

            for i in range(0, self.problem_size):
                # Eq.(2.5)
                if i <= int(self.problem_size / 2):
                    pop[i][self.ID_WEI] = 1 + np.random.uniform(0, 1, self.dimension) * np.log10(abs((pop[0][self.ID_FIT][2] - pop[i][self.ID_FIT][2])) / s + 1)
                    
                else:
                    pop[i][self.ID_WEI] = 1 - np.random.uniform(0, 1, self.dimension) * np.log10(abs((pop[0][self.ID_FIT][2] - pop[i][self.ID_FIT][2])) / s + 1)

            # 使用NumPy向量化操作代替for循环
            # indices = np.arange(self.problem_size)
            # condition = indices <= self.problem_size / 2
            # # # 创建一个与pop数组相同形状的随机权重数组
            # random_weights = np.random.uniform(0, 1, (self.problem_size, self.dimension))
            # FIT_Result = np.array([item[2] for item in pop[:,self.ID_FIT]])
            
            # # # print(result)
            
            # # # print(pop[:][self.ID_FIT][1])
            
            # # # 计算权重
            # log_argument = np.abs((pop[0,self.ID_FIT][2] - FIT_Result) / s + 1)
            # log_argument_2d = log_argument[:, np.newaxis] * np.ones((1, self.dimension))
            # # log_argument_2d = np.tile(log_argument, (1, self.dimension))
            # print(np.shape(log_argument_2d))
            # print(np.shape(random_weights))
            # # # 使用NumPy条件(condition)来分别计算两种情况的权重
            # # # print(np.shape(random_weights))
            # # print( np.log10(abs(log_argument[condition])))
            # print(np.shape(pop[condition,self.ID_POS][:]))
            # print(np.shape(random_weights[condition]))
            # print(np.shape(np.log10(log_argument_2d[condition])))
            # combined_weights = np.vstack(pop[:, self.ID_WEI])
            # pop[condition, self.ID_WEI] = 1 + random_weights[condition] * np.log10(log_argument_2d[condition])
            
            # pop[~condition, self.ID_WEI] = 1 - random_weights[~condition] * np.log10(log_argument_2d[~condition])
            
            a = np.arctanh(-((t + 1) / self.iteration) + 1)                        # Eq.(2.4)
            b = 1 - (t + 1) / self.iteration

            # Update the Position of search agents
            for i in range(0, self.problem_size):
                
                if np.random.uniform() < self.z:  # Eq.(2.7)
                    pop[i][self.ID_POS]  = np.random.uniform(self.lb, self.ub,self.dimension)
                else:
                    # p = np.tanh(abs(pop[i][self.ID_FIT][2] - g_best[self.ID_FIT][2]))    # Eq.(2.2)
                    # vb = np.random.uniform(-a, a, self.dimension)                      # Eq.(2.3)
                    # vc = np.random.uniform(-b, b, self.dimension)
                    # for j in range(0, self.dimension):
                    # # two positions randomly selected from population
                    #     id_a, id_b = np.random.choice(list(set(range(0, self.problem_size)) - {i}), 2, replace=False)

                    #     if np.random.uniform() < p:
                    #         pop[i][self.ID_POS][j] = g_best[self.ID_POS][j] + vb[j] * (
                    #                     pop[i][self.ID_WEI][j] * pop[id_a][self.ID_POS][j] - pop[id_b][self.ID_POS][j])
                    #     else:
                    #         pop[i][self.ID_POS][j] = vc[j] * pop[i][self.ID_POS][j]

                    # p = np.tanh(np.abs(pop[:, self.ID_FIT, 2] - g_best[self.ID_FIT, 2]))  # 计算p，使用向量化操作
                    p = np.tanh(abs(pop[i][self.ID_FIT][2] - g_best[self.ID_FIT][2])) 
                    vb = np.random.uniform(-a, a, self.dimension)  # Eq.(2.3)
                    vc = np.random.uniform(-b, b, self.dimension)

                    # random_indices = np.random.choice(np.delete(np.arange(self.problem_size), i), size=(2, self.dimension), replace=False)
                    # 创建一个空的NumPy数组，用于存储索引数组
                    random_indices = np.empty((self.dimension, 2), dtype=int)

                    # 循环处理每个self.dimension
                    for j in range(self.dimension):
                        # 创建一个从0到self.problem_size-1的整数序列
                        indices = np.arange(self.problem_size)
                        # print(j)
                        # 从整数序列中删除j，以确保选择不是j自己
                        
                        indices_without_j = np.delete(indices, i)
                        
                        # 从删除了j后的序列中随机选择两个不同的索引
                        selected_indices = np.random.choice(indices_without_j, size=2, replace=False)
                        
                        # 将选择的索引存储到index_arrays中
                        random_indices[j] = selected_indices

                    condition = np.random.uniform(0, 1, (self.dimension,)) < p
                    # print(random_indices[condition])
                    try:
                        pop[i][self.ID_POS][condition] = g_best[self.ID_POS][condition] + vb[condition] * (
                                    pop[i][self.ID_WEI][condition] * pop[random_indices[condition][0]][self.ID_POS][condition] -
                                    pop[random_indices[condition][1]][self.ID_POS][condition])
                    except:
                        pass
                    # 使用向量化操作更新pop[i][self.ID_POS]的另一部分
                    try:
                        pop[i][self.ID_POS][~condition] = vc[~condition] * pop[i][self.ID_POS][~condition]
                    except:
                        pass
                    # print(np.shape(pop[i][self.ID_POS]))
                    
                    # print(pop[i][self.ID_POS])
                    # exit(1)
                    # if np.random.uniform() < p:
                    # pop[i][self.ID_POS][condition] = g_best[self.ID_POS][condition] + vb[condition] * (
                    #             pop[i][self.ID_WEI][condition] * pop[id_a][self.ID_POS][condition] - pop[id_b][self.ID_POS][condition])
                    # # else:
                    # pop[i][self.ID_POS][condition] = vc[condition] * pop[i][self.ID_POS][condition]
                    # if np.all(condition):
                    #     pos_new = vc * pop[i][self.ID_POS]
                    # elif not np.all(condition):
                    #     pos_new = g_best[self.ID_POS] + vb * (pop[i][self.ID_WEI] * pop[random_indices[condition][0]][self.ID_POS] - pop[random_indices[condition][1]][self.ID_POS])
                    # else:
                    #     pos_1 = g_best[self.ID_POS] + vb * (pop[i][self.ID_WEI] * pop[random_indices[condition][0]][self.ID_POS] - pop[random_indices[condition][1]][self.ID_POS])
                    #     pos_2 = vc * pop[i][self.ID_POS]
                    #     pos_new = np.where(np.random.uniform(0, 1, self.dimension) < p, pos_1, pos_2)

                # Check bound and re-calculate fitness after each individual move
            for i in range(0, self.problem_size):
                pos_new = self.amend_position(pop[i][self.ID_POS])
                fit_new = self.get_fitness(pos_new)
                pop[i][self.ID_POS] = pos_new
                pop[i][self.ID_FIT] = fit_new
            
            
            # Sorted population and update the global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,g_best,self.compare_func,self.compare_bool_func)   
            
            convergence_curve[t]=g_best[self.ID_FIT][2]
            constrained_violation_curve[t] = g_best[self.ID_FIT][3]
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(g_best[self.ID_FIT])])
            t=(t+1)
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=g_best[self.ID_FIT][2] 
        self.bestIndividual = g_best[self.ID_POS]

class GA(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func,parents_portion,mutation_prob,crossover_prob,elit_ratio,cross_type):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.elit_ratio = elit_ratio
        self.parents_portion = parents_portion
        assert (self.parents_portion<=1\
                and self.parents_portion>=0),\
        "parents_portion must be in range [0,1]" 
        self.cross_type = cross_type
        self.parents_portion = parents_portion
        self.par_s=int(self.parents_portion*self.problem_size)
        trl=self.problem_size-self.par_s
        if trl % 2 != 0:
            self.par_s+=1


        self.mutation_prob = mutation_prob
        assert (self.mutation_prob<=1 and self.mutation_prob>=0), \
        "mutation_probability must be in range [0,1]"

        self.crossover_prob=crossover_prob
        assert (self.crossover_prob<=1 and self.crossover_prob>=0), \
        "mutation_probability must be in range [0,1]"

        assert (self.elit_ratio <=1 and self.elit_ratio>=0),\
        "elit_ratio must be in range [0,1]" 

        trl=self.problem_size*self.elit_ratio
        if trl<1 and self.elit_ratio>0:
            self.num_elit=1
        else:
            self.num_elit=int(trl)

        assert(self.par_s>=self.num_elit), \
        "\n number of parents must be greater than number of elits"
    
    def run(self):
        ## use Roulette Wheel & elit 
        timerStart=time.time() 
        self.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        pop,best_chromosome =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        
        while t < self.iteration:
            # self.progress(t,self.iteration,status="GA is running...")

            # Normalizing objective function
            normal_obj = np.zeros(self.problem_size)
            for index,i in enumerate(pop[:,self.ID_FIT].copy()):
                normal_obj[index] = i[2]
            # normal_obj = np.zeros(self.problem_size)
            # normal_obj=pop[:,self.ID_FIT].copy()
            # print(np.shape(normal_obj))
            maxnorm = np.amax(normal_obj)
            # print(normal_obj)
            # print('======================')
            # print(maxnorm)
            normal_obj = maxnorm - normal_obj +1

            #############################################################        
            # Calculate probability
            sum_normobj=np.sum(normal_obj)
            prob=np.zeros(self.problem_size)
            prob=normal_obj/sum_normobj
            
            cumprob=np.cumsum(prob)
            par = np.zeros((self.par_s,2),dtype=pop.dtype)
            # Select elite individuals
            # print(pop[0,1].dtype)
            par[0:self.num_elit] = pop[0:self.num_elit].copy()

            # Select non-elite individuals using roulette wheel selection
            index=np.searchsorted(cumprob,np.random.random(self.par_s-self.num_elit))
            par[self.num_elit:self.par_s]=pop[index].copy()

            ef_par_list = (np.random.random(self.par_s)<=self.crossover_prob)
            par_count = ef_par_list.sum()


            elite_par=par[ef_par_list].copy()

            ## New generation

            pop[:self.par_s] = par[:self.par_s].copy()
            
            for k in range(self.par_s,self.problem_size,2):
                r1 = np.random.randint(0,par_count)
                r2 = np.random.randint(0,par_count)
                pvar1 = elite_par[r1].copy()
                pvar2 = elite_par[r2].copy()

                ch1,ch2 = self.crossover(pvar1,pvar2)

                ch1=self.mut(ch1)
                ch2=self.mutmidle(ch2,pvar1,pvar2)
                pop[k] = ch1.copy()
                pop[k+1] = ch2.copy()
                pop[k,self.ID_FIT]=self.get_fitness(pop[k,self.ID_POS])
                pop[k+1,self.ID_FIT]=self.get_fitness(pop[k+1,self.ID_POS])
            
            pop,best_chromosome = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_chromosome,self.compare_func,self.compare_bool_func)
            convergence_curve[t]=best_chromosome[self.ID_FIT][2]
            constrained_violation_curve[t] = best_chromosome[self.ID_FIT][3]
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_chromosome[self.ID_FIT])])
            t=(t+1)
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_chromosome[self.ID_FIT][2] 
        self.bestIndividual = best_chromosome[self.ID_POS]
        


    def crossover(self,chromosome1,chromosome2):
         
        ofs1=chromosome1.copy()
        ofs2=chromosome2.copy()
        

        if self.cross_type=='one_point':
            ran=np.random.randint(0,self.dimension)
            for i in range(0,ran):
                ofs1[self.ID_POS][i]=chromosome2[self.ID_POS][i].copy()
                ofs2[self.ID_POS][i]=chromosome1[self.ID_POS][i].copy()
  
        if self.cross_type=='two_point':
                
            ran1=np.random.randint(0,self.dimension)
            ran2=np.random.randint(ran1,self.dimension)
                
            for i in range(ran1,ran2):
                ofs1[self.ID_POS,i]=chromosome2[self.ID_POS,i].copy()
                ofs2[self.ID_POS,i]=chromosome1[self.ID_POS,i].copy()
            
        if self.cross_type=='uniform':
                
            for i in range(0, self.dimension):
                
                ran=np.random.random()
                if ran <0.5:
                    ofs1[self.ID_POS][i]=chromosome2[self.ID_POS][i].copy()
                    ofs2[self.ID_POS][i]=chromosome1[self.ID_POS][i].copy() 
                   
        return ofs1,ofs2
    
    def mut(self,chromosome):
        
        # for i in self.integers[0]:
        #     ran=np.random.random()
        #     if ran < self.mutation_prob:
                
        #         chromosome[i]=self.lb+np.random.random()*(self.ub-self.lb)    
                    
        

        for i in range(self.dimension):            
            ran=np.random.random()
            if ran < self.mutation_prob:   
                chromosome[self.ID_POS][i]=self.lb+np.random.random()*(self.ub-self.lb)    
            
        return chromosome
    
    def mutmidle(self, x, p1, p2):
        # for i in self.integers[0]:
        #     ran=np.random.random()
        #     if ran < self.mutation_prob:
        #         if p1[i]<p2[i]:
        #             x[i]=np.random.randint(p1[i],p2[i])
        #         elif p1[i]>p2[i]:
        #             x[i]=np.random.randint(p2[i],p1[i])
        #         else:
        #             x[i]=np.random.randint(self.var_bound[i][0],\
        #          self.var_bound[i][1]+1)
                        
        for i in range(self.dimension):                
            ran=np.random.random()
            if ran < self.mutation_prob:   
                if p1[self.ID_POS][i]<p2[self.ID_POS][i]:
                    x[self.ID_POS][i]=p1[self.ID_POS][i]+np.random.random()*(p2[self.ID_POS][i]-p1[self.ID_POS][i])  
                elif p1[self.ID_POS][i]>p2[self.ID_POS][i]:
                    x[self.ID_POS][i]=p2[self.ID_POS][i]+np.random.random()*(p1[self.ID_POS][i]-p2[self.ID_POS][i])
                else:
                    x[self.ID_POS][i]=self.lb+np.random.random()*(self.ub-self.lb)   
        return x

        
    def show_time(self):
        print(self.executionTime)

    def get_best_individual(self):
        pass

class ES(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func,parents_portion,mutation_prob,crossover_prob,elit_ratio,cross_type):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        
    
    def run(self):
        ## use Roulette Wheel & elit 
        timerStart=time.time() 
        self.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        Best_pop = np.asarray([0 for x in range(self.dimension)]).astype(int)
        Best_fit = self.get_fitness(Best_pop)
        dimensions = np.arange(self.lb, self.ub + 1)
        mesh = np.meshgrid(*[dimensions] * self.dimension, indexing='ij')
        combinations = np.vstack([dim.ravel() for dim in mesh])


        # t = 0
        # convergence_curve=np.zeros(self.iteration)
        # constrained_violation_curve = np.zeros(self.iteration)
        # pop = [self.create_solution() for _ in range(self.problem_size)]
        # pop=np.array(pop)
        # pop,best_chromosome =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        
        # while t < self.iteration:
        #     # self.progress(t,self.iteration,status="GA is running...")

        #     # Normalizing objective function
        #     normal_obj = np.zeros(self.problem_size)
        #     for index,i in enumerate(pop[:,self.ID_FIT].copy()):
        #         normal_obj[index] = i[2]
        #     # normal_obj = np.zeros(self.problem_size)
        #     # normal_obj=pop[:,self.ID_FIT].copy()
        #     # print(np.shape(normal_obj))
        #     maxnorm = np.amax(normal_obj)
        #     # print(normal_obj)
        #     # print('======================')
        #     # print(maxnorm)
        #     normal_obj = maxnorm - normal_obj +1

        #     #############################################################        
        #     # Calculate probability
        #     sum_normobj=np.sum(normal_obj)
        #     prob=np.zeros(self.problem_size)
        #     prob=normal_obj/sum_normobj
            
        #     cumprob=np.cumsum(prob)
        #     par = np.zeros((self.par_s,2),dtype=pop.dtype)
        #     # Select elite individuals
        #     # print(pop[0,1].dtype)
        #     par[0:self.num_elit] = pop[0:self.num_elit].copy()

        #     # Select non-elite individuals using roulette wheel selection
        #     index=np.searchsorted(cumprob,np.random.random(self.par_s-self.num_elit))
        #     par[self.num_elit:self.par_s]=pop[index].copy()

        #     ef_par_list = (np.random.random(self.par_s)<=self.crossover_prob)
        #     par_count = ef_par_list.sum()


        #     elite_par=par[ef_par_list].copy()

        #     ## New generation

        #     pop[:self.par_s] = par[:self.par_s].copy()
            
        #     for k in range(self.par_s,self.problem_size,2):
        #         r1 = np.random.randint(0,par_count)
        #         r2 = np.random.randint(0,par_count)
        #         pvar1 = elite_par[r1].copy()
        #         pvar2 = elite_par[r2].copy()

        #         ch1,ch2 = self.crossover(pvar1,pvar2)

        #         ch1=self.mut(ch1)
        #         ch2=self.mutmidle(ch2,pvar1,pvar2)
        #         pop[k] = ch1.copy()
        #         pop[k+1] = ch2.copy()
        #         pop[k,self.ID_FIT]=self.get_fitness(pop[k,self.ID_POS])
        #         pop[k+1,self.ID_FIT]=self.get_fitness(pop[k+1,self.ID_POS])
            
        #     pop,best_chromosome = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_chromosome,self.compare_func,self.compare_bool_func)
        #     convergence_curve[t]=best_chromosome[self.ID_FIT][2]
        #     constrained_violation_curve[t] = best_chromosome[self.ID_FIT][3]
        #     if (t%1==0):
        #             print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_chromosome[self.ID_FIT])])
        #     t=(t+1)
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_chromosome[self.ID_FIT][2] 
        self.bestIndividual = best_chromosome[self.ID_POS]

#寫出窮舉法

class ExhaustiveSearch(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        
    
    def run(self):
        ## use Roulette Wheel & elit 
        timerStart=time.time() 
        self.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        Best_pop = np.asarray([0 for x in range(self.dimension)]).astype(int)
        Best_fit = self.get_fitness(Best_pop)
        dimensions = np.arange(self.lb, self.ub + 1)
        mesh = np.meshgrid(*[dimensions] * self.dimension, indexing='ij')
        combinations = np.vstack([dim.ravel() for dim in mesh])
        for i in range(self.problem_size):
            current_fit = self.get_fitness(combinations[i])
            if self.compare_bool_func(current_fit[0],Best_fit[0]):
                Best_fit = current_fit
                Best_pop = combinations[i]
        self.best = Best_fit[2]
        self.bestIndividual = Best_pop
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        print(self.best)
        print(self.bestIndividual)
        print(self.executionTime)
        print(self.endTime)
        print(self.startTime)
        return self.best, self.bestIndividual, self.executionTime, self.endTime, self.startTime
class NSGAII(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func,parents_portion,mutation_prob,crossover_prob):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.parents_portion = parents_portion
        self.num_of_tour_particips=2
        self.spread_factor = 1
        assert (self.parents_portion<=1\
                and self.parents_portion>=0),\
        "parents_portion must be in range [0,1]" 
        self.parents_portion = parents_portion
        self.par_s=int(self.parents_portion*self.problem_size)
        trl=self.problem_size-self.par_s
        if trl % 2 != 0:
            self.par_s+=1


        self.mutation_prob = mutation_prob
        assert (self.mutation_prob<=1 and self.mutation_prob>=0), \
        "mutation_probability must be in range [0,1]"

        self.crossover_prob=crossover_prob
        assert (self.crossover_prob<=1 and self.crossover_prob>=0), \
        "mutation_probability must be in range [0,1]"
        "elit_ratio must be in range [0,1]" 

    
    def run(self):
        ## use Roulette Wheel & elit 
        timerStart=time.time() 
        self.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        # pop = [self.create_solution() for _ in range(self.problem_size)]
        # pop=np.array(pop,dtype=object)
        ## fast_nondominated_sort
        if self.init_set == False:
            pop = [self.create_solution() for _ in range(self.problem_size)]
            pop=np.array(pop,dtype=object)
        else:
            pop = self.init_pop_same.copy()
        pop,best_chromosome =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        
        while t < self.iteration:
            original_pop = pop.copy()
            children_pop = None
            for i in range(self.problem_size//2):
            ## binary_tournament
                parent1,parent1_index = self.__tournament(pop)
                parent2,parent2_index = self.__tournament(pop,parent1_index)
                review_count = 0
                repeat_index = np.array([parent1_index])
                while (parent1[self.ID_POS]== parent2[self.ID_POS]).all():
                    review_count += 1
                    if review_count >= 2:
                        break
                    parent2,parent2_index = self.__tournament(pop,parent1_index)
                    repeat_index=np.append(repeat_index,parent2_index)

            ## CrossOver
                if np.random.random() < self.crossover_prob: 
                    child1_POS, child2_POS = self.__crossover(parent1, parent2)
                    child1_POS = np.clip(child1_POS, self.lb, self.ub)
                    child2_POS = np.clip(child2_POS , self.lb, self.ub)
                else:
                    child1_POS=parent1[self.ID_POS]
                    child1_POS=np.clip(parent1[self.ID_POS], self.lb, self.ub)
                    child2_POS=parent2[self.ID_POS]
                    child2_POS=np.clip(parent2[self.ID_POS], self.lb, self.ub)
            ## Mutaion
                if np.random.random() < self.mutation_prob:
                    child1_POS=self.__mutate(child1_POS)
                if np.random.random() < self.mutation_prob:
                    child2_POS=self.__mutate(child2_POS)
                child1_FIT=self.get_fitness(child1_POS)
                child2_FIT=self.get_fitness(child2_POS)
            ## put in children population
                if children_pop is None:
                    children_pop = np.array([(child1_POS,child1_FIT),(child2_POS,child2_FIT)],dtype=object)
                else:
                    children_pop=np.append(children_pop,np.array([(child1_POS,child1_FIT),(child2_POS,child2_FIT)],dtype=object),axis=0)
                # np.append(children_pop,np.array([(child1_POS,child1_FIT)]),axis=0)
                # np.append(children_pop,np.array([(child2_POS,child2_FIT)]),axis=0)
            # print(children_pop.shape)
            pop,best_chromosome = self.update_sorted_population_and_global_best_solution(children_pop,self.ID_MIN_PROB,best_chromosome,self.compare_func,self.compare_bool_func,original_pop)
            # = self.update_sorted_population_and_global_best_solution(children_pop,self.ID_MIN_PROB,best_chromosome,self.compare_func,self.compare_bool_func,original_pop)
            convergence_curve[t]=best_chromosome[self.ID_FIT][1]
            constrained_violation_curve[t] = best_chromosome[self.ID_FIT][2]
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_chromosome[self.ID_FIT])])
            t=t+1
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_chromosome[self.ID_FIT][1] 
        self.bestIndividual = best_chromosome[self.ID_POS]


    def __crossover(self, individual1, individual2):
        # child1=self.create_solution()
        # child2=self.create_solution()
        beta_array = np.array([self.__get_beta() for _ in range(self.dimension)])
        # beta = [self.__get_beta()]
        # child1_POS = ((1+beta_array)*individual1[self.ID_POS]+(1-beta_array)*individual2[self.ID_POS])*0.5
        
        child1_POS = ((1+beta_array)*individual1[self.ID_POS]+(1-beta_array)*individual2[self.ID_POS])*0.5
        child2_POS = ((1-beta_array)*individual1[self.ID_POS]+(1+beta_array)*individual2[self.ID_POS])*0.5
        # x1 = (individual1.features[i] + individual2.features[i]) / 2
        # x2 = abs((individual1.features[i] - individual2.features[i]) / 2)
        # child1.features[i] = x1 + beta * x2
        # child2.features[i] = x1 - beta * x2
        return child1_POS, child2_POS

    def __get_beta(self):
        u = random.random()
        if u <= 0.5:
            return (2 * u) ** (1 / (self.spread_factor + 1))
        return (2 * (1 - u)) ** (-1 / (self.spread_factor + 1))

    def __tournament(self, population,skip_num=None):
        # 如果有skip_num，則不會選到skip_num的index
        if skip_num is None:
            participants=np.random.choice(np.arange(0, population.shape[0]), size=self.num_of_tour_particips, replace=False)
        else:
            population_indices = np.arange(0, population.shape[0])
            population_indices = population_indices[population_indices != skip_num]
            participants = np.random.choice(population_indices, size=self.num_of_tour_particips, replace=False)

        best = None
        for participant in participants:
            if best is None or (self.compare_bool_func(population[participant][self.ID_FIT],best[self.ID_FIT])):
                best = population[participant]
                best_index = participant

        return best,best_index
        
        # 將結果限制在變數的範圍內
    def __mutate(self, individual):
        U = np.random.uniform(0, 1, len(individual))
        gamma_m = np.random.exponential()  # This is an example value, you should set it according to your problem
        theta_1 = (1 - (individual - self.lb) / (self.ub - self.lb)) ** (gamma_m + 1)
        theta_2 = (1 - (self.ub - individual) / (self.ub - self.lb)) ** (gamma_m + 1)
        theta = np.where(U <= 0.5, (2 * U + (1 - 2 * U) * theta_1) ** (1 / (gamma_m + 1)) - 1,
                        1 - (2 * (1 - U) + (2 * U - 1) * theta_2) ** (1 / (gamma_m + 1)))
        mutated_individual = individual + theta * (self.ub - self.lb)
        return mutated_individual
    
    # def __mutate(self, child_POS):
    #     # num_of_features = len(child.features)
    #     # for gene in range(num_of_features):
    #     ran=np.random.random()
    #     delta_array = np.array([self.__get_delta(child_POS) for _ in range(self.dimension)])
    #     child_POS = child_POS + delta_array * (self.ub - self.lb)
        
    #     # 將結果限制在變數的範圍內
    #     child_POS = self.amend_position(child_POS)
    #     return child_POS

    # def __get_delta(self, child_POS):
    #     # self.problem.variables_range[1]
    #     u = random.random()
    #     mutation_param = np.random.exponential()
    #     if u <= 0.5:
    #         theta1 = (1 - (child_POS - self.lb) / (self.ub - self.lb)) ** (mutation_param + 1)
    #         return ((2 * u) + (1 - 2 * u) * theta1) ** (1 / (mutation_param + 1)) - 1
    #     else:
    #         theta2 = (1 - (self.ub - child_POS) / (self.ub - self.lb)) ** (mutation_param + 1)
    #         return 1 - ((2 * (1 - u)) + (2 * u - 1) * theta2) ** (1 / (mutation_param + 1))
    
    # def mut(self,chromosome):
        
    #     num_of_features = len(child.features)
    #     for gene in range(num_of_features):
    #         u, delta = self.__get_delta()
    #         if u < 0.5:
    #             child.features[gene] += delta * (child.features[gene] - self.problem.variables_range[gene][0])
    #         else:
    #             child.features[gene] += delta * (self.problem.variables_range[gene][1] - child.features[gene])
    #         if child.features[gene] < self.problem.variables_range[gene][0]:
    #             child.features[gene] = self.problem.variables_range[gene][0]
    #         elif child.features[gene] > self.problem.variables_range[gene][1]:
    #             child.features[gene] = self.problem.variables_range[gene][1]
            
    #     return chromosome

    def __get_delta(self, child_POS):
        # self.problem.variables_range[1]
        u = random.random()
        mutation_param = np.random.exponential()
        if u <= 0.5:
            theta1 = (1 - (child_POS - self.lb) / (self.ub - self.lb)) ** (mutation_param + 1)
            return ((2 * u) + (1 - 2 * u) * theta1) ** (1 / (mutation_param + 1)) - 1
        else:
            theta2 = (1 - (self.ub - child_POS) / (self.ub - self.lb)) ** (mutation_param + 1)
            return 1 - ((2 * (1 - u)) + (2 * u - 1) * theta2) ** (1 / (mutation_param + 1))
    

        
    def show_time(self):
        print(self.executionTime)

    

    def produce_Pt_from_Rt(self):
        # Implement Algorithm 2 here
        pass

    

    
    
    
    