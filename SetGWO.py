"""
Created on Mon November 9 2020

@author: Dr. Mehdy Roayaei
This is the source code of SetGWO, a binarization of GWO Metaheuristic algorithm
"""

import random
import numpy as np
import math
from solution import solution
import time
import Util as util
from fitness import calculate_fitness 
import copy 

    

def GWO_Set_VC(GreyWolves_num,MaxIt,instance, threshold):    
    
    dim = instance.budget    
    
    # initialize alpha, beta, and delta_pos
    if instance.min_max == 1: 
        #minimization
        # initialize alpha, beta, and delta_pos
        Alpha_pos=np.zeros(dim, dtype=int)
        Alpha_score=float("+inf")
    
        Beta_pos=np.zeros(dim, dtype=int)
        Beta_score=float("+inf")
    
        Delta_pos=np.zeros(dim, dtype=int)
        Delta_score=float("+inf")
    elif instance.min_max == 2 :
        # maximization
        Alpha_pos=np.zeros(dim, dtype=int)
        Alpha_score=float("-inf")
    
        Beta_pos=np.zeros(dim, dtype=int)
        Beta_score=float("-inf")
    
        Delta_pos=np.zeros(dim, dtype=int)
        Delta_score=float("-inf")   
    
    #all items
    num_nodes = len(instance.graph.nodes())
    seq = list((instance.graph.nodes()).keys())
    seq_set = [set(seq)] * GreyWolves_num
    
    #Initialize the positions of greywolves
    GreyWolves = [None] * GreyWolves_num       

    i = 0
    for  i in range(GreyWolves_num):                
        GreyWolves[i] = set(random.sample(seq,k = dim))    
        

     # Loop counter
    print("GWO is optimizing  \""+ calculate_fitness.__name__+ "\"")  
    timerStart = time.time() 
    
 
    # Main loop
    Alpha_pos = set()
    Beta_pos = set()
    Delta_pos = set()
    for it in range(MaxIt):       
        for i in range(GreyWolves_num):                                    
            # Calculate objective function for each search wolf            
            fitness = calculate_fitness(instance, list(GreyWolves[i]))
            
            # Update Alpha, Beta, and Delta
            if fitness > Alpha_score :
                # Update alpha
                Alpha_score = fitness 
                Alpha_pos= set(GreyWolves[i].copy())
            
            
            if (fitness < Alpha_score and fitness > Beta_score):
                # Update beta
                Beta_score = fitness  
                Beta_pos = set(GreyWolves[i].copy())
            
            
            if (fitness < Alpha_score and fitness < Beta_score and fitness > Delta_score): 
                # Update delta
                Delta_score = fitness 
                Delta_pos= set(GreyWolves[i].copy())
            
                        
        # a increase linearly from 2 to 0, exploitation         
        a = 2 - it *(2/MaxIt)     
        
        for i in range(0,GreyWolves_num):                        
            if (len(seq_set[i]) < threshold * dim):
                # refill set_seq
                seq_set[i] = set(seq)
            
            #generate random numbers
            r1 = random.random()
            r2 = random.random()
            C = 2 * r2
            A = 2 * a * r1 - a
            
            # determine which leader is closer 
            intersec_alpha = len(Alpha_pos.intersection(GreyWolves[i]))
            intersec_beta  = len(Beta_pos.intersection(GreyWolves[i]))
            intersec_delta = len(Delta_pos.intersection(GreyWolves[i]))        
            
            leader = set()
            if (intersec_alpha > intersec_beta and intersec_alpha > intersec_delta):
                leader = Alpha_pos
            elif  (intersec_beta > intersec_alpha and intersec_beta > intersec_delta):
                leader = Beta_pos
            else :
                leader = Delta_pos   
            
     

            leader_items = list(leader)
            

            CBound = math.ceil(C * len(leader_items))
            CBound = min(CBound, len(leader_items))            
            leader_set = set(random.sample(leader, k=CBound))
            
            if (abs(A)<1):                
                A = abs(A)
                # exploitation : get closer to leaders
                # select new items which are not in omega
                new_items = leader_set - GreyWolves[i]
                old_items = GreyWolves[i] - leader_set
                D = len(old_items)
                selecteds_num = abs(math.ceil(A * D))
                selecteds_num = min(selecteds_num, len(old_items), len(new_items))
                GreyWolves[i].difference_update(random.sample(old_items,k=selecteds_num))
                GreyWolves[i] = GreyWolves[i].union(random.sample(new_items, k=selecteds_num))
            else:       
                # exploration: get farther from leaders                                                                
                # select new items randomly which are not currentlly in Greywolves[i]            
                new_items = seq_set[i] - (GreyWolves[i].union(leader_set))
                old_items = leader_set.intersection(GreyWolves[i])
                #D = int(len(new_items)/3)                              
                D = len(GreyWolves[i] - leader_set)
                selecteds_num = math.ceil(abs(A * D))
                selecteds_num = min(selecteds_num, len(old_items), len(new_items))
                if(selecteds_num != 0):
                    GreyWolves[i].difference_update(random.sample(old_items,k=selecteds_num))
                    news = random.sample(new_items, k=selecteds_num)
                    GreyWolves[i] = GreyWolves[i].union(news)  
                    seq_set[i] -= set(news)                                                                           
        print(['At iteration '+ str(it)+ ' the best fitness is '+ str(Alpha_score) + '  time: ' +str(time.time() - timerStart)])
                    
    result = Alpha_score
    total_time = time.time() - timerStart 
    return result, total_time
    
