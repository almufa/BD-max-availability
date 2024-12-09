# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:09:42 2020

@author: muffa
"""
import networkx as nx
import numpy as np
import copy 
import math
from collections import defaultdict
import time
import cplex
from cplex.callbacks import LazyConstraintCallback 


#------------------------------------------------------------------------------
class WorkerLP():   
    def __init__(self,t_max,t_o,t_d,t_p,N_o,N_d,N_p,f,len_Q):
         
        self.t_max = t_max
        self.t_o = t_o
        self.t_d = t_d
        self.t_p = t_p
        self.f =f
        self.N_o = N_o
        self.N_d = N_d
        self.N_p = N_p
        self.len_Q = len_Q
        
        global worker_time1, n_sub
        n_sub = 0
        worker_time1 = 0
        
    def sol(self, xsol):
        
        t_max = self.t_max
        N_o = self.N_o
        N_d = self.N_d
        N_p = self.N_p
        len_Q =self.len_Q
        
        #Update the coeficients in the WorkerLP objective function:

        x_hat_No = [(sum(xsol[k] for k in N_o[q])) for q in range(len_Q)]
        x_hat_Nd = [(sum(xsol[k] for k in N_d[q])) for q in range(len_Q)]
        x_hat_Np = [(sum(xsol[k] for k in N_p[q])) for q in range(len_Q)]
        
        #Initiate dual values
        alpha = []
        gamma = []
        beta = []
        rho = []
        phi = []
        mu = []
        sigma = []
        delta = []
        cond = []
        dual_obj=[]

        for q in range(len_Q):
            cond.append(t_max-min(x_hat_No[q]*t_o,t_o)-min(x_hat_Nd[q]*t_d,t_d)-min(x_hat_Np[q]*t_p,t_p))
            if cond[q]<0:
                beta.append(0)
                phi.append(0)
                sigma.append(0)
                delta.append(0)
                alpha.append(f[q])
                if x_hat_No[q]==0:
                    gamma.append(t_o*f[q])
                else: gamma.append(0)
                if x_hat_Nd[q] ==0:
                    mu.append(t_d*f[q])
                else: mu.append(0)
                if x_hat_Np[q]==0:
                    rho.append(t_p*f[q])
                else: rho.append(0)
            else:
                betacf = f[q]
                beta.append(betacf)
                alpha.append(f[q]-betacf)
                if x_hat_No[q]<=1:
                    gamma.append(t_o*betacf)
                    phi.append(0)
                else:
                    gamma.append(0)
                    phi.append(t_o*betacf)
                if x_hat_Nd[q]<=1:
                    mu.append(t_d*f[q])
                    sigma.append(0)
                else:
                    mu.append(0)
                    sigma.append(t_d*betacf)
                if x_hat_Np[q]<=1:
                    rho.append(t_p*betacf)
                    delta.append(0)
                else:
                    rho.append(0)
                    delta.append(t_p*betacf)
        
            #Compute Dual objective value
            dual_obj.append(t_max*alpha[q]+x_hat_No[q]*gamma[q]+x_hat_Nd[q]*mu[q]+(
            x_hat_Np[q]*rho[q]+phi[q]+sigma[q]+delta[q]))
        
        
        global n_sub
        n_sub = n_sub + 1
        
        return dual_obj, cond
    
    def seperate(self, xsol, etaSol, x, eta,worker_obj, dual_MW):
        "This method seperates the violated cuts"
        t_max = self.t_max
        N_o = self.N_o
        N_d = self.N_d
        N_p = self.N_p
        len_Q = self.len_Q
        violatedCutFound  = False
        
        t1 = time.time()
        
        alphaSol = dual_MW[0]
        phiSol = dual_MW[1]
        muSol = dual_MW[2]
        sigmaSol = dual_MW[3]
        deltaSol = dual_MW[4]
        gammaSol = dual_MW[5]
        rhoSol = dual_MW[6]
        
        
        
        # A violated cut is available iff (subproblem.obj_value < \eta)
        if sum(worker_obj) < etaSol-1e-4:
            
            # a (key, pair) dictionary that sets value to 0 by default
            ind_val_map = defaultdict(lambda: 0.0)

            # for every ind 'x[k]', accumulate our coefficient into the dictionary
            for q in range(len_Q):
                for k in N_o[q]:
                    ind_val_map[x[k]] += -gammaSol[q]
                for k in N_d[q]:
                    ind_val_map[x[k]] += -muSol[q]
                for k in N_p[q]:
                    ind_val_map[x[k]] += -rhoSol[q]

            # we now have a dict of the form (ind -> coef) 
            thecoefs1 = [1.0]
            thevars1 = [eta[0]]
            for ind, coef in ind_val_map.items():
                thevars1.append(ind)
                thecoefs1.append(coef)
            cutlhs = cplex.SparsePair(thevars1,thecoefs1)
                        
            #compute the rhs of the cut 
            rhs = []
            for q in range(len_Q):
                rhs.append(alphaSol[q]*t_max+phiSol[q]+sigmaSol[q]+deltaSol[q])
            cutrhs = sum(rhs)
            
            self.cutrhs =cutrhs
            self.cutlhs = cutlhs
            violatedCutFound = True
        
        deltat = time.time()-t1
        
        global worker_time1
        worker_time1 += deltat
        
        return violatedCutFound
#------------------------------------------------------------------------------
class xbar():
    "This method stores and updates the core point"
    def __init__(self,K,m):
       xbar = []
       for k in range(len(K)):
           xbar.append(m/len(K)) 
       self.xbar = xbar    
    
    def update(self,xsol):
        xbar = self.xbar
        for k in range(len(K)):
            xbar[k] = (xbar[k]+xsol[k])/2
        
        return xbar        
#------------------------------------------------------------------------------
class ParetoMW():
    "This class builds the LP to generate Pareto optimal cuts"
    def __init__(self,t_max,t_o,t_d,t_p,N_o,N_d,N_p,f,len_Q):
        
        self.t_max = t_max
        self.t_o = t_o
        self.t_d = t_d
        self.t_p = t_p
        self.f =f
        self.len_Q = len_Q
        self.N_o = N_o
        self.N_d = N_d
        self.N_p = N_p
        self.len_Q = len_Q
        
        global worker_time2
        worker_time2 = 0
        
    def sol(self, xsol,x_bar,cond):
        """ This method updates and solves the MW LP and returns 
        the dual values of the MW LP """
        
        N_o = self.N_o
        N_d = self.N_d
        N_p = self.N_p
        len_Q =self.len_Q
        
        t1= time.time()
        #Retreive coefficients in worker LP to add to opt(SPD) constraint:
        x_hat_No = [(sum(xsol[k] for k in N_o[q])) for q in range(len_Q)]
        x_hat_Nd = [(sum(xsol[k] for k in N_d[q])) for q in range(len_Q)]
        x_hat_Np = [(sum(xsol[k] for k in N_p[q])) for q in range(len_Q)]
          
        #Update the coeficients in the MW objective function:
        x_bar_No = [(sum(x_bar[k] for k in N_o[q])) for q in range(len_Q)]
        x_bar_Nd = [(sum(x_bar[k] for k in N_d[q])) for q in range(len_Q)]
        x_bar_Np = [(sum(x_bar[k] for k in N_p[q])) for q in range(len_Q)]
        
        
        alpha = []
        gamma = []
        beta = []
        rho = []
        phi = []
        mu = []
        sigma = []
        delta = []
        for q in range(len_Q):
            if cond[q]<0:
                beta.append(0)
                phi.append(0)
                sigma.append(0)
                delta.append(0)
                alpha.append(f[q])
                gamma.append(0)
                mu.append(0)
                rho.append(0)
                
            else:
                beta.append(f[q])
                alpha.append(0)
                if x_hat_No[q] == 0:
                    gamma.append(t_o*f[q])
                    phi.append(0)
                elif x_hat_No[q] == 1 and x_bar_No[q] <=1:
                    gamma.append(t_o*f[q])
                    phi.append(0)
                else:
                    gamma.append(0)
                    phi.append(t_o*f[q])
                if x_hat_Nd[q] == 0:
                    mu.append(t_d*f[q])
                    sigma.append(0)
                elif x_hat_Nd[q] ==1 and x_bar_Nd[q] <=1 :
                    mu.append(t_d*f[q])
                    sigma.append(0)
                else:
                    mu.append(0)
                    sigma.append(t_d*f[q])
                if x_hat_Np[q] == 0:
                    rho.append(t_p*f[q])
                    delta.append(0)
                elif x_hat_Np[q] == 1 and x_bar_Np[q] <=1:
                        rho.append(t_p*f[q])
                        delta.append(0)
                else:
                    rho.append(0)
                    delta.append(t_p*f[q])
        
        dual_MW = [alpha,phi,mu,sigma,delta,gamma,rho]
        
        
        deltat = time.time()-t1
        global worker_time2
        worker_time2 += deltat
        
        
        return dual_MW
    
#------------------------------------------------------------------------------        
class LazyConsCallback(LazyConstraintCallback):
    "Callback class to add cuts"
    def __call__(self):
      x = self.x
      eta = self.eta
      int_point = self.int_point
      workerLP = self.workerLP
      mw = self.mw

      
      #Get the current solution from the MasterLP
      xsol = []
      for i in range(len(x)):
          xsol.append([])
          xsol[i]=self.get_values(x[i])
          
      etaSol = self.get_values(eta[0])
    
      #Retrieve the obj value of the workerL   
      worker_obj = workerLP.sol(xsol)[0]
           
      #Update the core point
      
      x_bar = int_point.update(xsol)
      
      #retreive the CF condition to solve the MW
      cond = workerLP.sol(xsol)[1]

      #Retrieve dual values of MW
      
      dual_MW=mw.sol(xsol, x_bar, cond)      
        
      
      #If  cut is violated then add cut
      if workerLP.seperate(xsol, etaSol, x, eta, worker_obj, dual_MW):
          self.add(workerLP.cutlhs, sense="L", rhs=workerLP.cutrhs)
      
#------------------------------------------------------------------------------
def MasterILP(cpx,x,eta,K,m,len_Q):
    "This function creates the master ILP"
    
    cpx.objective.set_sense(cpx.objective.sense.maximize)
    
    #add variables x_k:
    for k in K:
        x.append(cpx.variables.get_num())
        var_names = "x_"+str(k)
        cpx.variables.add(obj=[0.0],lb=[0.0],ub=[1.0], types=["B"],
                          names = [var_names])
    #add variable eta
    for i in range(1):
        eta.append(cpx.variables.get_num())
        cpx.variables.add(obj=[1.0],lb=[0.0],ub=[10e+20],names=["eta"])
    
    #add number of facilities constraint
    thevars1=[]
    thecoefs1=[]
    for k in range(len(K)):
        thevars1.append(x[k])  
        thecoefs1.append(1.0)
    lhs = [cplex.SparsePair(thevars1,thecoefs1)]
    cpx.linear_constraints.add(lin_expr=lhs,senses=["L"],rhs=[m])
    
    #add redundant constraint
    thevars2 =[eta[0]]
    thecoefs2 =[1.0]
    for k in range(len(K)):
        thevars2.append(x[k])  
        thecoefs2.append(1.0)
    lhs2 = [cplex.SparsePair(thevars2,thecoefs2)]
    cpx.linear_constraints.add(lin_expr=lhs2,senses=["G"],rhs=[0])
              
#------------------------------------------------------------------------------
    
def BD_location(len_Q,t_max,t_o,t_d,t_p,m,N_o,N_d,N_p,K,f,nodeLim=None):
    "This function is called to solve the problem"
    #create master ILP
    cpx = cplex.Cplex()
    x = []
    eta = []
    int_point = xbar(K,m)
    MasterILP(cpx,x,eta,K,m,len_Q)
    
    
    #Create workerLP for Benders cuts
    workerLP = WorkerLP(t_max,t_o,t_d,t_p,N_o,N_d,N_p,f,len_Q)
    
    #Create the MW LP:
    mw = ParetoMW(t_max,t_o,t_d,t_p,N_o,N_d,N_p,f,len_Q)
    
    
    # For using callbacks, we need traditional B&C search
    cpx.parameters.mip.strategy.search.set(
            cpx.parameters.mip.strategy.search.values.traditional)
    
    # Set the node limit if given
    if not nodeLim is None:
        cpx.parameters.mip.limits.nodes.set(nodeLim)
    
    # Switch off "advanced" start
    cpx.parameters.advance.set(0)

    # Install the lazy constraint callback
    cpx.register_callback(LazyConsCallback)
    LazyConsCallback.x = x
    LazyConsCallback.eta = eta
    LazyConsCallback.workerLP = workerLP
    LazyConsCallback.int_point = int_point
    LazyConsCallback.mw = mw
    
    cpx.parameters.timelimit.set(3600)
    cpx.parameters.mip.limits.treememory.set(2000)
    #start time
    start_time = cpx.get_time()
    
    # Solve model
    cpx.solve()
    
    #end time
    end_time = cpx.get_time()
    
    total_time = end_time-start_time
    
    solution = cpx.solution
    best_bnd = solution.MIP.get_best_objective()
    nb_cuts = solution.MIP.get_num_cuts(solution.MIP.cut_type.user)
    num_nodes = solution.progress.get_num_nodes_processed()
    gap = solution.MIP.get_mip_relative_gap()
    global worker_time1, worker_time2
    worker_time=worker_time1+worker_time2
    
    return best_bnd,nb_cuts,total_time,worker_time,num_nodes,n_sub,gap
   
#------------------------------------------------------------------------------
def RandomData(n,d):
    "This function generates a random graph and creates the sets needed for the problem"

    #create Graph
    G = nx.generators.random_graphs.fast_gnp_random_graph(n,0.6,seed=10,directed=True)
    dist = []
    rnd = np.random
    rnd.seed(10)
    #add distances between nodes
    for e in G.edges():
        a = math.ceil(rnd.random_sample()*20)
        G.add_weighted_edges_from([(e[0],e[1],a)])
        dist.append([e[0],e[1],a])
 
    
    #Create all paths between pairs of nodes
    OD_no_f =[]
    ODD = dict(nx.all_pairs_dijkstra_path(G))
    for i in range(n):
        for j in range(n):
            if i!=j:
                 OD_no_f.append(ODD[i][j])
                 


    #Create set of potential locations
    K = list(range(n))
    
    
    #Create set of potential locations within coverage distance from origin
    cov_dist_origin = d #Coverage distance around origin nodes
    cov_dist_destination = d


    global cov_origin
    cov_origin = [] #List of nodes within the coverage distance
    cov_dest = []
    for i in range(n):
        dict_org = nx.single_source_dijkstra(G,i,cutoff=cov_dist_origin,
                                                weight='weight')
        cov_origin.append(sorted(list(dict_org[0].keys())))
        dict_dest = nx.single_source_dijkstra(G,i,cutoff=cov_dist_destination,
                                              weight='weight')
        cov_dest.append(sorted(list(dict_dest[-1].keys())))
        
    
    origins=[] #List of origins of each q
    destinations = []
    for i in range(len(OD_no_f)):
        origins.append(OD_no_f[i][0])
        destinations.append(OD_no_f[i][-1])
    
   
    
    N_o = [cov_origin[i] for i in origins] #list of potential locations within coverage N_o[q]
    N_d = [cov_dest[i]for i in destinations]
    
                
           
    global OD_just_path
    #Create set of potential locations on paths N_p[q]
    OD_just_path = copy.deepcopy(OD_no_f) #list with path nodes only
    f =[]
    N_p=[]
    for i in range(len(OD_just_path)):
        OD_just_path[i].remove(OD_just_path[i][0])
        OD_just_path[i].remove(OD_just_path[i][-1])
        N_p.append(sorted(OD_just_path[i]))
        f.append(math.ceil(rnd.random_sample()*100))
        

    
    len_Q = len(OD_no_f)
    

    return(len_Q,N_o,N_d,N_p,K,f)

#------------------------------------------------------------------------------

results =[]

n = 300
d = 3

data = RandomData(n,d)

t_max = 12
t_o = 5
t_d = 1
t_p = 2
len_Q = data[0]
N_o = data[1]
N_d = data[2]
N_p = data[3]
K = data[4]
f = data[5]

m = 60

res = BD_location(len_Q,t_max,t_o,t_d,t_p,m,N_o,N_d,N_p,K,f,nodeLim=None)
          
    
