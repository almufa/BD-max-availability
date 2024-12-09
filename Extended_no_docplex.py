# -*- coding: utf-8 -*-
"""

@author: muffa
"""

import cplex
#instances to try the code    
results = []
n = 10
for d in [1]:
#call function RandomData(n,d)
    from random_data_gen import RandomData
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
    
    res = []
    for m in list(range(5,101,5)): 
        # Complete model
        c = cplex.Cplex()
        c.parameters.threads.set(1)
        c.parameters.advance.set(0)
        c.parameters.preprocessing.reduce.set(0)
        c.objective.set_sense(c.objective.sense.maximize)
        c.parameters.timelimit.set(3600)
        c.parameters.mip.limits.treememory.set(2000)
        #add variables x_k:
        x=[]
        for k in K:
            x.append(c.variables.get_num())
            var_names = "x_"+str(k)
            c.variables.add(obj=[0.0],lb=[0.0],ub=[1.0], types=["B"],
                          names = [var_names])
    
        #add variables thetha_q
        tetha=[]
        for q in range(len_Q):
            var_name = "theta_"+str(q)
            tetha.append(c.variables.get_num())
            c.variables.add(obj=[f[q]], lb=[0.0],
                          ub=[10e+10],
                          names=[var_name])
    
        #add variables y_o
        y_o =[]
        for q in range(len_Q):
            var_name = "y_o"+str(q)
            y_o.append(c.variables.get_num())
            c.variables.add(obj=[0.0],lb=[0.0],ub=[1.0],types=["B"],
                          names=[var_names])
    
        #add variables y_d
        y_d =[]
        for q in range(len_Q):
            var_name = "y_d"+str(q)
            y_d.append(c.variables.get_num())
            c.variables.add(obj=[0.0],lb=[0.0],ub=[1.0],types=["B"],
                          names=[var_names])   
    
        #add variables y_p
        y_p =[]
        for q in range(len_Q):
            var_name = "y_p"+str(q)
            y_p.append(c.variables.get_num())
            c.variables.add(obj=[0.0],lb=[0.0],ub=[1.0],types=["B"],
                          names=[var_names])
    
        #add constraint sum(x)=m
        thevars1=[]
        thecoefs1=[]
        for k in range(len(K)):
            thevars1.append(x[k])  
            thecoefs1.append(1.0)
        lhs = [cplex.SparsePair(thevars1,thecoefs1)]
        c.linear_constraints.add(lin_expr=lhs,senses=["L"],rhs=[m])
    
        #add constraint theta<= t_max
        for q in range(len_Q):
            thevars2=[]
            thecoefs2=[]
            thevars2.append(tetha[q])
            thecoefs2.append(1.0)
            c.linear_constraints.add([cplex.SparsePair(thevars2,thecoefs2)],
                                   senses=["L"],rhs=[t_max])
    
        #add constraint tetha <= to*yo+td*yd+tpd*yp
        for q in range(len_Q):
            thevars3=[]
            thecoefs3=[]
            thevars3.append(tetha[q])
            thecoefs3.append(1.0)
            thevars3.append(y_o[q])
            thecoefs3.append(-t_o)
            thevars3.append(y_d[q])
            thecoefs3.append(-t_d)
            thevars3.append(y_p[q])
            thecoefs3.append(-t_p)
            c.linear_constraints.add([cplex.SparsePair(thevars3,thecoefs3)],
                                   senses=["L"],rhs=[0.0])
        #add constraint sum(x[k] for k in N_o)>=y_o
        for q in range(len_Q):
            thevars4=[]
            thecoefs4=[]
            thevars4.append(y_o[q])
            thecoefs4.append(-1.0)
            for k in N_o[q]:
                thevars4.append(x[k])
                thecoefs4.append(1.0)
            lhs = [cplex.SparsePair(thevars4,thecoefs4)]
            c.linear_constraints.add(lin_expr=lhs,senses=["G"],rhs=[0])
        
        #add constraint sum(x[k] for k in N_d)>=y_d
        for q in range(len_Q):
            thevars4=[]
            thecoefs4=[]
            thevars4.append(y_d[q])
            thecoefs4.append(-1.0)
            for k in N_d[q]:
                thevars4.append(x[k])
                thecoefs4.append(1.0)
            lhs = [cplex.SparsePair(thevars4,thecoefs4)]
            c.linear_constraints.add(lin_expr=lhs,senses=["G"],rhs=[0])
        
        #add constraint sum(x[k] for k in N_p)>=y_p
        for q in range(len_Q):
            thevars4=[]
            thecoefs4=[]
            thevars4.append(y_p[q])
            thecoefs4.append(-1.0)
            for k in N_p[q]:
                thevars4.append(x[k])
                thecoefs4.append(1.0)
            lhs = [cplex.SparsePair(thevars4,thecoefs4)]
            c.linear_constraints.add(lin_expr=lhs,senses=["G"],rhs=[0])
        
        start_time =  c.get_time()
        c.solve()
        end_time = c.get_time()
        
        total_time = end_time-start_time
        
        solution = c.solution
        best_bnd = solution.get_objective_value()
        num_nodes = solution.progress.get_num_nodes_processed()
        gap = solution.MIP.get_mip_relative_gap()
        res.append([best_bnd,total_time,num_nodes,gap])
    
    results.append(res)




