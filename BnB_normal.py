import numpy as np
import cvxpy as cp 
from queue import Queue
from copy import deepcopy
import time


verbose_relax = False

def evaluation_main (y, A, x, alpha = 1):
    value = 0.5 * np.linalg.norm( y - A @ x ) ** 2 + alpha * np.count_nonzero(x)    
    return value



def relax_problem (y, A, S0, S1, S, alpha = 1, M = 1000000):
    
    n = y.shape[0]
    X = cp.Variable((n, 1))
    obj =  0.5 * cp.sum_squares(y - A @ X ) 
    obj +=  alpha / M * cp.norm(X[S], 1)
    obj += alpha * len(S1)
    
    if len(S0) == 0:
        constraint = [X <= M, -X <= M] 
    else:
        constraint = [X[S0] == 0, X <= M, -X <= M] 

    prob = cp.Problem(cp.Minimize( obj ), constraint)
    result = prob.solve(verbose = verbose_relax, solver = cp.ECOS, abstol=1e-4, reltol=1e-4)
    #eliminate 0
    x_vector = X.value
    x_vector = np.where(abs(x_vector) < 1e-7, 0, x_vector )
    return result, x_vector









class Node:
    
    def __init__(self, n, S0, S1, S):
        self.level = n - len(S)
        self.S0 = S0
        self.S1 = S1
        self.S = S
        self.pvl = 0
        self.pu = None       #corresponds to obj func value in real problem with x
        self.x = None        
    
    def calculate_obj (self, y, A, alpha0, M0):
        self.pvl, self.x = relax_problem(y, A, S0 = self.S0, S1 = self.S1, S = self.S, alpha = alpha0, M = M0)
        self.pu =  evaluation_main(y, A, self.x, alpha0)
        
    def __le__ (self, other) -> bool:
        return self.pvl <= other.pvl
    
    def __str__(self) -> str:
        text = f"S0: {self.S0} | S1: {self.S1} | S: {self.S}\nPvl: {self.pvl}\n"
        return text
    
    def __repr__(self) -> str:
        text = f"S0: {self.S0} | S1: {self.S1} | Pvl: {self.pvl}"
        return text








class BnBNormalAlgorythm:
    
    def __init__ (self, y, A, alpha = 1, M = 1000):
        
        self.y = y
        self.A = A
        self.n = y.shape[0]
        
        self.alpha = alpha
        self.M = M
        
        self.pv_opt = None
        self.node_opt = None
        self.solving_time = None
        self.node_counter = 0
        self.verbose = None
        
        

    def create_node(self, S0, S1, S):
        return Node(self.n, S0, S1, S)
    
    def calculate_values(self, u):
        u.calculate_obj(y, A, self.alpha, self.M)
        
    def show(self, *l, **k):
        if self.verbose:
            print(*l)
            
    def copy_S_from_node (self, u):
        S = deepcopy(u.S)
        S1 = deepcopy(u.S1)
        S0 = deepcopy(u.S0)
        return S, S1, S0
    
    
    
    
    
    
    def check_bound(self, u, v):
        """
        Recieves Node objects
        """
        if v.pu <= u.pu:
            return v.pu, v
        else:
            return u.pu, u
    
    
    


    def solve (self, verbose = True):
        self.verbose = verbose
        
        print("Initializing normal solver BnB")
        
        time_init = time.time()
        
        S = list(range(self.n))
        S0 = []
        S1 = []
        
        #crear queue
        q = Queue()
        u = self.create_node(S0, S1, S)
        q.put(u)

        #optimum
        self.calculate_values(u)
        pv_opt = u.pu
        node_opt = deepcopy(u)
        self.show("First op:", pv_opt)
        
        
        def checking_node (name, v, pv_opt1, node_opt1):
            self.calculate_values(v)
            self.show(f"{name} pvl:", v.pvl)
            if not pv_opt1<= v.pvl:
                q.put(v)
                self.show(f"{name} pu:", v.pu)       
                pv_opt1, node_opt1 = self.check_bound(node_opt1, v)
            return pv_opt1, node_opt1
            


        while not q.empty():
            
            
            #For every loop, is going to print a .
            if not verbose:
                if self.node_counter % self.n == 0:
                    print(".", flush=True, end="")
            
            
            #count node in counter
            self.node_counter += 1            
            
            
            u = q.get()
            self.calculate_values(u)
            self.show(f"u node: S0: {u.S0} | S1: {u.S1} | Pvl: {u.pvl}\n")
            
            #if the node is leaf
            if u.level == self.n  or len(u.S) == 0:
                continue
            
            
            

            #None zero branch
            S_v, S1_v, S0_v = self.copy_S_from_node(u)            
            index = S_v.pop(0)
            S1_v.append(index)
            v1 = self.create_node(S0_v, S1_v, S_v)
            pv_opt, node_opt = checking_node("V1", v1, pv_opt, node_opt)
            
            
            
            #Zero branch
            S_v, S1_v, S0_v = self.copy_S_from_node(u)
            index = S_v.pop(0)
            S0_v.append(index)  
            v0 = self.create_node(S0_v, S1_v, S_v)
            pv_opt, node_opt = checking_node("V0", v0, pv_opt, node_opt)
            
            self.show("Queue:\t",list(q.queue), "\n")

        
        finish_time = time.time()
        
        self.solving_time = finish_time - time_init
        
        self.pv_opt = pv_opt
        self.node_opt = node_opt
        

        print(f"""\n
            {"#" * 60}
            \tSolver finished in:\t{self.solving_time} s
            {"#" * 60}""")
        
        print()
        return pv_opt, node_opt, self.node_counter





if __name__ == "__main__":
    np.random.seed(420)
    n = 15
    A = np.random.randint(-10, 10, (n, n))
    y = np.random.randint(-10, 10, (n, 1))
    
    lambda_0 = 1
    M0 = 100    
    
    solver = BnBNormalAlgorythm(y, A, alpha = lambda_0, M = M0)  
    
    pv, node, num_nodes = solver.solve(verbose=True)
    
    print("Solution Node")
    print(f"Nodes visited {num_nodes}/{2**(n) + 1}") #remember to count node 0
    print(node)