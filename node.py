import numpy as np
from utils import evaluation_main, relax_problem

class Node:

    def __init__(self, n, S0, S1, S, A, y, lambda0, M):
        self.A = A
        self.y = y
        self.level = n - len(S)
        self.S0 = S0
        self.S1 = S1
        self.S = S
        self.lambda0 = lambda0
        self.M = M
        self.n = n

        self.u = None
        self.omega = None
        self.pvl = 0
        self.pu = None  # corresponds to obj func value in real problem with x
        self.x = None

        self.upper_lambda = None
        self.lower_lambda = None


    def calculate_obj(self,):
        self.pvl, self.x = relax_problem(
            self.y, self.A, S0=self.S0, S1=self.S1, S=self.S, lambda0=self.lambda0, M=self.M)
        self.pu = evaluation_main(self.y, self.A, self.x, self.lambda0)
        self.u = self.y - self.A @ self.x
        

    def get_lambda_interval(self):
        self.omega = 0.5 * np.linalg.norm(self.y - self.A @ self.x)**2 - \
            0.5 * np.linalg.norm(self.y)**2 +  \
            0.5 * np.linalg.norm(self.y-self.u)**2
            
        for i in self.S1:
            self.omega += self.M*abs(self.A[:, i] @ self.u)

        class Au:
            def __init__(self, value, index):
                self.value = float(value)
                self.index = index
            def __str__(self):
                return str("val : " + str(round(self.value, 4)) + ", idx : "+str(self.index))
            def __repr__(self):
                return str(self)

        # decreasing
        test = sorted([Au(self.M*abs(self.A[:, i] @ self.u), i)
                       for i in self.S], reverse=True, key=lambda x: x.value)

        

        def calc_bound(j):
            constant1 = self.omega + \
                np.sum([test[i].value for i in range(j + 1)])
            constant2 = len(self.S1)-np.count_nonzero(self.x)+j+1
            
            if constant2 != 0:

                if not 0 <= constant2:
                    # negative
                    possible_lower_bound = constant1 / constant2
                    possible_upper_bound = np.inf
                else:
                    # positive
                    possible_upper_bound = constant1 / constant2
                    possible_lower_bound = -np.inf
                return (possible_lower_bound, possible_upper_bound)
            
            else:
                #when constant2 is zero we cannot check the condition
                return (-np.inf, np.inf)
            
        

        
        #find j0. argmax M|ai^T u|
        j0 = -1
        for i in range(len(test)):
            if (test[i].value >= self.lambda0):
                j0 = i
                
        print("omega:", self.omega)
        print("J0:", j0)
        print("First interval",
              "[", test[j0 + 1].value, ",", test[j0].value, "]")
        
        lower_bound = -np.inf
        upper_bound = np.inf
        #searches for the smaller bound
        for j in range(j0, len(test)):
            print("J:", j)
            if j == len(self.S) - 1:
                print("[", -np.inf, ",", test[j].value, "]")
            else:
                print("[", test[j + 1].value, ",", test[j].value, "]")
            
            possible_lower_bound, possible_upper_bound = calc_bound(j)
            #print("lower:", possible_lower_bound, "upper:", possible_upper_bound)
            if possible_lower_bound == -np.inf:
                if possible_upper_bound < test[j].value:
                    upper_bound = min(upper_bound, possible_upper_bound)
            
            if possible_upper_bound == np.inf:
                if possible_lower_bound > test[j].value:
                    lower_bound = max(lower_bound, possible_lower_bound)
        
        
        #now the other set of intervals
        for j in range(j0, -1, -1):
            print("J:", j)
            if j == 0:
                print("[", test[j].value, ",", np.inf, "]")
            else:
                print("[", test[j + 1].value, ",", test[j].value, "]")
            
            possible_lower_bound, possible_upper_bound = calc_bound(j)
            print("lower:", possible_lower_bound, "upper:", possible_upper_bound)
            if possible_lower_bound == -np.inf:
                if possible_upper_bound < test[j].value:
                    upper_bound = min(upper_bound, possible_upper_bound)
            
            if possible_upper_bound == np.inf:
                if possible_lower_bound > test[j].value:
                    lower_bound = max(lower_bound, possible_lower_bound)

            
                    
        
        print("[", lower_bound, ",", upper_bound, "]")






    def __le__(self, other) -> bool:
        return self.pvl <= other.pvl

    def __str__(self) -> str:
        text = f"S0: {self.S0} | S1: {self.S1} | S: {self.S}\nPvl: {self.pvl}\n"
        return text

    def __repr__(self) -> str:
        text = f"S0: {self.S0} | S1: {self.S1} | Pvl: {self.pvl}"
        return text