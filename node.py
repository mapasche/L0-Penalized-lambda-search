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
        self.verbose = False

        self.u = None
        self.omega = None
        self.pvl = 0
        self.pu = None  # corresponds to obj func value in real problem with x
        self.x = None

        self.lambda_ub = None
        self.lambda_lb = None

    def show(self, *l, **k):
        if self.verbose:
            print(*l)

    def calculate_obj(self,):
        self.pvl, self.x = relax_problem(
            self.y, self.A, S0=self.S0, S1=self.S1, S=self.S, lambda0=self.lambda0, M=self.M)
        self.pu = evaluation_main(self.y, self.A, self.x, self.lambda0)
        self.u = self.y - self.A @ self.x
        

    def get_lambda_interval(self, verbose = False):
        self.verbose = verbose
        
        self.show(self)
        self.show(self.x)
        
        self.omega = float( 0.5 * np.linalg.norm(self.y - self.A @ self.x)**2 - \
            0.5 * np.linalg.norm(self.y)**2 +  \
            0.5 * np.linalg.norm(self.y-self.u)**2 )
        for i in self.S1:
            self.omega += float( self.M*abs(self.A[:, i] @ self.u) )

        class Au:
            def __init__(self, value, index):
                self.value = float(value)
                self.index = index
            def __str__(self):
                return str("val : " + str(round(self.value, 4)) + ", idx : "+str(self.index))
            def __repr__(self):
                return str(self)

        # decreasing
        self.show(self.u)
        test = sorted([Au(self.M*abs(self.A[:, i] @ self.u), i)
                       for i in self.S], reverse=True, key=lambda x: x.value)
        
        

        def calc_bound_leaf_node ():
            constant1 = self.omega
            constant2 = float( len(self.S1) - np.count_nonzero(self.x) )
            self.show("Calc bound leaf constant1:", constant1, "constant 2:",constant2)
            if constant2 != 0:
                value =  constant1 / constant2
                if 0 > constant2:
                    # negative
                    self.show("Condition to refute: lambda* < ", round(value, 3) )
                    possible_lower_bound = value
                    possible_upper_bound = np.inf
                else:
                    # positive
                    self.show("Condition to refute:", round(value, 3), "< Lambda*")
                    possible_upper_bound = value
                    possible_lower_bound = -np.inf
                return (possible_lower_bound, possible_upper_bound)
            else: 
                if self.omega < 0:
                    self.show("Omega:", self.omega)
                    raise Exception("Omega is negative when the refute at Omega > lambda * 0")  
                value = (-np.inf, np.inf)
            return value
            
        
        def calc_bound(j):
            constant1 = float( self.omega + \
                np.sum([test[i].value for i in range(j + 1)]) )
            constant2 = float( len(self.S1)-np.count_nonzero(self.x)+j+1 )
            self.show(constant1)
            
            if constant2 != 0:
                value = constant1 / constant2
                if 0 > constant2:
                    # negative
                    self.show("Condition to refute: Lambda* <", round(value, 3))
                    possible_lower_bound = value
                    possible_upper_bound = np.inf
                else:
                    # positive
                    self.show("Condition to refute:", round(value, 3), "< Lambda*")
                    possible_upper_bound = value
                    possible_lower_bound = -np.inf
                return (possible_lower_bound, possible_upper_bound)
            
            else:
                #when constant2 is zero we cannot check the condition
                if constant1 < 0:
                    print("Omega + Sum M|ai u|:", constant1)
                    raise Exception("The sum of omega and sumatory is negative when the refute is at Omega + Sum < lambda * 0") 
                else:
                    self.show("No condition, we have Omega + Sum > 0") 
                return (-np.inf, np.inf)
            
        
        def check_if_better_bound (j, possible_lb, possible_ub, lower_bound, upper_bound):
            if possible_lb == -np.inf:
                if not (j == -1 or j == len(test) - 1 ):
                    if possible_ub < test[j].value:
                        self.show("Update upper bound")
                        upper_bound = min(upper_bound, possible_ub)
                else:
                    self.show("Lambda is", self.lambda0, "and the possible upper bound is", possible_ub)
                    if possible_ub < self.lambda0:
                        raise Exception("Imposible condition: possible ub < lambda")
                    ex_upper = upper_bound
                    upper_bound = min(upper_bound, possible_ub)
                    if ex_upper != upper_bound:
                        self.show("Upper bound changed!")
                        self.show("[", lower_bound, ",", upper_bound, "]")
            
            if possible_ub == np.inf:
                if not (j == len(test) - 1 or j == -1):
                    if possible_lb > test[j+1].value:
                        self.show("Update lower bound")
                        lower_bound = max(lower_bound, possible_lb)
                else:
                    self.show("Lambda is", self.lambda0, "and the possible lower bound is", possible_lb)
                    if possible_ub < self.lambda0:
                        raise Exception("Imposible condition: lambda < possible lb")
                    ex_lower = lower_bound
                    lower_bound = min(lower_bound, possible_ub)
                    if ex_lower != lower_bound:
                        self.show("Change lower bound!")
                        self.show("[", lower_bound, ",", upper_bound, "]")
                    
            return lower_bound, upper_bound
            
    
        

        
        #find j0. argmax M|ai^T u|
        j0 = -1
        for i in range(len(test)):
            if (test[i].value >= self.lambda0):
                j0 = i
                
        self.show(test)
        self.show("omega:", self.omega)
        self.show("J0:", j0)
        self.show("Lambda:", self.lambda0)
        
        
        # If we are in the solution node
        if len(test) == 0:
            self.show("Leaf Node")
            lower_bound, upper_bound = calc_bound_leaf_node()
        else:   
            if j0 == len(self.S) - 1:
                self.show("First interval", "[", -np.inf, ",", test[j0].value, "]")
            elif j0 == -1:
                self.show("First interval", "[", test[0].value, ",", np.inf, "]")
            else:
                self.show("First interval", "[", test[j0 + 1].value, ",", test[j0].value, "]")
            
            lower_bound = -np.inf
            upper_bound = np.inf
            #searches for the smaller bound
            for j in range(j0, len(test)):
                self.show("J:", j)
                if j == len(self.S) - 1:
                    self.show("If Lambda* in [", -np.inf, ",", test[j].value, "]")
                elif j == -1:
                    self.show("If Lambda* in [", test[0].value, ",", np.inf, "]")
                else:
                    self.show("If Lambda* in [", test[j + 1].value, ",", test[j].value, "]")
                
                possible_lower_bound, possible_upper_bound = calc_bound(j)
                lower_bound, upper_bound = check_if_better_bound(j,
                    possible_lower_bound, possible_upper_bound, 
                    lower_bound, upper_bound
                )
                self.show("Possible interval: [", possible_lower_bound, ", ", possible_upper_bound, "]")

            
            #now the other set of intervals
            for j in range(j0, -2, -1):
                self.show("J:", j)
                if j == len(self.S) - 1:
                    self.show("If Lambda* in [", -np.inf, ",", test[j].value, "]")
                elif j == -1:
                    self.show("If Lambda* in [", test[j].value, ",", np.inf, "]")
                else:
                    self.show("If Lambda* in [", test[j + 1].value, ",", test[j].value, "]")
                
                possible_lower_bound, possible_upper_bound = calc_bound(j)
                lower_bound, upper_bound = check_if_better_bound(j,
                    possible_lower_bound, possible_upper_bound, 
                    lower_bound, upper_bound
                )
                self.show("Possible interval: [", possible_lower_bound, ", ", possible_upper_bound, "]")

        self.show(self,"[", lower_bound, ",", upper_bound, "]", "\n\n")
        self.lambda_lb, self.lambda_ub = lower_bound, upper_bound
        return lower_bound, upper_bound




    def __le__(self, other) -> bool:
        return self.pvl <= other.pvl

    def __str__(self) -> str:
        text = f"S0: {self.S0} | S1: {self.S1} | S: {self.S}\nPvl: {round(self.pvl, 3)}\n"
        return text

    def __repr__(self) -> str:
        text = f"S0: {self.S0} | S1: {self.S1} | Pvl: {round(self.pvl, 3)}"
        return text