import numpy as np


"""
Least Square Algorythm
"""


class LeastSquareAlgo :
    
    def __init__(self, A, y):
        self.A = A
        self.y = y
        self.x = None
        
    
    def solve(self):
        
        try:
            self.x = np.linalg.inv(self.A.T @ self.A ) @ self.A.T @ self.y
            
        except Exception as e:
            print(e)
            
    def __str__(self):
        #Shows the result
        text = f"Solution LSA: {self.x.T}T"
        return text
            
            
            
            
if __name__ == "__main__":
    A = np.matrix([[1, 2, 3],[3, 4, 3],[2, 1, 0]])
    y = np.matrix([3,4,5]).T
    solver = LeastSquareAlgo(A, y)
    solver.solve()
    print(solver)
    
    