import numpy as np
from queue import Queue
from copy import deepcopy
import time
import pandas as pd
import os

from node import Node


class BnBNormalAlgorythm:

    def __init__(self, y, A, lambda0=1, M=1000):

        self.y = y
        self.A = A
        self.n = y.shape[0]

        self.lambda0 = lambda0
        self.M = M

        self.pv_opt = None
        self.node_opt = None
        self.solving_time = None
        self.node_counter = 0
        self.verbose = None
        self.nodes_visited = []
        self.lower_lambda = -np.inf
        self.upper_lambda = np.inf

    def create_node(self, S0, S1, S):
        return Node(n=self.n, S0=S0, S1=S1, S=S, A=self.A, y=self.y, lambda0=self.lambda0, M=self.M)

    def show(self, *l, **k):
        if self.verbose:
            print(*l)

    def copy_S_from_node(self, u):
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


    def solve(self, verbose=True):
        self.verbose = verbose

        print("Initializing normal solver BnB")

        time_init = time.time()

        S = list(range(self.n))
        S0 = []
        S1 = []

        # crear queue
        q = Queue()
        u = self.create_node(S0, S1, S)
        q.put(u)


        # optimum
        u.calculate_obj()
        pv_opt = u.pu
        node_opt = deepcopy(u)
        self.show("First op:", pv_opt)


        def checking_node(name, v, pv_opt1, node_opt1):

            v.calculate_obj()
            self.show(f"{name} pvl:", round(v.pvl, 3))
            if not pv_opt1 <= v.pvl:
                q.put(v)
                self.show(f"{name} pu:", round(v.pu, 3))
                ex_node = node_opt1
                pv_opt1, node_opt1 = self.check_bound(node_opt1, v)
                self.show(f"Change node to {pv_opt1}\n" if ex_node != pv_opt1 else "", end="")
            return pv_opt1, node_opt1

        while not q.empty():

            # For every loop, is going to print a .
            if not verbose:
                if self.node_counter % self.n == 0:
                    print(".", flush=True, end="")

            # count node in counter
            self.node_counter += 1

            u = q.get()
            u.calculate_obj()
            self.nodes_visited.append(u)
            
            self.show(f"u node: S0: {u.S0} | S1: {u.S1} | Pvl: {u.pvl}\n")
            self.show(f"Pv opt: {round(pv_opt, 3)}")

            # if the node is leaf
            if u.level == self.n or len(u.S) == 0:
                continue

            # None zero branch
            S_v, S1_v, S0_v = self.copy_S_from_node(u)
            index = S_v.pop(0)
            S1_v.append(index)
            v1 = self.create_node(S0_v, S1_v, S_v)
            pv_opt, node_opt = checking_node("V1", v1, pv_opt, node_opt)

            # Zero branch
            S_v, S1_v, S0_v = self.copy_S_from_node(u)
            index = S_v.pop(0)
            S0_v.append(index)
            v0 = self.create_node(S0_v, S1_v, S_v)
            pv_opt, node_opt = checking_node("V0", v0, pv_opt, node_opt)

            self.show("Queue:\t", list(q.queue), "\n")

        finish_time = time.time()

        self.solving_time = finish_time - time_init

        self.pv_opt = pv_opt
        self.node_opt = node_opt

        print(f"""\n
            {"#" * 60}
            \tSolver finished in:\t{self.solving_time} s
            {"#" * 60}""")

        print()
        return pv_opt, node_opt, self.node_counter, self.nodes_visited


if __name__ == "__main__":
    np.random.seed(42)
    n = 10
    A = np.random.randint(-10, 10, (n, n))
    y = np.random.randint(-10, 10, (n, 1))
    
    df_A = pd.DataFrame(A)
    df_A.to_csv(os.path.join('files', 'matrix.csv'), index=False, header=False)
    df_y = pd.DataFrame(y)
    df_y.to_csv(os.path.join('files', 'y.csv'), index=False, header=False)    
    
    np.random.seed(42)
    n = 10
    A = np.random.randint(-10, 10, (n, n))
    y = np.random.randint(-10, 10, (n, 1))

    lambda_0 = 1
    M0 = 100
    solver = BnBNormalAlgorythm(y, A, lambda0=lambda_0, M=M0)
    pv, node_opt, num_nodes = solver.solve(verbose=False)

    print("Solution Node")
    print(f"Nodes visited {num_nodes}/{2**(n) + 1}")
    print(node_opt)
    print(node_opt.x)