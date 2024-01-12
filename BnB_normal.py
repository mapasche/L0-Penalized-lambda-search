import numpy as np
import cvxpy as cp
from queue import Queue
from copy import deepcopy
import time


verbose_relax = False


def evaluation_main(y, A, x, lambda0=1):
    value = 0.5 * np.linalg.norm(y - A @ x) ** 2 + \
        lambda0 * np.count_nonzero(x)
    return value


def relax_problem(y, A, S0, S1, S, lambda0=1, M=1000000):

    n = y.shape[0]
    X = cp.Variable((n, 1))
    obj = 0.5 * cp.sum_squares(y - A @ X)
    obj += lambda0 / M * cp.norm(X[S], 1)
    obj += lambda0 * len(S1)

    if len(S0) == 0:
        constraint = [X <= M, -X <= M]
    else:
        constraint = [X[S0] == 0, X <= M, -X <= M]

    prob = cp.Problem(cp.Minimize(obj), constraint)
    result = prob.solve(verbose=verbose_relax,
                        solver=cp.ECOS, abstol=1e-4, reltol=1e-4)
    # eliminate 0
    x_vector = X.value
    x_vector = np.where(abs(x_vector) < 1e-7, 0, x_vector)
    return result, x_vector


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

        self.u = self.y - self.A@self.x

    def get_lambda_interval(self):
        self.omega = 0.5 * np.linalg.norm(self.y - self.A @ self.x) ** 2 - \
            0.5 * np.linalg.norm(self.y)**2 + 0.5 * \
            np.linalg.norm(self.y-self.u)**2

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

        print(test)

        j0 = -1
        for i in range(len(test)):
            if (test[i].value >= self.lambda0):
                j0 = i

        def calc_bound(j):
            constant1 = self.omega + \
                np.sum([test[i].value for i in range(j + 1)])
            constant2 = len(self.S1)-np.count_nonzero(self.x)+j+1

            if not 0 <= constant2:
                # negative
                possible_lower_bound = constant1 / constant2
                possible_upper_bound = np.inf
            else:
                # positive
                possible_upper_bound = constant1 / constant2
                possible_lower_bound = -np.inf
            return (possible_lower_bound, possible_upper_bound)

        possible_lower_bound, possible_upper_bound = calc_bound(j0)
        print("omega:", self.omega)
        # print(constant1, constant2)

        test.append(Au(-np.inf, j0+1))
        print("J0:", j0)
        print("First interval",
              "[", test[j0 + 1].value, ",", test[j0].value, "]")

        while (j0 >= 0 and (possible_upper_bound >= test[j0].value or possible_lower_bound <= test[j0+1].value)):
            j0 -= 1
            possible_lower_bound, possible_upper_bound = calc_bound(j0)

        self.upper_lambda = possible_upper_bound
        self.lower_lambda = possible_lower_bound

        print(self.lower_lambda, self.upper_lambda)

        # if not 0 <= constant2:
        #     # negative
        #     possible_lower_bound = constant1 / constant2
        #     print(possible_lower_bound)

        #     while (possible_lower_bound >= test[j0].value or possible_lower_bound <= test[j0+1].value):
        #         j0 -= 1
        #         constant1, constant2 = calc_bound(j0)

        # else:
        #     # positive
        #     possible_upper_bound = constant1 / constant2

    def __le__(self, other) -> bool:
        return self.pvl <= other.pvl

    def __str__(self) -> str:
        text = f"S0: {self.S0} | S1: {self.S1} | S: {self.S}\nPvl: {self.pvl}\n"
        return text

    def __repr__(self) -> str:
        text = f"S0: {self.S0} | S1: {self.S1} | Pvl: {self.pvl}"
        return text


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
            self.show(f"{name} pvl:", v.pvl)
            if not pv_opt1 <= v.pvl:
                q.put(v)
                self.show(f"{name} pu:", v.pu)
                pv_opt1, node_opt1 = self.check_bound(node_opt1, v)
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
            u.get_lambda_interval()

            if (u.upper_lambda < self.upper_lambda):
                self.upper_lambda = u.upper_lambda
            if (u.lower_lambda > self.lower_lambda):
                self.lower_lambda = u.lower_lambda

            self.show(f"u node: S0: {u.S0} | S1: {u.S1} | Pvl: {u.pvl}\n")

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
        return pv_opt, node_opt, self.node_counter


if __name__ == "__main__":
    np.random.seed(420)
    n = 15
    A = np.random.randint(-10, 10, (n, n))
    y = np.random.randint(-10, 10, (n, 1))

    lambda_0 = 0.5
    M0 = 100

    solver = BnBNormalAlgorythm(y, A, lambda0=lambda_0, M=M0)

    pv, node, num_nodes = solver.solve(verbose=False)

    print("Solution Node")
    # remember to count node 0
    print(f"Nodes visited {num_nodes}/{2**(n) + 1}")
    print(node)

    print("Getting lambda intervals")
    print(solver.lower_lambda, solver.upper_lambda)

    # S = list(range(n))
    # S0 = []
    # S1 = []
    # node = Node(n, S0, S1, S, A, y, lambda_0, M0)
    # node.calculate_obj()
    # node.get_lambda_interval()
