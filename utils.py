import numpy as np
import cvxpy as cp

def evaluation_main(y, A, x, lambda0=1):
    n = y.shape[0]
    value = 0.5 * np.linalg.norm(y - A @ x) ** 2 / n + \
        lambda0 * np.count_nonzero(x)
    return value


def relax_problem(y, A, S0, S1, S, lambda0=1, M=1000000, verbose = False):

    n = y.shape[0]
    X = cp.Variable((n, 1))
    obj = 0.5 * cp.sum_squares(y - A @ X) / n
    obj += lambda0 / M * cp.norm(X[S], 1)
    obj += lambda0 * len(S1)

    if len(S0) == 0:
        constraint = [X <= M, -X <= M]
    else:
        constraint = [X[S0] == 0, X <= M, -X <= M]

    prob = cp.Problem(cp.Minimize(obj), constraint)
    result = prob.solve(verbose=verbose,
                        solver=cp.ECOS, abstol=1e-4, reltol=1e-4)
    # eliminate 0
    x_vector = X.value
    x_vector = np.where(abs(x_vector) < 1e-7, 0, x_vector)
    return result, x_vector


