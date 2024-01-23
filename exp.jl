using El0ps
using Random
using CSV
using DataFrames
Random.seed!(42)

#Lambda must be a float

# println(pwd())
matrix = CSV.File("parsimonious/files/matrix.csv", header=false) |> DataFrame
A = Matrix(matrix)
df_y = CSV.File("parsimonious/files/y.csv", header=false) |> DataFrame
y = Vector(df_y[:, 1])

M = 100.
f = LeastSquares(y)
h = Bigm(M)
λ = 1.0;
problem = Problem(f, h, A, λ);
problem = Problem(f, h, A, λ);

solver = BnbSolver(maxtime= 60.)
result = optimize(solver, problem)
result.x