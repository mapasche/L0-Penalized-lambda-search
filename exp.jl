using El0ps
using Random
using CSV
using DataFrames
Random.seed!(42)


println(pwd())
#Data generation
matrix = CSV.File("parsimonious/matrix.csv", header=false) |> DataFrame
A = Matrix(matrix)
df_y = CSV.File("parsimonious/y.csv", header=false) |> DataFrame
y = Vector(df_y[:, 1])
#y = randn(10)




M = 100.
f = LeastSquares(y)
α = 1.0
h = Bigm(M)
#A = rand(m, n)
λ = 0.5

#Problem instantiation
problem = Problem(f, h ,A, λ)

#problem resolution
solver = BnbSolver(maxtime= 60.)
result = optimize(solver, problem)

result.x


#Path fitting
#path = fit_path(solver, f, h, A)