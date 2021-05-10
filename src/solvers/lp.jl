using JuMP
using Gurobi
using Hungarian
using LinearAlgebra
import MathOptInterface as MOI


"""
Approximately solve the QAP by linearizing the problem and solving
the resulting linear integer program.
"""
function qap_linearization(A, B, obj)
	P, _ = adams_johnson_linearization(A, B, obj, integer=true)
	matching = [findfirst(==(1), P[i,:]) for i in 1:size(A,1)]
	return P, matching
end

"""
Approximately solve the QAP by linearizing and relaxing the problem
and solving the resulting LIP with linear programming and rounding
via the Hungarian algorithm.
"""
function qap_lprounding(A, B, obj)
	x, _ = adams_johnson_linearization(A, B, obj, integer=false)

	W = maximum(x) .- x
	P = Hungarian.munkres(W)

	P = Matrix(P .== Hungarian.STAR)
	matching = [findfirst(P[i,:]) for i in 1:size(A,1)]

	return P, matching
end

"""
QAP linearization from "Improved linear programming-based
lower bounds for the quadratic assignment problem" by Adams and Johnson.
"""
function adams_johnson_linearization(A, B, obj; integer::Bool=true)
	N = size(A, 1)

	model = Model(Gurobi.Optimizer)
	set_silent(model)

	@variable(model, x[1:N,1:N], binary=integer)
	@variable(model, y[1:N,1:N,1:N,1:N], binary=integer)

	@constraint(model, [u=1:N], sum(x[u,:]) == 1)
	@constraint(model, [p=1:N], sum(x[:,p]) == 1)
	@constraint(model, [v=1:N,p=1:N,q=1:N], sum(y[:,p,v,q]) == x[v,q])
	@constraint(model, [u=1:N,v=1:N,q=1:N], sum(y[u,:,v,q]) == x[v,q])
	@constraint(model, [u=1:N,p=1:N,v=1:N,q=1:N], y[u,p,v,q] == y[v,q,u,p])

	if !integer
		@constraint(model, [u=1:N,p=1:N], 0 ≤ x[u,p] ≤ 1)
		@constraint(model, [u=1:N,p=1:N,v=1:N,q=1:N], 0 ≤ y[u,p,v,q] ≤ 1)
	end

	model_obj = (obj == :min) ? MOI.MIN_SENSE : MOI.MAX_SENSE
	@objective(model, model_obj, sum(A[u,v] * B[p,q] * y[u,p,v,q] for u=1:N, p=1:N, v=1:N, q=1:N))

	optimize!(model)

	status = Int(termination_status(model))
	@assert status == 1

	x_sol = value.(x)
	y_sol = value.(y)

	clamp!(x_sol, 0, 1)
	clamp!(y_sol, 0, 1)

	if integer
		x_sol = x_sol .> 0.5
		y_sol = y_sol .> 0.5
	end

	return x_sol, y_sol
end
