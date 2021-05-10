using JuMP
using Gurobi
using Hungarian
using LinearAlgebra


"""
Solve the linear assignment problem using linear integer programming.
"""
function lap_ip(A, B, obj; seeds=0)
	@assert obj in [:min, :max]
	n = size(A, 1) - seeds

	if seeds != 0
		A = A[seeds+1:end, seeds+1:end]
		B = B[seeds+1:end, seeds+1:end]
	end

	model = Model(Gurobi.Optimizer)
	set_silent(model)

	@variable(model, P[1:n,1:n], Bin)
	@variable(model, E₁[1:n,1:n] ≥ 0)
	@variable(model, E₂[1:n,1:n] ≥ 0)

	@constraint(model, sum(P, dims=1) .== 1)
	@constraint(model, sum(P, dims=2) .== 1)

	@constraint(model, (A*P) - (P*B) .== E₁ - E₂)

	model_obj = (obj == :min) ? Min : Max
	@objective(model, model_obj, sum(E₁ + E₂))

	optimize!(model)

	status = Int(termination_status(model))
	if status != 1
		error("No optimal solution found")
	end

	P = Int.(value.(P))
	matching = argmax(P', dims=2)
	matching = hcat(getindex.(matching,1), getindex.(matching,2))
	matching = vcat(hcat(1:seeds, 1:seeds), matching .+ seeds)
	matching = sortslices(matching, dims=1)

	return P, matching
end

"""
Approximate the linear assignment problem using linear programming
with rounding via the Hungarian Algorithm.
"""
function lap_relaxation(A, B, obj)
	@assert obj in [:min, :max]
	n = size(A, 1)

	model = Model(Gurobi.Optimizer)
	set_silent(model)

	@variable(model, P[1:n,1:n] ≥ 0)
	@variable(model, E₁[1:n,1:n] ≥ 0)
	@variable(model, E₂[1:n,1:n] ≥ 0)

	@constraint(model, 0 .≤ P .≤ 1)

	@constraint(model, sum(P, dims=1) .== 1)
	@constraint(model, sum(P, dims=2) .== 1)

	@constraint(model, (A*P) - (P*B) .== E₁ - E₂)

	model_obj = (obj == :min) ? Min : Max
	@objective(model, model_obj, sum(E₁ + E₂))

	optimize!(model)

	status = Int(termination_status(model))
	if status != 1
		error("No optimal solution found")
	end

	W = value.(P)
	if (obj == :max) W = maximum(W) - W end
	P = Hungarian.munkres(W)
	P = Matrix(P .== Hungarian.STAR)

	matching = [findfirst(P[i,:]) for i in 1:n]

	return P, matching
end

"""
QAP linearization from "Improved linear programming-based
lower bounds for the quadratic assignment problem" by Adams and Johnson.
"""
function adams_johnson_linearization(A, B; integer::Bool=true)
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

	@objective(model, Max, sum(A[u,v] * B[p,q] * y[u,p,v,q] for u=1:N, p=1:N, v=1:N, q=1:N))

	optimize!(model)

	status = Int(termination_status(model))
	@assert status == 1

	x_sol = value.(x)
	y_sol = value.(y)
	clamp!(x_sol, 0, 1)
	clamp!(y_sol, 0, 1)

	return x_sol, y_sol
end
