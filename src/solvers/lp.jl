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
