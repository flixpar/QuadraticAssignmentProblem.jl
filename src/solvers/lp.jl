using JuMP
using Gurobi
using Hungarian
using LinearAlgebra


"""
Approximate the linear assignment problem using linear programming
with rounding via the Hungarian Algorithm.
"""
function lap_relaxation(A, B)
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

	@objective(model, Min, sum(E₁ + E₂))

	optimize!(model)

	status = Int(termination_status(model))
	if status != 1
		error("No optimal solution found")
	end

	P = value.(P)
	P = Hungarian.munkres(P)
	P = Int.(Matrix(P .== Hungarian.STAR))

	matching = [findfirst(==(1), P[i,:]) for i in 1:n]

	return P, matching
end
