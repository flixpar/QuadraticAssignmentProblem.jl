using JuMP
using Gurobi

using Hungarian

using LinearAlgebra

using Random
using Distributions


"""
Run the algorithm from "Maximum Quadratic Assignment Problem: Reduction
from Maximum Label Cover and LP-based Approximation Algorithm" by
Makarychev, Manokaran, and Sviridenko (MMS) to approximately solve the QAP.
"""
function mms(A, B, obj)
	@assert obj == :max
	x, y = lp_mms(A, B)
	perm = rounding_mms(A, B, x, y)
	P = Int.(Matrix(I, size(A,1), size(A,1)))[perm,:]
	return P, perm
end

"""
LP relaxation used by the MMS algorithm.
"""
function lp_mms(A, B)
	N = size(A, 1)

	model = Model(Gurobi.Optimizer)
	set_silent(model)

	@variable(model, x[1:N,1:N] ≥ 0)
	@variable(model, y[1:N,1:N,1:N,1:N] ≥ 0)

	@constraint(model, [u=1:N], sum(x[u,:]) == 1)
	@constraint(model, [p=1:N], sum(x[:,p]) == 1)
	@constraint(model, [v=1:N,p=1:N,q=1:N], sum(y[:,p,v,q]) == x[v,q])
	@constraint(model, [u=1:N,v=1:N,q=1:N], sum(y[u,:,v,q]) == x[v,q])
	@constraint(model, [u=1:N,p=1:N,v=1:N,q=1:N], y[u,p,v,q] == y[v,q,u,p])
	@constraint(model, [u=1:N,p=1:N], 0 ≤ x[u,p] ≤ 1)
	@constraint(model, [u=1:N,p=1:N,v=1:N,q=1:N], 0 ≤ y[u,p,v,q] ≤ 1)

	@objective(model, Max, sum(A[u,v] * B[p,q] * y[u,p,v,q] for u=1:N, p=1:N, v=1:N, q=1:N))

	optimize!(model)

	status = Int(termination_status(model))
	@assert status == 1

	x_sol = max.(0.0, value.(x))
	y_sol = max.(0.0, value.(y))

	return x_sol, y_sol
end


"""
LP rounding step of the MMS algorithm.
"""
function rounding_mms(A, B, x, y)
	N = size(A, 1)
	n2 = Int(floor(N/2))

	La = randperm(N)[1:n2]
	Lb = randperm(N)[1:n2]

	Ra = setdiff(1:N, La)
	Rb = setdiff(1:N, Lb)

	ϕ̃ = fill(-1, N)
	for u in La
		probs = x[u,Lb]
		push!(probs, 1 - sum(probs))
		p′ = rand(Categorical(probs))
		p = (p′ == length(probs)) ? -1 : p′
		ϕ̃[u] = p
	end

	ϕ = fill(-1, N)
	for p in Lb
		us = findall(==(p), ϕ̃)
		if isempty(us) continue end
		u = rand(us)
		ϕ[u] = p
	end

	La = findall(!=(-1), ϕ)
	if isempty(La)
		return rounding_mms(A, B, x, y)
	end

	W = [sum(A[u,v] * B[ϕ[u],q] for u in La) for v in Ra, q in Rb]
	matching = Hungarian.munkres(W)
	ψ = fill(-1, N)
	for i in 1:Int(ceil(N/2))
		j = findfirst(matching[i,:] .== Hungarian.STAR)
		ψ[Ra[i]] = Rb[j]
	end

	perm = [(ϕ[i] > 0) ? ϕ[i] : (ψ[i] > 0) ? ψ[i] : -1 for i in 1:N]

	missing_ind = findall(==(-1), perm)
	missing_vals = shuffle(setdiff(1:N, perm))
	perm[missing_ind] = missing_vals

	return perm
end
