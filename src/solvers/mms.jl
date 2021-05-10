using Hungarian
using LinearAlgebra
using Random
using Distributions

include("lp.jl")


"""
Run the algorithm from "Maximum Quadratic Assignment Problem: Reduction
from Maximum Label Cover and LP-based Approximation Algorithm" by
Makarychev, Manokaran, and Sviridenko (MMS) to approximately solve the QAP.
"""
function qap_mms(A, B, obj)
	@assert obj == :max
	x, y = adams_johnson_linearization(A, B, integer=false)
	perm = rounding_mms(A, B, x, y)
	P = Matrix(I, size(A)...)[perm,:]
	return P, perm
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
		probvec!(probs)
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
	W = maximum(W) .- W
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

function probvec!(xs::Array{<:Real,1})
	z = zero(xs[1])
	s = z
	for i in 1:length(xs)
		if xs[i] < z
			xs[i] = z
		end
		s += xs[i]
	end
	if s != 1
		for i in 1:length(xs)
			xs[i] = xs[i] / s
		end
	end
	return xs
end
