using Random
using Distributions
using LinearAlgebra


"""
Generate a pair of graphs from a ρ-correlated stochastic block model with all
of the parameters of the model specified.
"""
function generate_ρsbm(N::Int, Λ::Array{Float64,2}, b::Array{Int,1}, ρ::Float64)

	A = zeros(Int, N, N)
	B = zeros(Int, N, N)

	# sample graph A from a SBM
	for i in 1:N
		for j in i+1:N
			λ = Λ[b[i], b[j]]
			A[i,j] = rand(Bernoulli(λ))

			θ = ((1 - ρ) * λ) + (ρ * A[i,j])
			B[i,j] = rand(Bernoulli(θ))
		end
	end

	# make adjacency matricies symmetric
	A = A + A'
	B = B + B'

	# construct matching between graphs
	matching = hcat(1:N, 1:N)

	return A, B, matching
end

"""
Generate a pair of graphs from a ρ-correlated Erdős-Rényi model with
parameter p.
"""
function generate_erdosrenyi(N::Int, p::Float64, ρ::Float64)

	A = zeros(Int, N, N)
	B = zeros(Int, N, N)

	# sample graph A and B from a ρ-correlated Erdős-Rényi model
	for i in 1:N
		for j in i+1:N
			A[i,j] = rand(Bernoulli(p))

			λ = ((1 - ρ) * p) + (ρ * A[i,j])
			B[i,j] = rand(Bernoulli(λ))
		end
	end

	# make adjacency matricies symmetric
	A = A + A'
	B = B + B'

	# construct matching between graphs
	matching = hcat(1:N, 1:N)

	return A, B, matching
end

"""
Generate a pair of graphs from a ρ-correlated Bernoulli model with
parameters generated from uniform(0,1) on N vertices.
"""
function generate_bernoulli(N::Int, ρ::Float64)
	@assert 0 <= ρ <= 1

	P = zeros(Float64, N, N)
	dist = Uniform(0,1)
	for i in 1:N
		for j in i+1:N
			P[i,j] = rand(dist)
		end
	end
	P = P + P'

	A, B, matching = generate_bernoulli(P, ρ)
	return A, B, matching, P
end

"""
Generate a pair of graphs from a ρ-correlated Bernoulli model with
parameters generated from uniform(p-δ,p+δ) on N vertices.
"""
function generate_bernoulli(N::Int, p::Float64, δ::Float64, ρ::Float64)
	@assert 0 <= p <= 1
	@assert 0 <= δ <= min(p, 1-p)

	if δ == 0
		A, B, matching = generate_erdosrenyi(N, p, ρ)
		P = fill(p, (N, N)) - (Matrix(I, N, N) * p)
		return A, B, matching, P
	end

	P = zeros(Float64, N, N)
	for i in 1:N
		for j in i+1:N
			P[i,j] = rand(Uniform(p-δ,p+δ))
		end
	end
	P = P + P'

	A, B, matching = generate_bernoulli(P, ρ)
	return A, B, matching, P
end

"""
Generate a pair of graphs from a ρ-correlated Bernoulli model with
parameters p.
"""
function generate_bernoulli(p::Array{Float64,2}, ρ::Float64)

	@assert size(p, 1) == size(p, 2)
	N = size(p, 1)

	A = zeros(Int, N, N)
	B = zeros(Int, N, N)

	# sample graph A and B from a ρ-correlated Bernoulli model
	for i in 1:N
		for j in i+1:N
			A[i,j] = rand(Bernoulli(p[i,j]))

			λ = ((1 - ρ) * p[i,j]) + (ρ * A[i,j])
			B[i,j] = rand(Bernoulli(λ))
		end
	end

	# make adjacency matricies symmetric
	A = A + A'
	B = B + B'

	# construct matching between graphs
	matching = hcat(1:N, 1:N)

	return A, B, matching
end

"""
Generate a pair of graphs from a Bernoulli random graph model with
edge likelihood values Λ and edge correlation values R.
"""
function generate_hetero_bernoulli(Λ::Array{Float64,2}, R::Array{Float64,2})

	@assert size(Λ, 1) == size(Λ, 2)
	@assert size(R, 1) == size(R, 2)
	@assert size(Λ, 1) == size(R, 1)
	N = size(Λ, 1)

	A = zeros(Int, N, N)
	B = zeros(Int, N, N)

	# sample graph A and B from a ρ-correlated Bernoulli model
	for i in 1:N
		for j in i+1:N
			A[i,j] = rand(Bernoulli(Λ[i,j]))

			λ = ((1 - R[i,j]) * Λ[i,j]) + (R[i,j] * A[i,j])
			B[i,j] = rand(Bernoulli(λ))
		end
	end

	# make adjacency matricies symmetric
	A = A + A'
	B = B + B'

	# construct matching between graphs
	matching = hcat(1:N, 1:N)

	return A, B, matching
end

"""
Compute the total correlation between two graphs, given the matrix of edge
likelihoods used to generate them and the correlation between them.
"""
function total_correlation(p::Array{Float64, 2}, ρe::Float64)
	N = size(p, 1)

	μ = sum(triu(p,1)) / (N * (N-1) / 2)
	σ² = sum(triu(p .- μ,1).^2) / (N * (N-1) / 2)
	ρh = σ² / (μ * (1-μ))

	ρtotal = 1 - ((1-ρh) * (1-ρe))
	return ρtotal
end

"""
Compute the total correlation between two graphs generated from a ρe-correlated
Bernoulli random graph model with parameters generated from Uniform[p-δ, p+δ].
"""
function total_correlation(p::Float64, δ::Float64, ρe::Float64)
	ρh = δ^2 / (3 * p * (1-p))
	ρtotal = 1 - ((1-ρh) * (1-ρe))
	return ρtotal
end
