using Random
using LinearAlgebra


"""
Generate an N×N matrix with uniformly random entries in [0,1].
"""
function uniform_matrix(N)
	return rand(N, N)
end

"""
Generate a random N×N matrix with entries satisfying the triangle inequality.
"""
function metric_matrix(N; dim=5, p=2)
	points = rand(N, dim)
	M = [norm(points[i,:] - points[j,:], p) for i in 1:N, j in 1:N]
	return M
end

"""
Generate a random 0-1 N×N matrix with edge probability p.
"""
function zeroone_matrix(N; p=0.25, selfedges=true, symmetric=false)
	M = rand(N, N) .≤ p
	if symmetric
		M = triu(M) .| triu(M)'
	end
	if !selfedges
		for i in 1:N
			M[i,i] = 0
		end
	end
	return M
end

"""
Generate a pair of N×N matricies with known optimal permutation using
Algorithm 4 from "Generating Quadratic Assignment Test Problems with
Known Optimal Permutations" by Li and Pardalos.
"""
function generate_qap(N)
	D = Matrix(I, N, N)
	O = .~D

	ΔA = rand(1:100)
	ΔB = rand(1:100)

	A = fill(ΔA, N, N)
	A[D] .= 0

	B = Array{Int,2}(undef, N, N)
	for i in 1:N, j in 1:N
		if i == j
			B[i,j] = 0
		else
			B[i,j] = rand(0:ΔB)
		end
	end

	R = findall(O)
	sort!(R, by=(ij -> B[ij]))

	X = rand(1:ΔA, N*(N-1))
	sort!(X, rev=true)

	for (k, ij) in enumerate(R)
		A[ij] = A[ij] - X[k]
	end

	σ = randperm(N)
	σinv = invperm(σ)
	B = B[σinv,σinv]

	return A, B, σ
end
