using LinearAlgebra
using LinearAlgebra.BLAS: gemm
using Hungarian


"""
Perform the FAQ algorithm from "Fast Approximate Quadratic Programming for Graph Matching"
to approximately solve the QAP.
"""
function faq(A::AbstractArray{<:Real,2}, B::AbstractArray{<:Real,2}; iter::Int=30, initmethod::Symbol=:barycenter)
	@assert initmethod in [:barycenter, :random]

	# 0. Get size
	N = size(A,1)

	# 0. Convert adjacency matricies
	A = Float32.(A)
	B = Float32.(B)

	# 0. Set constant
	ϵ = 1e-3

	# 1. Initialize P
	P = ones(Float32, N, N) / N
	if initmethod == :random
		P = sample_doublystochastic(N)
	end

	# 2. While stopping criteria not met:
	for i = 1:iter

		# 3. Compute the gradient
		s1 = gemm('T', 'N', A, gemm('N', 'N', P, B))
		s2 = gemm('N', 'N', A, gemm('N', 'T', P, B))
		∇f = s1 + s2

		# 4. Find best permutation via Hungarian algorithm
		mm = maximum(∇f)
		matching = Hungarian.munkres(-∇f .+ (mm+0.01))
		Q = Float32.(Matrix(matching .== Hungarian.STAR))

		# 5. Compute the (maybe) optimal step size
		s3 = gemm('T', 'N', A, gemm('N', 'N', Q, B))
		c = tr(gemm('N', 'T', s1, P))
		d = tr(gemm('N', 'T', s1, Q)) + tr(gemm('N', 'T', s3, P))
		e = tr(gemm('N', 'T', s3, Q))

		α̃::Float32 = 0
		if (c - d + e != 0)
			α̃ = - (d - 2e) / (2 * (c - d + e))
		end

		# 6. Update P

		f0 = 0
		f1 = c - e
		falpha = ((c - d + e) * (α̃^2)) + ((d - 2e) * α̃)

		if α̃ < 1-ϵ && α̃ > ϵ && falpha > f0 && falpha > f1
			P = (α̃ * P) + ((1-α̃) * Q)
		elseif f0 > f1
			P = Q
		else
			break
		end

	end

	# 7. Find the nearest permutation matrix with Hungarian algorithm
	mm = maximum(P)
	matching = Hungarian.munkres(-P .+ (mm+0.001))
	P̂ = Matrix(matching .== Hungarian.STAR)

	# 8. Compute the matching associated to P̂
	matching = argmax(P̂', dims=2)
	matching = hcat(getindex.(matching,1), getindex.(matching,2))
	matching = sortslices(matching, dims=1)

	# 9. Return P̂ and the associated matching
	return P̂, matching

end

"""
Perform the SGM algorithm from "Seeded Graph Matching" to approximately
solve the QAP with m seeds.
"""
function sgm(A::AbstractArray{<:Real,2}, B::AbstractArray{<:Real,2}, m::Int; initmethod::Symbol=:barycenter, maxiter::Int=20, returniter::Bool=false)

	# 0. Checks
	@assert size(A,1) == size(A,2)
	@assert size(B,1) == size(B,2)
	@assert size(A,1) == size(B,1)
	@assert initmethod in [:barycenter, :random]

	# 0. Get size
	N = size(A,1)
	n = N - m

	# 0. Convert adjacency matricies
	A = Float32.(A)
	B = Float32.(B)

	# 0. Split A and B into seeded and non-seeded parts

	A11 = A[1:m, 1:m]
	A12 = A[1:m, m+1:end]
	A21 = A[m+1:end, 1:m]
	A22 = A[m+1:end, m+1:end]

	B11 = B[1:m, 1:m]
	B12 = B[1:m, m+1:end]
	B21 = B[m+1:end, 1:m]
	B22 = B[m+1:end, m+1:end]

	# 0. Precompute constant values
	s1 = tr(gemm('T', 'N', A11, B11))
	s2 = gemm('N', 'T', A21, B21)
	s3 = gemm('T', 'N', A12, B12)
	ϵ = 1e-3

	# 1. Initialize P
	P = ones(Float32, n, n) / n
	if initmethod == :random
		P = sample_doublystochastic(n)
	end

	# 2. While stopping criteria not met:
	it = 0
	while it < maxiter
		it += 1

		# 3. Compute the gradient
		s4 = gemm('T', 'N', A22, gemm('N', 'N', P, B22))
		∇f = s2 + s3 + s4 + gemm('N', 'N', A22, gemm('N', 'T', P, B22))

		# 4. Find best permutation via Hungarian algorithm
		mm = maximum(∇f) + ϵ
		matching = Hungarian.munkres(-∇f .+ mm)
		Q = Float32.(Matrix(matching .== Hungarian.STAR))

		# 5. Compute the (maybe) optimal step size
		s5 = gemm('T', 'N', A22, gemm('N', 'N', Q, B22))
		c = tr(gemm('N', 'T', s4, P))
		d = tr(gemm('N', 'T', s4, Q)) + tr(gemm('N', 'T', s5, P))
		e = tr(gemm('N', 'T', s5, Q))
		u = tr(gemm('T', 'N', P, s2)) + tr(gemm('T', 'N', P, s3))
		v = tr(gemm('T', 'N', Q, s2)) + tr(gemm('T', 'N', Q, s3))

		α̃::Float32 = 0
		if (c - d + e != 0)
			α̃ = - (d - 2e + u - v) / (2 * (c - d + e))
		end

		# 6. Update P

		f0 = 0
		f1 = c - e + u - v
		falpha = (c - d + e) * (α̃^2) + (d - 2e + u - v) * α̃

		if ϵ < α̃ < 1-ϵ && falpha > f0 && falpha > f1
			P = (α̃ * P) + ((1-α̃) * Q)
		elseif f0 > f1
			P = Q
		else
			break
		end

	end

	# 7. Find the nearest permutation matrix with Hungarian algorithm
	mm = maximum(P) + ϵ
	matching = Hungarian.munkres(-P .+ mm)
	P̂ = Matrix(matching .== Hungarian.STAR)
	P̂ = [Matrix(I, m, m) zeros(Bool, m, n); zeros(Bool, n, m) P̂]
	P̂ = Int.(P̂)

	# 8. Compute the matching associated to P̂
	matching = argmax(P̂', dims=2)
	matching = hcat(getindex.(matching,1), getindex.(matching,2))
	matching = sortslices(matching, dims=1)

	if returniter
		return P̂, matching, it
	end

	# 9. Return P̂ and the associated matching
	return P̂, matching

end

"""
Randomly sample a doubly stochastic matrix by iteratively normalizing the
rows and columns of a uniform random matrix.
"""
function sample_doublystochastic(N::Int; maxiter=50, ε::Float64=0.001)
	err = x -> sum(abs.(sum(x, dims=1) .- 1)) + sum(abs.(sum(x, dims=2) .- 1))

	M = rand(Float32, N, N)

	it = 0
	while err(M) > ε && it < maxiter
		M = M ./ sum(M, dims=1)
		M = M ./ sum(M, dims=2)
		it += 1
	end

	return M
end
