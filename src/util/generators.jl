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
