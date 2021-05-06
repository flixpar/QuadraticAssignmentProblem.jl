using Random
using LinearAlgebra


function uniform_matrix(N)
	return rand(N, N)
end

function metric_matrix(N; dim=5, p=2)
	points = rand(N, dim)
	M = [norm(points[i,:] - points[j,:], p) for i in 1:N, j in 1:N]
	return M
end
