using Random


"""
Approximately solve the QAP through random sampling.
"""
function qap_random(A, B; iter=100)
	N = size(A, 1)
	Im = Matrix(I, N, N)

	score = Inf
	P = nothing
	matching = nothing

	for i in 1:iter
		perm = randperm(N)
		Q = Im[perm,:]
		s = norm(A*Q - Q*B, 2)^2
		if s < score
			score = s
			P = Q
			matching = perm
		end
	end

	return Int.(P), matching
end
