using Random


"""
Approximately solve the QAP through random sampling.
"""
function qap_random(A, B, obj; iter=100)
	@assert obj in [:min, :max]
	N = size(A, 1)

	best_sol = ((obj == :min) ? Inf : -Inf, nothing)

	for i in 1:iter
		perm = randperm(N)
		s = sum(A .* B[perm, perm])

		if obj == :min && s < best_sol[1]
			best_sol = (s, perm)
		elseif obj == :max && s > best_sol[1]
			best_sol = (s, perm)
		end
	end

	perm = best_sol[2]
	P = Matrix(I, N, N)[perm,:]

	return P, perm
end
