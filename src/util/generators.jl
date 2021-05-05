"""
Randomly permute B.
"""
function permute(A::AbstractArray{<:Real,2}, B::AbstractArray{<:Real,2})
	N = size(A,1)

	perm = randperm(N)
	P = Matrix(I, N, N)[perm, :]
	matching = hcat(1:N, perm)

	B = P * B * P'

	return A, B, matching
end

"""
Randomly permute B while maintaining m seeds.
"""
function permute_seeded(A::AbstractArray{<:Real,2}, B::AbstractArray{<:Real,2}, m::Int)
	@assert m>=0 "Cannot have negative seeds."

	N = size(A,1)

	# seed
	seeds = sort(randperm(N)[1:m])
	nonseeds = sort(setdiff(1:N, seeds))
	perm = vcat(seeds, nonseeds)
	P = Matrix(I, N, N)[perm, :]
	A = P * A * P'
	B = P * B * P'

	# permute
	perm = vcat(1:m, shuffle(m+1:N))
	P = Matrix(I, N, N)[perm, :]
	B = P * B * P'
	matching = hcat(1:N, perm)

	return A, B, matching
end

"""
Randomly permute B while maintaining m seeds, with seed error rate r.
"""
function permute_seeded_errors(A::AbstractArray{<:Real,2}, B::AbstractArray{<:Real,2}, m::Int, r::Float64)
	@assert 0 ≦ r ≦ 1

	A, B, matching = permute_seeded(A, B, m)

	# generate errors
	e = round(Int, r * m)
	errs = shuffle(m-r+1:m)
	while any(errs .== collect(m-r+1:m))
		errs = shuffle(m-r+1:m)
	end

	# create errors in graph
	perm_err = vcat(1:m-r, errs, m+1:N)
	P_err = Matrix(I, N, N)[perm_err, :]
	B = P_err * B * P_err'

	errors = hcat(m-r+1:m, errs)
	return A, B, matching, errors
end
