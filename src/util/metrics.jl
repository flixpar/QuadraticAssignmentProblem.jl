using LinearAlgebra


"""
Compute the QAP objective function value.
"""
function qap_objective(A::AbstractArray{<:Real,2}, B::AbstractArray{<:Real,2}, P::Union{BitMatrix, Matrix{Bool}, Matrix{Int}})
	return tr(A * P * B' * P')
end

function qap_objective(A::AbstractArray{<:Real,2}, B::AbstractArray{<:Real,2}, perm::Array{Int,1})
	return sum(A .* B[perm,perm])
end

"""
Compute the match ratio of two permutations.
"""
function match_ratio(perm1::Array{Int,1}, perm2::Array{Int,1})
	@assert length(perm1) == length(perm2)
	@assert isperm(perm1) && isperm(perm2)
	N = length(perm1)
	mr = sum(perm1 .== perm2) / N
	return mr
end
