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
Compute the accuracy of an estimated graph matching relative to the canonical
matching, excluding seeeds.
"""
function match_ratio(true_match::Array{Int,2}, est_match::Array{Int,2}; seeds::Int=0)

	true_match = sortslices(true_match, dims=1)
	est_match  = sortslices(est_match, dims=1)
	@assert true_match[:,1] == est_match[:,1]

	n = size(true_match, 1)
	acc = sum(true_match[seeds+1:end,2] .== est_match[seeds+1:end,2]) / (n - seeds)
	return acc

end

"""
Compute the accuracy of an estimated graph matching relative to the canonical
graph matching, excluding seeds and errors.
"""
function match_ratio(true_match::Array{Int,2}, est_match::Array{Int,2}, seeds::Int, errors::Array{Int,2})

	ignore = union(1:seeds, errors[:,1], errors[:,2])
	keepind = setdiff(1:size(true_match, 1), ignore)

	true_match = sortslices(true_match, dims=1)
	est_match  = sortslices(est_match, dims=1)
	@assert true_match[:,1] == est_match[:,1]

	true_match = true_match[keepind,2]
	est_match  = est_match[keepind,2]

	acc = sum(true_match .== est_match) / length(true_match)
	return acc

end
