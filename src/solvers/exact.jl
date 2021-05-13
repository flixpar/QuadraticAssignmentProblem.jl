using JuMP
using Convex
import MathOptInterface as MOI


"""
Solve the QAP using quadratic integer progamming.
"""
function qap_exact(A::AbstractArray{<:Real,2}, B::AbstractArray{<:Real,2}, obj::Symbol; seeds::Int=0)
	@assert obj in [:min, :max]
	n = size(A,1) - seeds

	if seeds != 0
		A = A[seeds+1:end, seeds+1:end]
		B = B[seeds+1:end, seeds+1:end]
	end

	optimizer = get_optimizer()
	model = Model(optimizer)
	set_silent(model)

	@variable(model, P[1:n,1:n], Bin)

	@constraint(model, sum(P, dims=1) .== 1)
	@constraint(model, sum(P, dims=2) .== 1)

	model_obj = (obj == :min) ? MOI.MIN_SENSE : MOI.MAX_SENSE
	@objective(model, model_obj, tr(A * P * B' * P'))

	optimize!(model)
	status = Int(termination_status(model))

	if status != 1
		error("No optimal solution found")
	end

	P = Int.(value.(P))

	matching = [findfirst(==(1), P[i,:]) for i in 1:n]
	matching = matching .+ seeds
	matching = vcat(1:seeds, matching)

	return P, matching
end

"""
Solve the QAP using quadratic integer progamming. Uses Convex.jl for modeling.
"""
function qap_exact_alt(A::AbstractArray{<:Real,2}, B::AbstractArray{<:Real,2}, obj::Symbol; seeds::Int=0)
	@assert obj in [:min, :max]
	n = size(A,1) - seeds

	if seeds != 0
		A = A[seeds+1:end, seeds+1:end]
		B = B[seeds+1:end, seeds+1:end]
	end

	P = Variable((n, n), :Bin)

	optim = (obj == :min) ? minimize : maximize
	problem = optim(tr(A * P * B' * P'))

	o = ones(Int, n)
	problem.constraints += P * o == o
	problem.constraints += P' * o == o

	optimizer = get_optimizer()
	solve!(problem, optimizer(OutputFlag=0))

	P̂ = Int.(P.value)

	matching = [findfirst(==(1), P̂[i,:]) for i in 1:n]
	matching = matching .+ seeds
	matching = vcat(1:seeds, matching)

	return P̂, matching
end
