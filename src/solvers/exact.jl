using JuMP
using Convex
using Gurobi


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

	model = Model(Gurobi.Optimizer)
	set_silent(model)

	@variable(model, P[1:n,1:n], Bin)

	@constraint(model, sum(P, dims=1) .== 1)
	@constraint(model, sum(P, dims=2) .== 1)

	model_obj = (obj == :min) ? Min : Max
	@objective(model, model_obj, tr(A * P * B' * P'))

	optimize!(model)
	status = Int(termination_status(model))

	if status != 1
		error("No optimal solution found")
	end

	P = Int.(value.(P))
	matching = argmax(P', dims=2)
	matching = hcat(getindex.(matching,1), getindex.(matching,2))
	matching = vcat(hcat(1:seeds, 1:seeds), matching .+ seeds)
	matching = sortslices(matching, dims=1)

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

	solve!(problem, Gurobi.Optimizer(OutputFlag=0))

	P̂ = Int.(P.value)
	matching = argmax(P̂', dims=2)
	matching = hcat(getindex.(matching,1), getindex.(matching,2))
	matching = vcat(hcat(1:seeds, 1:seeds), matching .+ seeds)
	matching = sortslices(matching, dims=1)

	return P̂, matching
end
