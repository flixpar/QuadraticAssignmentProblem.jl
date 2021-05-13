using QuadraticAssignmentProblem

using ProgressMeter
using DataFrames
using CSV
using Dates
using Random

using Gurobi
qap_set_optimizer!(Gurobi.Optimizer)


Ns = collect(10:10:100)
iterations = 4

problem = (
	objective = :max,
	matrix_type = :known,
	density = missing,
)

algorithm = (name = "faq", method = faq)
# (name = "opt",    method = qap_exact)
# (name = "lin",    method = qap_linearization)
# (name = "lpr",    method = qap_lprounding)
# (name = "faq",    method = faq)
# (name = "rand",   method = qap_random)
# (name = "mms",    method = qap_mms)
# (name = "ns",     method = qap_ns)
# (name = "star",   method = qap_starpacking)
# (name = "dense",  method = qap_densemapping)


function evaluate(Ns, problem, algorithm, iterations)
	evaluate_step(5, problem, algorithm)

	results = []
	@showprogress for N in Ns, i in 1:iterations
		push!(results, evaluate_step(N, problem, algorithm))
	end

	save_results(results)

	return results
end

function evaluate_step(N, problem, algorithm)
	A, B, opt = generate_example(N, problem)
	uid = rand(UInt)

	alg = algorithm.method

	time = @elapsed P, _ = alg(A, B, problem.objective)
	obj = qap_objective(A, B, P)

	result = merge(problem, (;N, uid, method=algorithm.name, time, obj, ratio=obj/opt))
	return result
end

function generate_example(N, problem)
	opt = missing
	if problem.matrix_type == :uniform
		A = uniform_matrix(N)
		B = uniform_matrix(N)
	elseif problem.matrix_type == :metric
		A = metric_matrix(N)
		B = metric_matrix(N)
	elseif problem.matrix_type == :undirected_graph
		A = zeroone_matrix(N, p=problem.density, selfedges=false, symmetric=true)
		B = zeroone_matrix(N, p=problem.density, selfedges=false, symmetric=true)
	elseif problem.matrix_type == :directed_graph
		A = zeroone_matrix(N, p=problem.density, selfedges=false, symmetric=false)
		B = zeroone_matrix(N, p=problem.density, selfedges=false, symmetric=false)
	elseif problem.matrix_type == :known
		A, B, perm = generate_qap(N)
		opt = qap_objective(A, B, perm)
	else
		error("Invalid matrix type: $(problem.matrix_type)")
	end

	return A, B, opt
end

function save_results(results)

	d = Dates.format(now(), "yyyymmddHHMM")
	d = isdir("results/$d/") ? d * "_" * randstring(8) : d

	dir = joinpath("./results/", d)
	mkpath(dir)

	metric_cols = [:time, :obj, :ratio]
	problem_cols = setdiff(keys(results[1]), metric_cols, [:uid])

	results_df_long = DataFrame(results)
	results_df_long |> CSV.write(joinpath(dir, "results_long.csv"))

	mean(xs) = sum(xs) / length(xs)
	results_mean_df = combine(groupby(results_df_long, problem_cols), [m => mean => m for m in metric_cols]...)
	results_mean_df |> CSV.write(joinpath(dir, "results_mean.csv"))

	return
end

if abspath(PROGRAM_FILE) == @__FILE__
	evaluate(Ns, problem, algorithm, iterations)
end
