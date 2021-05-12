using QuadraticAssignmentProblem

using ProgressMeter
using DataFrames
using CSV
using Dates
using Random


iterations = 10

problem = (
	N = 10,
	objective = :max,
	matrix_type = :uniform,
	density = missing,
)

algorithms = [
	(name = "opt",    method = qap_exact),
	(name = "lin",    method = qap_linearization),
	(name = "lpr",    method = qap_lprounding),
	(name = "faq",    method = faq),
	(name = "rand",   method = qap_random),
	(name = "mms",    method = qap_mms),
	# (name = "ns",     method = qap_ns),
	# (name = "star",   method = qap_starpacking),
	# (name = "dense",  method = qap_densemapping),
]


function compare(problem, algorithms, iterations)
	warmup_problem = merge(problem, (;N=5))
	compare_step(warmup_problem, algorithms)

	results = []
	@showprogress for i in 1:iterations
		append!(results, compare_step(problem, algorithms))
	end

	save_results(results)

	return results
end

function compare_step(problem, algorithms)
	A, B, opt = generate_example(problem)
	uid = rand(UInt)

	results = []
	for a in algorithms
		alg = a.method
		time = @elapsed P, _ = alg(A, B, problem.objective)
		obj = qap_objective(A, B, P)

		if (a.name == "opt") opt = obj end

		r = merge(problem, (;uid, method=a.name, time, obj, ratio=obj/opt))
		push!(results, r)
	end

	return results
end

function generate_example(problem)
	N = problem.N
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

	results_df_wide = stack(results_df_long, metric_cols)
	transform!(results_df_wide, [:variable, :method] => ByRow((metric, method) -> metric*"_"*method) => :key)
	select!(results_df_wide, Not([:variable, :method]))
	results_df_wide = unstack(results_df_wide, :key, :value)
	results_df_wide |> CSV.write(joinpath(dir, "results_wide.csv"))

	return
end

if abspath(PROGRAM_FILE) == @__FILE__
	compare(problem, algorithms, iterations)
end
