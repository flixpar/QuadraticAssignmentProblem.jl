using QuadraticAssignmentProblem

using ProgressMeter
using DataFrames
using CSV
using Dates

default_algorithms = [
	(name = "opt",  method = qap_ip),
	(name = "lip",  method = lap_ip),
	(name = "lpr",  method = lap_relaxation),
	(name = "faq",  method = faq),
	(name = "rand", method = qap_random),
	(name = "mms",  method = mms),
]


function compare(N, iter, algorithms)
	compare_step(6, algorithms)

	results = []
	@showprogress for i in 1:iter
		push!(results, compare_step(N, algorithms))
	end

	save_results(results, algorithms)

	return results
end

function compare_step(N, algorithms)
	A = uniform_matrix(N)
	B = uniform_matrix(N)

	r = Dict{String,Any}("N" => N, "uid" => rand(UInt))
	for a in algorithms
		a_name, alg = a.name, a.method
		t = @elapsed P, _ = alg(A,B)
		o = qap_objective(A, B, P)
		r["t_"*a_name] = t
		r["o_"*a_name] = o
		r["r_"*a_name] = o / r["o_opt"]
	end

	return r
end

function save_results(results, algorithms)

	rv = Dates.format(now(), "yyyymmddHHMM")

	if !isdir("results/")
		mkdir("results/")
	end

	col_order = vcat(["N", "uid"], [m*"_"*a.name for m in ["r", "o", "t"], a in algorithms][:])
	results_df = DataFrame([k => [r[k] for r in results] for k in col_order])
	results_df |> CSV.write("results/results_$(rv).csv")

	mean(xs) = sum(xs) / length(xs)

	results_df_long = stack(results_df, Not([:N, :uid]), variable_name=:metric)
	insertcols!(results_df_long, 3, :algorithm => [m[3:end] for m in results_df_long.metric])
	results_df_long.metric = [(m[1] == 't') ? "time" : (m[1] == 'o') ? "obj" : "ratio" for m in results_df_long.metric]
	results_df_long = unstack(results_df_long, :metric, :value)
	mean_results_df = combine(groupby(results_df_long, [:N, :algorithm]), :ratio => mean, :obj => mean, :time => mean)
	mean_results_df |> CSV.write("results/results_mean_$(rv).csv")

	return
end

if abspath(PROGRAM_FILE) == @__FILE__
	compare(10, 10, default_algorithms)
end
