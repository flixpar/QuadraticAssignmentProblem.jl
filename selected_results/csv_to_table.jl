using CSV
using DataFrames

fn = ARGS[1]

data = DataFrame(CSV.File(fn))

constcols = [c for c in names(data) if all(coalesce.(data[!,c],"") .== coalesce(data[1,c],""))]
constvals = [c => data[1,c] for c in constcols]
select!(data, Not(constcols))

table = String[]

colname_cvt = Dict("N" => "N", "method" => "Method", "time" => "Time", "obj" => "Objective", "ratio" => "\$\\alpha\$")
header = [haskey(colname_cvt, c) ? colname_cvt[c] : c for c in names(data)]
header_str = foldl((a,b) -> a * " & " * b, header) * " \\\\\n"
push!(table, header_str)
push!(table, "\\hline\n")

method_lookup = Dict(
	"opt" => "Quadratic Programming",
	"lin" => "Linearization",
	"lpr" => "LP + Rounding",
	"faq" => "FAQ",
	"rand" => "Random Search",
	"mms" => "MMS",
	"ns" => "NS",
	"star" => "Common Star Packing",
	"dense" => "Dense Subgraph Mapping",
)

fmt(v::String, m) = (m == "method") ? method_lookup[v] : v
fmt(v::Int, m) = string(v)
fmt(v::Float64, m) = (v < 100) ? string(round(v, sigdigits=4)) : string(round(v, digits=2))

for row in eachrow(data)
	s = [fmt(row[c],c) for c in names(data)]
	line = foldl((a,b) -> a * " & " * b, s) * " \\\\\n"
	push!(table, line)
end

table_str = foldl(*, table)

println("Params:")
println(constvals)

println("Table:")
print(table_str)
