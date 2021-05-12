using Random


"""
Approximate the maxQAP using the complete algorithm from "On the
Maximum Quadratic Assignment Problem" by Nagarajan and Sviridenko (NS).
"""
function qap_ns(A, B, obj)
	@assert obj == :max

	n = size(A, 1)
	z = 1 / (2 * n * n)
	g = Int(ceil(log2(2 * n * n)))

	A = A ./ maximum(A)
	B = B ./ maximum(B)

	A[A .<= z] .= 0
	B[B .<= z] .= 0

	best_sol = (-Inf, nothing, nothing)
	for k in 1:g
		m1, m2 = (1/2)^k, (1/2)^(k-1)
		Ak = m1 .< A .<= m2
		Bk = m1 .< B .<= m2
		P, perm = qap_ns(Ak, Bk, :max)
		obj = qap_objective(A, B, perm)
		if obj > best_sol[1]
			best_sol = (obj, P, perm)
		end
	end

	return best_sol[2], best_sol[3]
end

"""
Approximate the 0-1 QAP using the complete algorithm from "On the
Maximum Quadratic Assignment Problem" by Nagarajan and Sviridenko (NS).
"""
function qap_ns(A::Union{BitMatrix, Matrix{Bool}}, B::Union{BitMatrix, Matrix{Bool}}, obj::Symbol)
	@assert obj == :max

	P_starpacking, σ_starpacking = qap_starpacking(A, B, :max)
	P_densemapping, σ_densemapping = qap_densemapping(A, B, :max)

	o_starpacking = qap_objective(A, B, P_starpacking)
	o_densemapping = qap_objective(A, B, P_densemapping)

	if o_starpacking > o_densemapping
		return P_starpacking, σ_starpacking
	else
		return P_densemapping, σ_densemapping
	end
end

"""
Approximate the 0-1 QAP using the common star packing algorithm from "On the
Maximum Quadratic Assignment Problem" by Nagarajan and Sviridenko (NS).
"""
function qap_starpacking(A::Union{BitMatrix, Matrix{Bool}}, B::Union{BitMatrix, Matrix{Bool}}, obj::Symbol)
	@assert obj == :max
	N = size(A, 1)

	best_sol = (-Inf, nothing)
	for p in 1:N
		σ = commonstarpacking(A, B, p)
		obj = qap_objective(A, B, σ)
		if obj > best_sol[1]
			best_sol = (obj, σ)
		end
	end

	perm = best_sol[2]
	P = Matrix(I, N, N)[perm,:]

	return P, perm
end

function commonstarpacking(A, B, p)
	N = size(A, 1)

	G = map(e -> Tuple(e), findall(triu(A)))
	H = map(e -> Tuple(e), findall(triu(B)))

	ESs = fill(Tuple{Int,Int}[], p)
	ETs = fill(Tuple{Int,Int}[], p)
	σ = Dict{Tuple{Int,Int},Tuple{Int,Int}}()
	stars = fill(-1 => -1, p)
	C = fill(0, p)

	improvement = true
	while improvement
		for i in 1:p, x in 1:N, y in 1:N, c in 0:N
			improvement, update = csp_move(G, H, ESs, ETs, σ, i, x, y, c)
			if improvement
				ESs, ETs, σ = update
				stars[i] = x => y
				C[i] = c
				break
			end
		end
	end

	# all_mappings = []
	# for i in 1:p
	# 	x, y = stars[i]
	# 	if (x == -1) continue end
	# 	s = map(e -> (e[1] == x) ? e[2] : e[1], ESs[i])
	# 	t = map(e -> (e[1] == y) ? e[2] : e[1], ETs[i])
	# 	push!(all_mappings, s => t)
	# end

	mapping = filter(s -> s[1] != -1, stars)
	for (s, t) in pairs(σ)
		for (a,b) in stars
			if a in s && b in t
				m = ((a==s[1]) ? s[2] : s[1]) => ((b==t[1]) ? t[2] : t[1])
				if m[1] ∉ [p[1] for p in mapping] && m[2] ∉ [p[2] for p in mapping]
					push!(mapping, m)
				end
				break
			end
		end
	end

	perm = fill(-1, N)
	for (a,b) in mapping
		perm[a] = b
	end

	missing_keys = findall(==(-1), perm)
	missing_vals = setdiff(1:N, perm)[1:length(missing_keys)]
	perm[missing_keys] = shuffle(missing_vals)

	return perm
end

function csp_move(G, H, ESs, ETs, σ, i, x, y, c)
	ESs, ETs, σ = deepcopy(ESs), deepcopy(ETs), deepcopy(σ)
	return csp_move!(G, H, ESs, ETs, σ, i, x, y, c)
end

function csp_move!(G, H, ESs, ETs, σ, i, x, y, c)
	p = length(ESs)

	σinv = Dict(v => k for (k,v) in pairs(σ))

	# i
	sS = length(ESs[i])
	for e in ESs[i]
		delete!(σ, e)
	end
	ESs[i] = Tuple{Int,Int}[]
	ETs[i] = Tuple{Int,Int}[]

	# ii
	ES = vcat(ESs...)
	X = filter(e -> e[1] == x || e[2] == x, ES)
	sX = length(X)
	σX = [σ[e] for e in X]
	for j in 1:p
		setdiff!(ESs[j], X)
		setdiff!(ETs[j], σX)
	end

	# iii
	ET = vcat(ETs...)
	Y = filter(e -> e[1] == y || e[2] == y, ET)
	sY = length(Y)
	σinvY = [σinv[e] for e in Y]
	for j in 1:p
		setdiff!(ESs[j], σinvY)
		setdiff!(ETs[j], Y)
	end

	# iv
	ES = vcat(ESs...)
	ESv = [e[j] for e in ES, j in 1:2][:]
	A = filter(e -> e[1] == x || e[2] == x, G)
	filter!(e -> e[1] ∉ ESv && e[2] ∉ ESv, A)
	shuffle!(A)
	if length(A) < c
		return false, nothing
	elseif length(A) > c
		A = A[1:c]
	end

	# v
	ET = vcat(ETs...)
	ETv = [e[j] for e in ET, j in 1:2][:]
	B = filter(e -> e[1] == y || e[2] == y, H)
	filter!(e -> e[1] ∉ ETv && e[2] ∉ ETv, B)
	shuffle!(B)
	if length(B) < c
		return false, nothing
	elseif length(B) > c
		B = B[1:c]
	end

	# vi
	ESs[i] = A
	ETs[i] = B
	for (a,b) in zip(A, B)
		σ[a] = b
	end

	improvement = (c - sS - sX - sY) > 0

	return improvement, (ESs, ETs, σ)

end

"""
Approximate the 0-1 QAP using the dense subgraph mapping algorithm from "On the
Maximum Quadratic Assignment Problem" by Nagarajan and Sviridenko (NS).
"""
function qap_densemapping(G::Union{BitMatrix, Matrix{Bool}}, H::Union{BitMatrix, Matrix{Bool}}, obj::Symbol)
	@assert obj == :max

	C₁ = vertex_cover(G)

	N = size(G, 1)
	k = length(C₁) ÷ 2
	z = min(3k, N)

	leftovers = setdiff(1:N, C₁)
	leftover_degree_list = [(i, sum(G[i,:]) + sum(G[:,i])) for i in leftovers]
	sort!(leftover_degree_list, by=(x -> x[2]), rev=true)
	C₂ = [d[1] for d in leftover_degree_list[1:z-length(C₁)]]

	C = vcat(C₁, C₂)
	shuffle!(C)

	B = dense_ksubgraph(H, z)
	shuffle!(B)

	perm = fill(-1, N)
	perm[C] = B

	missing_keys = findall(==(-1), perm)
	missing_vals = setdiff(1:N, perm)
	perm[missing_keys] = shuffle(missing_vals)

	P = Matrix(I, N, N)[perm,:]

	return P, perm
end

function vertex_cover(G)
	G = triu(G)
	S = Int[]
	while sum(G) > 0
		edges = findall(G)
		u, v = Tuple(rand(edges))
		push!(S, u, v)
		G[u,:] .= 0
		G[:,u] .= 0
		G[v,:] .= 0
		G[:,v] .= 0
	end
	return S
end

function dense_ksubgraph(G, k)
	v1 = dense_ksubgraph_trivial(G, k)
	v2 = dense_ksubgraph_greedy(G, k)
	v3 = dense_ksubgraph_walks(G, k)

	s1 = sum(G[v1,v1])
	s2 = sum(G[v2,v2])
	s3 = sum(G[v3,v3])
	s = max(s1, s2, s3)

	if s1 == s
		return v1
	elseif s2 == s
		return v2
	else
		return v3
	end
end

function dense_ksubgraph_trivial(G, k)
	N = size(G, 1)
	G = triu(G)

	edges = Tuple.(findall(G))
	shuffle!(edges)

	E = min(length(edges), Int(floor(k/2)))
	edges = edges[1:E]

	vertices = [e[i] for e in edges, i in [1,2]][:]
	vertices = unique(vertices)

	leftovers = setdiff(1:N, vertices)
	shuffle!(leftovers)
	append!(vertices, leftovers[1:k-length(vertices)])

	return vertices
end

function dense_ksubgraph_greedy(G, k)
	N = size(G, 1)

	degree_list = [(i, sum(G[i,:]) + sum(G[:,i])) for i in 1:N]
	sort!(degree_list, by=(x -> x[2]), rev=true)
	H = [d[1] for d in degree_list[1:round(Int, k/2)]]

	leftovers = setdiff(1:N, H)
	hdegree_list = [(i, sum(G[i,H]) + sum(G[H,i])) for i in leftovers]
	sort!(hdegree_list, by=(x -> x[2]), rev=true)
	C = [d[1] for d in hdegree_list[1:(k-length(H))]]

	return vcat(H, C)
end

function dense_ksubgraph_walks(G, k)
	return randperm(size(G, 1))[1:k]
end
