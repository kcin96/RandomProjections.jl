abstract type RandomProjection end

mutable struct GaussianRandomProjection{T<:Real} <: RandomProjection
    source_dim::Union{Integer,Nothing}
    target_dim::Union{Integer,Nothing}
    projection_mat::AbstractMatrix{T}
    rng::AbstractRNG
    eps::AbstractFloat
end

mutable struct SparseRandomProjection{T<:Real} <: RandomProjection
    source_dim::Union{Integer,Nothing}
    target_dim::Union{Integer,Nothing}
    projection_mat::AbstractMatrix{T}
    s::Union{Real,Nothing} 
    rng::AbstractRNG
    eps::AbstractFloat 
end

# constructors
"""
    GaussianRandomProjection(target_dim::Union{Integer,Nothing} = nothing;
                            rng::AbstractRNG = Random.default_rng(), eps = 0.1)

Initializes `GaussianRandomProjection` for gaussian random projections.

Elements of gaussian random matrix are drawn from a ``N(μ=0.0, σ=1/\\sqrt{target\\_dim})`` distribution.

# Arguments
- `target_dim`: The target dimension to be reduced to. 
- `rng`: Random number generator.
- `eps`: Sets the target_dim (if unspecified) according to the Johnson-Lindenstrauss lemma.  

# Examples
```jldoctest
julia> using RandomProjections
julia> GaussianRandomProjection()
GaussianRandomProjection{Float64}(nothing, nothing, [0.0;;], Random.TaskLocalRNG(), 0.1)

julia> GaussianRandomProjection(1000)
GaussianRandomProjection{Float64}(nothing, 1000, [0.0;;], Random.TaskLocalRNG(), 0.1)

julia> using Random
julia> rng = MersenneTwister(9)
MersenneTwister(9)
julia> GaussianRandomProjection(1000,rng=rng, eps=0.5)
GaussianRandomProjection{Float64}(nothing, 1000, [0.0;;], MersenneTwister(9), 0.5)
```
"""
function GaussianRandomProjection(target_dim::Union{Integer,Nothing} = nothing; 
                                    rng::AbstractRNG = Random.default_rng(), 
                                    eps = 0.1)

    #initialize source dimension to nothing
    source_dim = nothing
    #initialize projection matrix with zeros
    projection_matrix = zeros(1,1)   
    return GaussianRandomProjection(source_dim, target_dim, projection_matrix, rng, eps)
end

"""
    SparseRandomProjection(target_dim::Union{Integer,Nothing} = nothing; 
                            s::Union{Real,Nothing} = nothing, 
                            rng::AbstractRNG = Random.default_rng(), eps = 0.1)

Initializes `SparseRandomProjection` for sparse random projections.

Elements of sparse random matrix are set to 
```math 
\\begin{cases}
 -\\sqrt{\\frac{s}{target\\_dim}} \\qquad with \\ probability \\ \\frac{1}{2s} \\\\
 0.0 \\ \\qquad\\qquad\\qquad with \\ probability \\ \\frac{1}{s} \\\\
 +\\sqrt{\\frac{s}{target\\_dim}} \\qquad with \\ probability \\ \\frac{1}{2s}
 \\end{cases}
```
as proposed by Ping Li et al. [1].

# Arguments
- `target_dim`: The target dimension to be reduced to. 
- `s`: s value defined by Achlioptas [2]. If unspecified, ``s = \\sqrt{D}`` where D is the source dimension as recommended in [1].
- `rng`: Random number generator.
- `eps`: Sets the target_dim (if unspecified) according to the Johnson-Lindenstrauss lemma.  

# References:
[1] Ping Li, T. Hastie and K. W. Church, 2006, “Very Sparse Random Projections”. \\
[2] Dimitris Achlioptas. Database-friendly random projections: Johnson-Lindenstrauss 
with binary coins. Journal of Computer and System Sciences, 66(4):671–687, 2003.

# Examples
```jldoctest
julia> SparseRandomProjection()
SparseRandomProjection{Float64}(nothing, nothing, [0.0;;], nothing, TaskLocalRNG(), 0.1)

julia> SparseRandomProjection(1000)
SparseRandomProjection{Float64}(nothing, 1000, [0.0;;], nothing, TaskLocalRNG(), 0.1)

julia> using Random
julia> rng = MersenneTwister(9)
MersenneTwister(9)
julia> SparseRandomProjection(1000,rng=rng, eps=0.5)
SparseRandomProjection{Float64}(nothing, 1000, [0.0;;], nothing, MersenneTwister(9), 0.5)
```
"""
function SparseRandomProjection(target_dim::Union{Integer,Nothing} = nothing; 
                                s::Union{Real,Nothing} = nothing, 
                                rng::AbstractRNG = Random.default_rng(), 
                                eps = 0.1)

    #initialize source dimension to nothing
    source_dim = nothing
    #initialize projection matrix with zeros
    projection_matrix = zeros(1,1) 
    return SparseRandomProjection(source_dim, target_dim, projection_matrix, s, rng, eps)
end

# properties 
"""
    projection(M::RandomProjection)

Returns the projection matrix. 
"""
projection(M::RandomProjection) = M.projection_mat

"""
    projection_size(M::RandomProjection)

Returns the size of projection matrix. 
"""
projection_size(M::RandomProjection) = Base.size(M.projection_mat)

# functions
"""
    johnson_lindenstrauss_min_dim(n_samples, eps)

Returns the minimum dimension to be projected onto that satifies the Johnson-Lindenstrauss lemma.

The dimension is computed from  
`` target\\_dim \\geq \\frac{4ln(n\\_samples)}{\\frac{\\epsilon ^2}{2}-\\frac{\\epsilon ^3}{3}}``
as defined in [1].

# References:
[1] Sanjoy Dasgupta, Anupam Gupta. An Elementary Proof of a Theorem of Johnson and Lindenstrauss.
Random Structures and Algorithms, 22(1):60-65, 2003.

# Examples
```jldoctest
julia> johnson_lindenstrauss_min_dim(8000, 0.5)
431

julia> johnson_lindenstrauss_min_dim(1000, [0.1,0.2,0.7,0.9])
4-element Vector{Int64}:
 5920
 1594
  211
  170
```
"""
function johnson_lindenstrauss_min_dim(n_samples, eps)
    if n_samples <= 0
        throw(ArgumentError("Number of samples should be greater than zero.n_samples = $n_samples provided."))
    end
    if any(eps .<= 0.0) || any(eps .>= 1.0) 
        throw(ArgumentError("eps not in the range (0,1). eps = $eps provided."))
    end
    return floor.(Int64,(4 .* log(n_samples) ./(eps .^2 ./2 .- eps .^3 ./ 3)))
end

# Returns a (source_dim x target_dim) gaussian matrix
function gaussian_random_matrix(source_dim, target_dim, rng::AbstractRNG)
    check_input_size(source_dim, target_dim)
    d = Normal(0.0, 1/sqrt(target_dim))
    return Random.rand(rng,d,(source_dim, target_dim))
end

# Returns a (source_dim x target_dim) sparse matrix
function sparse_random_matrix(source_dim, target_dim, s::Real, rng::AbstractRNG)
    check_input_size(source_dim, target_dim)
    check_s_factor(s)
    if s == 1
        return 1/sqrt(target_dim) .* (Random.rand(rng, Distributions.Binomial(1,0.5),(source_dim, target_dim)) .* 2 .- 1)
    
    else
        rowindex::Vector{Int64} = []
        offset::Int64 = 1
        colptr::Vector{Int64} = [offset]
        for _ in 1:target_dim
            non_zero_i = Random.rand(rng, Distributions.Binomial(source_dim, 1/s))
            indices_i = StatsBase.sample(rng, 1:source_dim, non_zero_i, replace = false)
            reduce(append!,(rowindex,indices_i))
            offset += non_zero_i
            push!(colptr, offset)


        end

        val = (Random.rand(rng, Distributions.Binomial(1,0.5), length(rowindex)) .* 2 .- 1)
        sparse_mat = SparseMatrixCSC(source_dim, target_dim, colptr, rowindex, val)
        return sqrt(s/target_dim) .* sparse_mat
    end
end

"""
    fit!(M::GaussianRandomProjection, X::AbstractMatrix)

Given `X` of dimensions (n\\_samples, source\\_dim), fits `GaussianRandomProjection` with a gaussian 
random projection matrix of (source\\_dim, target\\_dim) dimensions.

# Examples
```jldoctest
julia> X = ones(100,7000);
julia> rng = MersenneTwister(9)
MersenneTwister(9)
julia> gaussian_model = GaussianRandomProjection(500;rng=rng,eps=0.2)
GaussianRandomProjection{Float64}(nothing, 500, [0.0;;], MersenneTwister(9), 0.2)
julia> fit!(gaussian_model,X);
julia> projection_size(gaussian_model)
(7000, 500)
```
"""
function fit!(M::GaussianRandomProjection, X::AbstractMatrix)
    (n_s, source_dim) = Base.size(X)
    M.source_dim = source_dim    
    if M.target_dim === nothing
        M.target_dim = johnson_lindenstrauss_min_dim(n_s, M.eps)
    end
    if !check_source_target_size(M.source_dim, M.target_dim)
        @warn ("Source dimension less than target dimension to be reduced to. \
                source_dim = $(M.source_dim), target_dim = $(M.target_dim). \
                Dimension will not be reduced.")
    end
    M.projection_mat = gaussian_random_matrix(M.source_dim, M.target_dim, M.rng)
end

"""
    fit!(M::SparseRandomProjection, X::AbstractMatrix)

Given `X` of dimensions (n\\_samples, source\\_dim), fits `SparseRandomProjection` with a 
sparse random projection matrix of (source\\_dim, target\\_dim) dimensions.

# Examples
```jldoctest
julia> X = ones(100,7000);
julia> rng = MersenneTwister(9)
MersenneTwister(9)
julia> sparse_model = SparseRandomProjection(100;rng=rng)
SparseRandomProjection{Float64}(nothing, 100, [0.0;;], nothing, MersenneTwister(9), 0.1)
julia> fit!(sparse_model,X);
julia> projection_size(sparse_model)
(7000, 100)
```
"""
function fit!(M::SparseRandomProjection, X::AbstractMatrix)
    (n_s, source_dim) = Base.size(X)
    M.source_dim = source_dim
    if M.target_dim === nothing
        M.target_dim = johnson_lindenstrauss_min_dim(n_s, M.eps)
    end
    if !check_source_target_size(M.source_dim, M.target_dim)
        @warn ("Source dimension less than target dimension to be reduced to. \
                source_dim = $(M.source_dim), target_dim = $(M.target_dim). \
                Dimension will not be reduced.")
    end
    M.s = (M.s === nothing) ? sqrt(M.source_dim) : M.s
    M.projection_mat = sparse_random_matrix(M.source_dim, M.target_dim, M.s, M.rng)  
end

"""
    predict(M::RandomProjection, X::AbstractMatrix)

Projects data `X` to target dimension.

# Examples
```jldoctest
julia> X = ones(100,7000);
julia> rng = MersenneTwister(9)
MersenneTwister(9)
julia> sparse_model = SparseRandomProjection(100;rng=rng)
SparseRandomProjection{Float64}(nothing, 100, [0.0;;], nothing, MersenneTwister(9), 0.1)
julia> fit!(sparse_model,X);
julia> p = predict(sparse_model,10X);
julia> size(p)
(100, 100)
```
"""
function predict(M::RandomProjection, X::AbstractMatrix)
    return X * M.projection_mat  
end

"""
    fit_predict!(M::RandomProjection, X::AbstractMatrix)

Generates random projection matrix and projects data `X` to target dimension.

# Examples
```jldoctest
julia> X = ones(3,10);
julia> rng = MersenneTwister(9)
MersenneTwister(9)
julia> sparse_model = SparseRandomProjection(5;rng=rng)
SparseRandomProjection{Float64}(nothing, 5, [0.0;;], nothing, MersenneTwister(9), 0.1)
julia> fit_predict!(sparse_model,X)
3×5 Matrix{Float64}:
 -1.59054  0.795271  …  -0.795271  1.59054
 -1.59054  0.795271     -0.795271  1.59054
 -1.59054  0.795271     -0.795271  1.59054
```
"""
function fit_predict!(M::RandomProjection, X::AbstractMatrix)
    fit!(M, X)
    return predict(M, X)
end

"""
    inverse_projection(M::RandomProjection)

Returns the psuedo-inverse projection matrix.
"""
function inverse_projection(M::RandomProjection)
    return pinv(Matrix(M.projection_mat))
end



