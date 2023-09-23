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

Elements of gaussian random matrix are drawn from a `N(μ=0.0, σ=1/√`target_dim`)` distribution.

# Arguments
- `target_dim`: The target dimension to be reduced to. 
- `rng`: Random number generator.
- `eps`: Sets the target_dim (if unspecified) according to the Johnson-Lindenstrauss 
lemma.  

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
- -√(s/target_dim) with probability 1/(2s)
- 0.0 with probability 1/s
- +√(s/√target_dim) with probability 1/(2s)
```
as proposed by Ping Li et al [1].

# Arguments
- `target_dim`: The target dimension to be reduced to. 
- `s`: s value defined by Achlioptas [2]. If unspecified, s = sqrt(D) where D is the source 
dimension as recommended in [1].
- `rng`: Random number generator.
- `eps`: Sets the target_dim (if unspecified) according to the Johnson-Lindenstrauss 
lemma.  

References:
[1] Ping Li, T. Hastie and K. W. Church, 2006, “Very Sparse Random Projections”.
[2] Dimitris Achlioptas. Database-friendly random projections: Johnson-Lindenstrauss 
with binary coins. Journal of Computer and System Sciences, 66(4):671–687, 2003.
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

The dimension is computed according to  
```math
    `target_dim` \\geq \frac{4ln(n_samples)}{\epsilon^2/2-\epsilon^3/3}
```
as defined in [1].

References:
[1] Sanjoy Dasgupta, Anupam Gupta. An Elementary Proof of a Theorem of Johnson and Lindenstrauss.
Random Structures and Algorithms, 22(1):60-65, 2003.
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
    fit!((M::GaussianRandomProjection, X::AbstractMatrix)

Given `X` of dimensions (n_samples, source_dim), fits `GaussianRandomProjection` with a gaussian 
random projection matrix of (source_dim, target_dim) dimensions.
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

Given `X` of dimensions (n_samples, source_dim), fits `SparseRandomProjection` with a 
sparse random projection matrix of (source_dim, target_dim) dimensions.
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
"""
function predict(M::RandomProjection, X::AbstractMatrix)
    return X * M.projection_mat  
end

"""
    fit_predict!(M::RandomProjection, X::AbstractMatrix)

Generates random projection matrix and projects data `X` to target dimension.
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



