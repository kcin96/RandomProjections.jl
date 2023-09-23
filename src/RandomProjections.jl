module RandomProjections

using Random, Distributions, SparseArrays, StatsBase, LinearAlgebra

# exported interfaces
export 
    GaussianRandomProjection,
    SparseRandomProjection,
    projection,
    projection_size,
    johnson_lindenstrauss_min_dim,
    fit!,
    predict,
    fit_predict!,
    inverse_projection

# source files
include("randomprojection.jl")
include("auxiliary.jl")

end # module
