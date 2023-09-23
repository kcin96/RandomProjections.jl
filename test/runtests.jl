using RandomProjections
using Test
using Random, Statistics
import StatsBase.countmap

@testset "RandomProjections.jl" begin
    # Auxiliary tests here.
    @testset "auxiliary" begin
        @test_throws ArgumentError RandomProjections.check_input_size(0, 0)
        @test_throws ArgumentError RandomProjections.check_input_size(-1, 10)
        @test_throws ArgumentError RandomProjections.check_input_size(1, -10)

        @test RandomProjections.check_s_factor(1) === nothing
        @test RandomProjections.check_s_factor(3.0) === nothing
        @test_throws ArgumentError RandomProjections.check_s_factor(-0.5)

        @test RandomProjections.check_source_target_size(5, 4) == true
        @test RandomProjections.check_source_target_size(4, 5) == false
        @test RandomProjections.check_source_target_size(4, 4) == false
    end 

    # Function tests
    @testset "johnson_lindenstrauss_min_dim" begin
        @test johnson_lindenstrauss_min_dim(8000, 0.1) == 7703
        @test johnson_lindenstrauss_min_dim(8000, 0.5) == 431
        @test johnson_lindenstrauss_min_dim(8000, 0.9) == 221
        @test johnson_lindenstrauss_min_dim(8000.0, 0.9) == 221
        @test johnson_lindenstrauss_min_dim(1000, [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]) == [5920, 1594, 767, 331, 211, 170] 
        @test_throws ArgumentError johnson_lindenstrauss_min_dim(0, 0.1)
        @test_throws ArgumentError johnson_lindenstrauss_min_dim(-1.0, 0.1)
        @test_throws ArgumentError johnson_lindenstrauss_min_dim(0, 0)
        @test_throws ArgumentError johnson_lindenstrauss_min_dim(10, [0.1, -0.2, 0.3])
        @test_throws ArgumentError johnson_lindenstrauss_min_dim(10, [1.1, 0.2, 0.3])
    end

    @testset "gaussian_random_matrix" begin
        rng = MersenneTwister(43)
        @test isapprox(mean(RandomProjections.gaussian_random_matrix(5, 5, rng)), 0.0; atol = 0.1)  
        @test isapprox(std(RandomProjections.gaussian_random_matrix(5, 5, rng)), 1/sqrt(5); atol = 0.1) 
    end

    @testset "sparse_random_matrix" begin
        #test with s < 1
        @testset begin
            s = 0.4
            rng = MersenneTwister(43)
            @test_throws ArgumentError RandomProjections.sparse_random_matrix(500, 500, s, rng)
        end

        # test with s = 1
        @testset begin
            s = 1
            rng = MersenneTwister(43)
            result = RandomProjections.sparse_random_matrix(500, 500, s, rng)
            @test Base.size(result) == (500, 500)
            res_dict = countmap(result)
            for k in keys(res_dict)
                @test isapprox(res_dict[k]/(500*500), 0.5; atol = 0.1)  #Checks that the 2 keys have 50%/50% split
            end
        end 

        # test with s = 3
        @testset begin
            s = 3
            rng = MersenneTwister(43)
            result = RandomProjections.sparse_random_matrix(100, 10, s, rng)
            @test Base.size(result) == (100, 10)
            res_dict = countmap(reduce(vcat,result))
            for k in keys(res_dict)
                if convert(Float64, k) == 0  
                    @test isapprox(res_dict[k]/(100*10), 1-1/s; atol = 0.1)  #Checks that 0 has prob: 1-1/s 
                else
                    @test isapprox(res_dict[k]/(100*10), 1/(2s); atol = 0.1) #Checks that the +1 -1 keys have prob: 1/(2s) 
                end
            end
        end 
    end     

    # Gaussian projection tests
    @testset "Gaussian projection" begin
        @testset begin 
            # test initialization 
            g0 = GaussianRandomProjection() # Calls function GaussianRandomProjection()
            rng = Random.default_rng()
            g1 = RandomProjections.GaussianRandomProjection(nothing, nothing, zeros(1,1), rng, 0.1)

            @test g0.target_dim == g1.target_dim
            @test g0.source_dim == g1.source_dim
            @test g0.projection_mat == g1.projection_mat
            @test g0.eps == g1.eps

            # test fit!() for source_dim < target_dim
            n_samples = 8000
            source_dim = 100
            target_dim = johnson_lindenstrauss_min_dim(n_samples, 0.1)
            X = ones(n_samples, source_dim)  #(n_samples, source_dim)
            @test_logs (:warn, "Source dimension less than target dimension to be reduced to. \
                         source_dim = $source_dim, target_dim = $target_dim. \
                         Dimension will not be reduced.") fit!(g0, X)   # throws warning as source_dim < target_dim
            
            # test fit!() for source_dim > target_dim 
            n_samples = 8000
            X = ones(n_samples,10000)
            fit!(g0, X)   
            @test g0.target_dim == johnson_lindenstrauss_min_dim(n_samples, 0.1)
            @test g0.source_dim == 10000
            @test Base.size(g0.projection_mat) == (g0.source_dim, g0.target_dim)
        end

        @testset begin 
            # test initialization with rng argument
            rng = MersenneTwister(43)
            g0 = GaussianRandomProjection(rng = rng) #Calls function GaussianRandomProjection(rng = rng)
            rng = MersenneTwister(43)
            g1 = RandomProjections.GaussianRandomProjection(nothing, nothing, zeros(1,1), rng, 0.1)

            @test g0.target_dim == g1.target_dim
            @test g0.source_dim == g1.source_dim
            @test g0.projection_mat == g1.projection_mat
            @test g0.rng == g1.rng
            @test g0.eps == g1.eps
        end

        @testset begin
            # test initialization with target_dim, rng and eps argument 
            rng = MersenneTwister(43)
            g0 = GaussianRandomProjection(5, rng = rng, eps = 0.5) #Calls function GaussianRandomProjection(5, rng = rng, eps = 0.5) 
            rng = MersenneTwister(43)
            g1 = RandomProjections.GaussianRandomProjection(nothing, 5, zeros(1,1), rng, 0.5)

            @test g0.target_dim == g1.target_dim
            @test g0.source_dim == g1.source_dim
            @test g0.projection_mat == g1.projection_mat
            @test g0.rng == g1.rng
            @test g0.eps == g1.eps

            # test fit!() and predict()
            n_samples = 10
            X = ones(n_samples,100)
            fit!(g0, X)
            @test Base.size(g0.projection_mat) == (100,5) 
            result = predict(g0, 3X)
            @test Base.size(result) == (10, 5)

            # test fit_predict()
            rng = MersenneTwister(43)
            g0 = GaussianRandomProjection(5, rng = rng, eps = 0.5) #Calls function GaussianRandomProjection(5, rng = rng, eps = 0.5) 
            @test result == fit_predict!(g0, 3X)
        end
    end

    # Sparse projection tests
    @testset "Sparse projection" begin
        @testset begin 
            # test initialization 
            g0 = SparseRandomProjection() # Calls function SparseRandomProjection()
            rng = Random.default_rng()
            g1 = RandomProjections.SparseRandomProjection(nothing, nothing, zeros(1,1), nothing, rng, 0.1)

            @test g0.target_dim == g1.target_dim
            @test g0.source_dim == g1.source_dim
            @test g0.projection_mat == g1.projection_mat
            @test g0.s == g1.source_dim
            @test g0.eps == g1.eps

            # test fit!() for source_dim < target_dim
            n_samples = 8000
            source_dim = 100
            target_dim = johnson_lindenstrauss_min_dim(n_samples, 0.1)
            X = ones(n_samples, source_dim)  #(n_samples, source_dim)
            @test_logs (:warn, "Source dimension less than target dimension to be reduced to. \
                            source_dim = $source_dim, target_dim = $target_dim. \
                            Dimension will not be reduced.") fit!(g0, X)   # throws warning as source_dim < target_dim
            
            # test fit!() for source_dim > target_dim
            g0 = SparseRandomProjection() 
            n_samples = 8000
            X = ones(n_samples,10000)
            fit!(g0, X)   
            @test g0.target_dim == johnson_lindenstrauss_min_dim(n_samples, 0.1)
            @test g0.source_dim == 10000
            @test Base.size(g0.projection_mat) == (g0.source_dim, g0.target_dim)
            @test g0.s == sqrt(g0.source_dim)
        end

        @testset begin 
            # test initialization with rng argument
            rng = MersenneTwister(43)
            g0 = SparseRandomProjection(rng = rng) #Calls function SparseRandomProjection(rng = rng)
            rng = MersenneTwister(43)
            g1 = RandomProjections.SparseRandomProjection(nothing, nothing, zeros(1,1), nothing, rng, 0.1)

            @test g0.target_dim == g1.target_dim
            @test g0.source_dim == g1.source_dim
            @test g0.projection_mat == g1.projection_mat
            @test g0.s == g1.source_dim
            @test g0.rng == g1.rng
            @test g0.eps == g1.eps
        end

        @testset begin
            # test initialization with target_dim, s, rng and eps argument 
            rng = MersenneTwister(43)
            g0 = SparseRandomProjection(5, s = 4, rng = rng, eps = 0.5) #Calls function SparseRandomProjection(5, rng = rng, eps = 0.5) 
            rng = MersenneTwister(43)
            g1 = RandomProjections.SparseRandomProjection(nothing, 5, zeros(1,1), 4, rng, 0.5)

            @test g0.target_dim == g1.target_dim
            @test g0.source_dim == g1.source_dim
            @test g0.projection_mat == g1.projection_mat
            @test g0.s == g1.s
            @test g0.rng == g1.rng
            @test g0.eps == g1.eps

            # test fit!() and predict()
            n_samples = 10
            X = ones(n_samples,100)
            fit!(g0, X)
            @test Base.size(g0.projection_mat) == (100,5) 
            result = predict(g0, 3X)
            @test Base.size(result) == (10, 5)
            @test g0.s == 4    

            # test fit_predict()
            rng = MersenneTwister(43)
            g0 = SparseRandomProjection(5, s = 4, rng = rng, eps = 0.5) #Calls function GaussianRandomProjection(5, rng = rng, eps = 0.5) 
            @test result == fit_predict!(g0, 3X)
        end
    end

    # Inverse_projection tests 
    @testset "inverse_projection"  begin
        g0 = GaussianRandomProjection(5, eps = 0.5)
        # test inverse_projection
        n_samples = 10
        X = ones(n_samples,5)
        fit!(g0, X)
        result = predict(g0, 2X)
        @test isapprox(result * inverse_projection(g0), 2X; atol=0.1)
    end  
    
end
