# Auxiliary functions

# Checks that source_dim and target_dim > 0
function check_input_size(source_dim, target_dim)
    if source_dim <= 0
        throw(ArgumentError("Source dimension should be greater than zero.source_dim = $source_dim provided."))
    end
    if target_dim <= 0
        throw(ArgumentError("Target dimension should be greater than zero.target_dim = $target_dim provided."))
    end
end

# Checks that s >= 1
function check_s_factor(s) 
    if s < 1 
        throw(ArgumentError("s should be greater than or equal to 1. s = $s provided"))
    end
end

# Checks that target dimension to be reduced to is less than original source dimension.
function check_source_target_size(source_dim, target_dim)::Bool
    return target_dim < source_dim ? true : false
end