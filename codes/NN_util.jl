using AutoGrad

# === building blocks for NN ===

function init_weights(;n_in=1, n_hidden=10, n_out=1)
    W1 = randn(n_hidden, n_in) # for left multiply W1*x
    b1 = zeros(n_hidden)
    W2 = randn(n_out, n_hidden)
    b2 = zeros(n_out)
    params = [W1, b1, W2, b2]
    return params
end

function sizes_and_length(params)
    sizes = map(size, params)
    lengths = map(length, params)
    total_l = sum(lengths)
    return sizes, lengths, total_l
end

function predict(params, t, y0, t0; act=tanh)
    W1, b1, W2, b2 = params

    # normal NN calculation
    a = act.(W1*t .+ b1)
    out = W2*a .+ b2

    # force intial condition
    y = y0 .+ (t .- t0) .* out
    return y
end

# gradient w.r.t t
# get diagonal of jacobian to vectorize over t
predict_sum(params, t, y0, t0) = sum(predict(params, t, y0, t0))
predict_dt = grad(predict_sum, 2)

# === flatten & unflatten ===

flat_opt = p->collect(Iterators.flatten(p)) # flatten operation for a single NN params

function unflatten(params_flat, sizes, lengths)
    params = []
    i1 = 1
    for j in 1:length(sizes)
        s = sizes[j]
        l = lengths[j]
        i2 = i1+l
        #p = reshape(params_flat[i1:i2-1], s)
        p = reshape(view(params_flat,i1:i2-1), s)
        push!(params, p)
        i1 = i2
    end
    return params
end

function unflatten_all(all_flattened_params, Nvar, total_l, sizes, lengths)
    params_list = []
    all_params_reshape = reshape(all_flattened_params, total_l, Nvar)
    for i in 1:Nvar
        params = unflatten(all_params_reshape[:,i], sizes, lengths)
        push!(params_list, params)
    end
    return params_list
end
