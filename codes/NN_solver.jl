using Optim
include("NN_util.jl")

mutable struct NNModel
    Nvar     # number of ODE equations
    f_list   # f(t, y) on the right hand side
    y0_list  # initial conditions
    t        # training points
    t0       # initial t, i.e. t[1]
    n_hidden # number of hidden units

    params_list    # a list of NN parameters
    sizes   # size of NN parameters, for flatten operation
    lengths # length of NN parameters, for flatten operation
    total_l  # total length of NN parameters, for flatten operation
    all_flattened_params # flattenned parameters, for Optim.jl
end

function show(nn::NNModel)
    println("Neural ODE Solver")
    println("Number of equations:       ", nn.Nvar)
    println("Initial condition y0:      ", nn.y0_list)
    println("Numnber of hidden units:   ", nn.n_hidden)
    println("Number of training points: ", length(nn.t))
end

function pre_init_nn(f!, t, y0_list; n_hidden=10)
    Nvar = length(y0_list)
    t0 = t[1]

    function f_list(t, y)
        dy = Vector{}(length(y))
        f!(t, y, dy)
        return dy
    end

    return NNModel(Nvar, f_list, y0_list, t, t0, n_hidden,
                   nothing, nothing, nothing, nothing, nothing)
end

function reset_weights!(nn::NNModel)
    # update weights
    nn.params_list = [init_weights(n_hidden = nn.n_hidden)
                      for _ in 1:nn.Nvar]
    nn.sizes, nn.lengths, nn.total_l = sizes_and_length(nn.params_list[1])

    # update flattened weights
    nn.all_flattened_params = vcat(map(flat_opt, nn.params_list)...)
    return nothing
end

function init_nn(f!, t, y0_list; n_hidden=10)
    nn = pre_init_nn(f!, t, y0_list, n_hidden=n_hidden)
    reset_weights!(nn)
    return nn
end

function loss_func(params_list, nn::NNModel)
    # need to expose params_list to optim.jl

    # shortcut
    Nvar = nn.Nvar
    f_list = nn.f_list
    t = nn.t
    t0 = nn.t0

    y_pred_list = []
    dydt_pred_list = []

    for i = 1:Nvar
        params = params_list[i]
        y0 = y0_list[i]
        y_pred =  predict(params, t, y0, t0)
        dydt_pred = predict_dt(params, t, y0, t0)

        push!(y_pred_list, y_pred)
        push!(dydt_pred_list, dydt_pred)
    end

    f_pred_list = f_list(t, y_pred_list)

    loss_total = 0.0
    for i = 1:Nvar
        f_pred = f_pred_list[i]
        dydt_pred = dydt_pred_list[i]
        loss = mean(abs2, dydt_pred - f_pred)
        loss_total += loss
    end
    return loss_total
end

function get_unflat(all_flattened_params, nn::NNModel)
    # don't modify nn itself
    return unflatten_all(all_flattened_params,
        nn.Nvar, nn.total_l, nn.sizes, nn.lengths)
end

function train!(nn::NNModel, method; kwargs...)
    # for optim.jl
    function loss_wrap(all_flattened_params)
        params_list = get_unflat(all_flattened_params, nn)
        return loss_func(params_list, nn)
    end

    # configuration optimazation options
    od = OnceDifferentiable(loss_wrap, nn.all_flattened_params; autodiff =:forward);
    println(kwargs)
    option = Optim.Options(;store_trace=true, extended_trace=true, kwargs...)

    # training
    res = optimize(od, nn.all_flattened_params, method, option)

    # update weights
    nn.all_flattened_params = res.minimizer # flattened weights
    nn.params_list = get_unflat(res.minimizer, nn) # original weights

    return res
end

function predict(nn::NNModel; t=nothing)
    if t == nothing
        t = nn.t # predict on training points by default
    end

    # shortcut
    Nvar = nn.Nvar
    f_list = nn.f_list
    params_list = nn.params_list
    t0 = nn.t0

    y_pred_list = []
    dydt_pred_list = []

    for i = 1:Nvar
        params = params_list[i]
        y0 = y0_list[i]
        y_pred =  predict(params, t, y0, t0)
        dydt_pred = predict_dt(params, t, y0, t0)

        push!(y_pred_list, y_pred)
        push!(dydt_pred_list, dydt_pred)
    end

    return y_pred_list, dydt_pred_list
end
