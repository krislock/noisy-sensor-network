using LinearAlgebra
using Statistics
using Printf
using Plots, LaTeXStrings
using JuMP, SCS
using Zygote
using SparseArrays
using Arpack


macro javascript_str(s) display("text/javascript", s); end

javascript"""
MathJax.Hub.Config({
    TeX: { equationNumbers: { autoNumber: "AMS" } }
});
"""


function genprob(n, m, r)
    P = rand(n, r)
    
    # Center points at origin
    P̄ = mean(P, dims=1)
    P .-= P̄
    
    # Sensor positions
    S = P[1:n-m,:]
    
    # Anchor positions
    A = P[n-m+1:n,:]
    
    S, A
end


function edges(P, R, m)
    n, r = size(P)
    E = CartesianIndex{r}[]
    d = Float64[]
    for i = 1:n
        for j = i+1:n
            dij = norm(P[i,:] - P[j,:])
            if dij < R || min(i,j) > n - m
                push!(E, CartesianIndex(i,j))
                push!(d, dij^2)
            end
        end
    end
    E, d
end


function K(X)
    n = size(X,1)
    [X[i,i] + X[j,j] - 2X[i,j] for i=1:n, j=1:n]
end


function K(X, E)
    T = typeof(X[1,1] + X[2,2] - 2X[1,2])
    d2 = T[]
    for e in E
        i, j = Tuple(e)
        push!(d2, X[i,i] + X[j,j] - 2X[i,j])
    end
    d2
end


function plot_network(P, E, m; plot_edges=true, lims=(-0.6,0.6), legend=:outerright)
    
    n = size(P, 1)
    
    # sensors and anchors
    S = P[1:n-m, :]
    A = P[n-m+1:n, :]

    plt = plot(aspect_ratio=:equal, lims=lims, legend=legend)
    if plot_edges
        for e in E
            i, j = Tuple(e)
            
            # Color inter-anchor edges red
            c = min(i,j) > n - m ? :red : :black
            plot!(P[[i j],1]', P[[i j],2]', label=:none, c=c)
        end
    end
    scatter!(S[:,1], S[:,2], c=:white, ms=3, label="sensors")
    scatter!(A[:,1], A[:,2], c=:red, m=:square, ms=3, label="anchors")
    
    plt
end


function plot_errors(P, S̄, m; lims=(-0.6,0.6), plot_anchors=false, legend=:outerright)
    
    n = size(P, 1)
    
    # sensors and anchors
    S = P[1:n-m, :]
    A = P[n-m+1:n, :]
    
    P̄ = [S̄; A]

    plt = plot(aspect_ratio=:equal, lims=lims, size=(900,900), legend=legend)
    for i = 1:n-m
        plot!([P[i,1], P̄[i,1]], [P[i,2], P̄[i,2]], label=:none, c=:red)
    end
    scatter!(S[:,1], S[:,2], c=:white, label="true positions")
    if plot_anchors
        scatter!(A[:,1], A[:,2], m=:square, c=:red, label="anchor")
    end
    scatter!(S̄[:,1], S̄[:,2], m=:plus, c=:red, label="estimate")    
    
    plt
end


inter_anchor(E, n, m) = [min(Tuple(e)...) > n - m for e in E]


function add_noise(E, d, p, n, m)
    # Noise factor
    nf = (1 .+ p*randn(length(d))).^2
    
    # No noise on inter-anchor distances
    ainds = inter_anchor(E, n, m)
    nf[ainds] .= 1

    # Multiplicative noise
    nd = nf.*d
end


function noise_stats(E, d, p, n, m; N=10000)
    
    errors = zeros(N)
    stds = zeros(N)

    ainds = inter_anchor(E, n, m)
    yinds = ainds .|> !

    dy = d[yinds]

    for i = 1:N
        nd = add_noise(E, d, p, n, m)

        ndy = nd[yinds]

        errors[i] = norm(ndy - dy)
        stds[i] = std((ndy - dy)./dy)
    end
    
    errors, stds
end


function opt_trace(E, nd, errtol, n, m; sense=:max)

    optimizer_constructor = optimizer_with_attributes(SCS.Optimizer,
            "eps" => 1e-6,
            "verbose" => 0)
    model = Model(optimizer_constructor)

    @variable(model, X[1:n,1:n], PSD)
    
    if sense == :max
        @objective(model, Max, tr(X))
    else
        @objective(model, Min, tr(X))
    end
    
    # norm(K(X,E) - nd) ≤ errtol
    @constraint(model, [errtol; K(X,E) - nd] in SecondOrderCone())
    
    @constraint(model, sum(X, dims=2) .== 0)
    optimize!(model)
    
    Xopt = value.(X)
end


function procrustes(A, B)
    U, ~, V = svd!(B'A)
    Q = U*V'
end 


function align_anchors(A, P̄, m)
    n = size(P̄, 1)
    
    Ā = P̄[n-m+1:n,:]

    v = mean(A, dims=1)
    v̄ = mean(Ā, dims=1)

    Q = procrustes(A .- v, Ā .- v̄)

    P̄aligned = (P̄ .- v̄)*Q .+ v
end


function low_rank_soln(X; r=2)
    F = eigen(X)
    P = F.vectors[:,n-1:n]*Diagonal(sqrt.(F.values[n-1:n]))
end


function rmsd(S, Strue)
    n = size(S,1)
    norm(S - Strue)/sqrt(n)
end


function q(S)
    P = [S; A]
    error = 0.0
    N = length(E)
    for k = 1:N
        i, j = Tuple(E[k])
        dij = dnoisy[k]
        d = norm(P[i,:] - P[j,:])
        error += ((d - sqrt(dij))/sqrt(dij))^2
    end
    error
end


function refine(S, Strue; N=500)
    
    S̄ = copy(S)
    qS, dS = q(S̄), q'(S̄)

    for k = 1:N
        
        # Armijo-Goldstein backtracking linesearch
        t, α, β = 1.0, 0.5, 0.5
        qval = q(S̄ - t*dS)
        while qval > qS - α*t*dot(dS,S̄)
            t *= β
            qval = q(S̄ - t*dS)
        end

        #=
        plt = plot(legend=:none)
        tmax = 2t
        plot!(t -> q(S̄ - t*dS), 0, tmax)
        plot!([0, tmax], [qS, qS - α*tmax*dot(dS,S̄)])
        scatter!([0, t], [qS, qval], c=1)
        display(plt)
        =#

        # Update
        S̄ -= t*dS
        qS, dS = qval, q'(S̄)

        if k % 1000 == 1
            @printf("\n%4s %8s %12s %12s %12s\n", "k", "t", "qval", "RMSD", "norm(dS)")
        end
        if k % 100 == 0 || (N <= 100 && k % 10 == 0) || (N<=10)
            @printf("%4d %8.0e %12.2e %12.2e %12.2e\n", k, t, qval, rmsd(S̄, Strue), norm(dS))
        end
    end
    
    S̄
end


L(X,E) = K(X,E)


function Ps(v, E)
    Is = [e[1] for e in E]
    Js = [e[2] for e in E]
    Psv = sparse(Is, Js, v/2, n, n)
    Psv + Psv'
end


Ks(D) = 2*(Diagonal(sum(D,dims=2)[:]) - D)


Ls(v, E) = Ks(Ps(v, E))


nothing