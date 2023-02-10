module GenPushForward
using Gen
using Zygote


struct OutOfDomain end
const out_of_domain = OutOfDomain()
in_domain(f::Function) = x -> f(x) != out_of_domain
in_domain(f::Function, xs) = filter( x -> f(x) != out_of_domain, xs)


"""
    PushForward
    
The push-forward of a distribution ...
"""
struct PushForward{T,T′} <: Gen.Distribution{T′} 
    dist :: Gen.Distribution{T}
    f       :: Function
    finv    :: Function
    detdf   :: Union{Function, Nothing}
    return_type :: Type{T′} # Return type of `f`, don't think we need that really ...
end

function Gen.random(Q::PushForward, args...) Q.f(random(Q.dist, args...))
end
(Q::PushForward)(args...) = random(Q, args...)
function Gen.logpdf(Q::PushForward{T,T′}, y::T′, args...) where {T, T′}
    x = Q.finv(y)
        
    # We only require `f` to be invertible over its image, i.e.
    # its inverse might not be defined everywhere...
    if x == OutOfDomain() return -Inf; end
        
    log_p = logpdf(Q.dist, x, args...)
        
    # Check if f is a mapping between discrete spaces
    # so there is no `df`, if not add the appropriate 
    # correction term
    if Q.detdf == nothing
            return log_p
    else
            return log_p - log(abs(Q.detdf(x)))
    end
end

function Gen.logpdf_grad(Q::PushForward{T,T′}, v::T′, args...) where {T, T′} 
        Zygote.gradient((v, args...) -> logpdf(Q, v, args...), v, args...)
end

# `Q` has same arguments as its underlying distribution `dist` and
# `f` is assumed to be differentiable, so we can set ...
Gen.has_argument_grads(Q::PushForward) = Gen.has_argument_grads(Q.dist) 
Gen.has_output_grad(   Q::PushForward) = Gen.has_output_grad(Q.dist) 

function invert(d::Dict)
    inv = Dict()
    for (k,v) in d; inv[v] = k; end
    return inv
end

function get_value_type(d::Dict)
    types = map(typeof, values(d))
    return length(types) == 1 ? types[1] : Union{types...}
end
    
function PushForward(dist::Gen.Distribution{T}, f::Dict, return_type::Type{T′}) where {T,T′}
    finv = invert(f)
    return PushForward{T,T′}(dist, x -> f[x], y -> finv[y], nothing, return_type)
end
     
function PushForward(dist::Gen.Distribution{T}, f::Dict) where {T}
    return PushForward(dist, f, get_value_type(f))
end
        
# Not sure how I feel about this but oh well ...
Base.:*(f::Dict, dist::Gen.Distribution{T}) where {T} = PushForward(dist, f)
        
# 
# Reality Checks (for f,g, and det(df))
# 
using LinearAlgebra: norm, det
using StatsBase: mean
function check_inverse(Q::PushForward, args; eps=1e-6 ,metric=(x,y) -> norm(x-y), n=100)
    xs  = [Q.dist(args...) for t=1:n]
    ys  = Q.f.(xs)
    xs′ = Q.finv.(ys)
    err = Q.detdf == nothing ? mean(xs != xs′) : mean(metric.(xs, xs′));
    println("Mean Inverse error ``|g(f(x)) - x|``: ", err)
    if err > eps
        error("`f` and `g` don't seem inverse to each other... high error.")
    else
        println("All good.")
    end
    
end
using ForwardDiff: jacobian, derivative
function check_detdf(Q::PushForward, args; eps=1e-6, n=100)
    if Q.detdf == nothing; 
        println("No jacobian to check... all good.")
        return nothing; 
    end
    
    err = []
    for t=1:n
        x = Q.dist(args...)
        j = typeof(x) == Float64 ? derivative(Q.f,x) : jacobian(Q.f,x);
        if det(j) == 0; error("Something's off with the determinant... det == 0."); end
        push!(err, abs(det(j) - Q.detdf(x)))
    end
    err = mean(err)
    println("Mean det ``|det(df_x)|``: ", err)
    if err > eps
        error("Something's off with the determinant... high error.")
    else
        println("All good.")
    end   
end
        
function check(Q::PushForward, args; eps=1e-6 ,metric=(x,y) -> norm(x-y), n=100)
    check_inverse(Q, args; eps=eps ,metric=metric, n=n)
    check_detdf(Q, args; eps=eps, n=n)
end

    
export PushForward, OutOfDomain, out_of_domain, in_domain, check, check_inverse, check_detdf
end # module