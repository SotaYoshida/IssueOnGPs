using LinearAlgebra
using Distributions
using SpecialFunctions
using Printf
using StatsBase

function Mat52(tau,sigma,xi,xj,inlog)
    if inlog
        theta_r = sqrt(5.0)/sigma * abs(log(xi)-log(xj))
    else
        theta_r = sqrt(5.0)/sigma * abs(xi-xj)
    end
    return tau * (1.0 + theta_r + theta_r^2 /3) * exp(-theta_r)
end

function RBF(tau,sigma,xi,xj,inlog)
    if inlog 
        r = log(xi)-log(xj)
    else
        r = xi-xj
    end
    return tau * exp( - r^2 / (2.0*sigma*sigma) )
end


function KernelMat(tau::Float64,sigma::Float64,xt::T,xp::T,inlog) where {T<:Array{Float64,1}}
    lt =length(xt); lp=length(xp)
    #Ktt_M = zeros(Float64,lt,lt); Kpt_M = zeros(Float64,lp,lt); Kpp_M = zeros(Float64,lp,lp)
    Ktt_R = zeros(Float64,lt,lt); Kpt_R = zeros(Float64,lp,lt); Kpp_R = zeros(Float64,lp,lp)
    @inbounds @simd for j=1:lt
        for i=j:lt
            # tmp = Mat52(tau,sigma,xt[i],xt[j],inlog)
            # Ktt_M[i,j] = tmp; Ktt_M[j,i] = tmp 
            tmp = RBF(tau,sigma,xt[i],xt[j],inlog)
            Ktt_R[i,j] = tmp; Ktt_R[j,i] = tmp 
        end
        for i=1:lp
            # Kpt_M[i,j] = Mat52(tau,sigma,xp[i],xt[j],inlog)
            Kpt_R[i,j] = RBF(tau,sigma,xp[i],xt[j],inlog)
        end
    end
    @inbounds @simd for j=1:lp
        for i=j:lp
            # tmp  = Mat52(tau,sigma,xp[i],xp[j],inlog)
            # Kpp_M[i,j] = tmp; Kpp_M[j,i] = tmp
            tmp  = RBF(tau,sigma,xp[i],xp[j],inlog)
            Kpp_R[i,j] = tmp; Kpp_R[j,i] = tmp
        end
    end
    #return Ktt_M,Kpt_M,Kpp_M,Ktt_R,Kpt_R,Kpp_R
    return Ktt_R,Kpt_R,Kpp_R
end

function calcSj(cLinv,Ktt,Kpt,Kpp,yt)
    tKtp=cLinv * transpose(Kpt)
    return Kpt*(transpose(cLinv)*(cLinv*yt)), Kpp - transpose(tKtp) * tKtp

end

function Mchole(tmpA,ln::Int64) 
    cLL=cholesky(tmpA).L
    logLii=0.0
    @simd for i = 1:ln
        logLii += log(cLL[i,i])
    end
    return inv(cLL), 2.0*logLii
end

function main()
    inlog=false
    xt = [6.0,8.0]
    yt = [-1.0,1.0]
    lt = length(xt)
    muy=mean(yt);sigy=std(yt)
    xp = [10.0,12.0]
    tau=1.0; sigma = 1.0

    Ktt,Kpt,Kpp=KernelMat(tau,sigma,xt,xp,inlog)    
    cLinv,llh= Mchole(Ktt,lt)
    muj,Sj = calcSj(cLinv,Ktt,Kpt,Kpp,yt)
    println("muj ", 0.0*xp .+ muy + muj," Sj ",Sj[1,1], " ", Sj[2,2])
end

main()
