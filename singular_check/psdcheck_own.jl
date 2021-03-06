using LinearAlgebra
using Distributions
using SpecialFunctions
using Printf
using StatsBase
#using GaussianProcesses
using PyCall
using PyPlot

function Mat52(tau,sigma,xi,xj,inlog)
    if inlog
        theta_r = sqrt(5.0)/sigma * abs(log(xi)-log(xj))
    else
        theta_r = sqrt(5.0)/sigma * abs(xi-xj)
    end
    return tau * (1.0 + theta_r + theta_r^2 /3) * exp(-theta_r)
end

function Mat32(tau,sigma,xi,xj,inlog)
    if inlog
        theta_r = sqrt(3.0)/sigma * abs(log(xi)-log(xj))
    else
        theta_r = sqrt(3.0)/sigma * abs(xi-xj)
    end
    return tau * (1.0 + theta_r) * exp(-theta_r)
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
    Ktt_3 = zeros(Float64,lt,lt); Kpt_3 = zeros(Float64,lp,lt); Kpp_3 = zeros(Float64,lp,lp)
    Ktt_M = zeros(Float64,lt,lt); Kpt_M = zeros(Float64,lp,lt); Kpp_M = zeros(Float64,lp,lp)
    Ktt_R = zeros(Float64,lt,lt); Kpt_R = zeros(Float64,lp,lt); Kpp_R = zeros(Float64,lp,lp)
    @inbounds @simd for j=1:lt
        for i=j:lt
            tmp = Mat32(tau,sigma,xt[i],xt[j],inlog)
            Ktt_3[i,j] = tmp; Ktt_3[j,i] = tmp
            tmp = Mat52(tau,sigma,xt[i],xt[j],inlog)
            Ktt_M[i,j] = tmp; Ktt_M[j,i] = tmp
            tmp = RBF(tau,sigma,xt[i],xt[j],inlog)
            Ktt_R[i,j] = tmp; Ktt_R[j,i] = tmp
        end
        for i=1:lp
            Kpt_3[i,j] = Mat32(tau,sigma,xp[i],xt[j],inlog)
            Kpt_M[i,j] = Mat52(tau,sigma,xp[i],xt[j],inlog)
            Kpt_R[i,j] = RBF(tau,sigma,xp[i],xt[j],inlog)
        end
    end
    @inbounds @simd for j=1:lp
        for i=j:lp
            tmp  = Mat32(tau,sigma,xp[i],xp[j],inlog)
            Kpp_3[i,j] = tmp; Kpp_3[j,i] = tmp
            tmp  = Mat52(tau,sigma,xp[i],xp[j],inlog)
            Kpp_M[i,j] = tmp; Kpp_M[j,i] = tmp
            tmp  = RBF(tau,sigma,xp[i],xp[j],inlog)
            Kpp_R[i,j] = tmp; Kpp_R[j,i] = tmp
        end
    end
    return Ktt_3,Kpt_3,Kpp_3,Ktt_M,Kpt_M,Kpp_M,Ktt_R,Kpt_R,Kpp_R
end

function calcSj(cLinv,Ktt,Kpt,Kpp,yt,muy,muyp)
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

function TFplot(Vs,inlog)
    plt=pyimport("matplotlib.pyplot")
    cm=pyimport("matplotlib.cm")
    fig = figure(figsize=(12,3))
    axs = [fig.add_subplot(131),fig.add_subplot(132),fig.add_subplot(133)]
    for i=1:3
        #axs[i].set_xlim(-3,3.1);axs[i].set_ylim(-3,3.1)
        axs[i].set_xlabel("\$ \\log_{10}\\tau \$ ")
        axs[i].set_ylabel("\$ \\log_{10}\\ell \$ ")
    end
    if inlog
        axs[1].set_title("logRBF");axs[2].set_title("logMat52");axs[3].set_title("logMat32")
    else
        axs[1].set_title("RBF");axs[2].set_title("Mat52");axs[3].set_title("Mat32")
    end
    for V in Vs
        tau,sigma,TF_R, TF_M, TF_3=V
        TFs=[TF_R, TF_M, TF_3]
        for i =1:3
            TF= TFs[i]
            if TF == 1.0
                tm = "o"; tc = "green"
            elseif TF==3.0
                tm = "D"; tc = "red"
            else
                tm = "x"; tc = "blue"
            end
            axs[i].scatter(log10(tau),log10(sigma),marker=tm,s=8.,color=tc)#,alpha=0.8)
        end
    end
    if inlog
        savefig("TFplot_log.eps",bbox_inches="tight", pad_inches=0.1)
        savefig("TFplot_log.pdf",bbox_inches="tight", pad_inches=0.1)
    else
        savefig("TFplot.pdf",bbox_inches="tight", pad_inches=0.1)
    end
    plt.close()
end

function difplot(difs,inlog,sigy)
    tls= [ log10(i)+j for j=-3:1:3 for i=1:2:9 ]
    ln=length(tls)
    x = [ [ tls[i] for j=1:ln ] for i=1:ln]
    y = [ [ tls[i] for i=1:ln ] for j=1:ln]
    z = [ [ -Inf  for j=1:ln] for i=1:ln]
    zs = [ deepcopy(z) for i=1:3]
    for nth =1:3
        for tmp in difs[nth] # push!(difs_M3,[tau, sigma, (muj3-M3_mu)*sigy])
            tau,sigma, mudifs = tmp
            mdif = maximum( abs.(mudifs) )
            for (i,raw) in enumerate(x)
                for (j,tcol) in enumerate(raw)
                    if abs(log10(tau)-x[i][j])<1.e-5 && abs(log10(sigma)-y[i][j])<1.e-5
                        zs[nth][i][j]=log10(mdif*sigy)
                    end
                end
            end
        end
    end
    plt=pyimport("matplotlib.pyplot")
    cm=pyimport("matplotlib.cm")
    patches=pyimport("matplotlib.patches")
    fig = figure(figsize=(10,3))
    axs = [fig.add_subplot(131),fig.add_subplot(132),fig.add_subplot(133)]
    for i =1:length(axs)
        axs[i].set_xlabel("\$ \\log_{10} \\tau \$")
        axs[i].set_ylabel("\$ \\log_{10} \\ell \$")
        axs[i].set_xlim(-3,3);axs[i].set_ylim(-3,3)
        if inlog
            axs[i].add_patch(patches.Polygon([[-6, -6], [6, -6], [6, 0.5], [-6, 0.5]],closed=true,
                                             fill=false, color="gray", hatch="....", zorder=1))
        else
            axs[i].add_patch(patches.Polygon([[-6, -6], [6, -6], [6, 1.5], [-6, 1.5]],closed=true,
                                             fill=false, color="gray", hatch="....", zorder=1))
        end
    end
    if inlog
        axs[1].set_title("logRBF");axs[2].set_title("logMat52");axs[3].set_title("logMat32")
    else
        axs[1].set_title("RBF");axs[2].set_title("Mat52");axs[3].set_title("Mat32")
    end

    tmin = -8.0; tmax = 1.0

    ims = [axs[1].pcolormesh(x,y,zs[1],vmin=tmin,vmax=tmax,cmap=plt.cm.plasma,zorder=1000),
           axs[2].pcolormesh(x,y,zs[2],vmin=tmin,vmax=tmax,cmap=plt.cm.plasma,zorder=1000),
           axs[3].pcolormesh(x,y,zs[3],vmin=tmin,vmax=tmax,cmap=plt.cm.plasma,zorder=1000)]

    # ims = [axs[1].pcolormesh(x,y,zs[1],vmin=tmin,cmap=plt.cm.plasma,zorder=1000),
    #        axs[2].pcolormesh(x,y,zs[2],vmin=tmin,cmap=plt.cm.plasma,zorder=1000),
    #        axs[3].pcolormesh(x,y,zs[3],vmin=tmin,cmap=plt.cm.plasma,zorder=1000)]

    fig.colorbar(ims[1], ax=axs[1]);fig.colorbar(ims[2], ax=axs[2]);fig.colorbar(ims[3], ax=axs[3])
    fig.tight_layout()
    if inlog
        savefig("dif_log.eps",bbox_inches="tight")
        savefig("dif_log.pdf",bbox_inches="tight")
    else
        savefig("dif.eps",bbox_inches="tight")
        savefig("dif.pdf",bbox_inches="tight")
    end
    plt.close()
end

function main()
    xt = [6.0,8.0,10.0,12.0,14.0]
    oyt= [-28.602,-30.213,-31.176,-31.713,-31.977]

    meany = mean(oyt)
    muy=0*oyt #.+ meany
    sigy=std(oyt)
    println("sigy $sigy")
    yt =(oyt - muy)/sigy
    xp = collect(16.0:2.0:20.0)

    muyp=0.0*xp .+meany
    lt=length(xt); lp=length(xp)
    tau=1.0

    inlog=false
    inlog=true

    taus   = [ i * 10.0^j for j=-4:1:3 for i=1:2:9]
    sigmas = [ i * 10.0^j for j=-4:1:3 for i=1:2:9]
    push!(taus,1.0 * 10.0^4)
    push!(sigmas,1.0 * 10.0^4)

    epsilon = 1.e-12

    for inlog in [true,false]
        if inlog
            xt_forlib = log.([6.0,8.0,10.0,12.0,14.0])
            xp_forlib = log.(collect(16.0:2.0:20.0))
        else
            xt_forlib = copy(xt)
            xp_forlib = copy(xp)
        end

        Vs=[];difs_R=[];difs_M=[];difs_M3=[]
        for tau in taus
            for sigma in sigmas
                PSD_R=PSD_M=PSD_3=-1.0
                cLinv3=[];cLinv=[];cLinvR=[];muj3=[];mujM=[];mujR=[]
                Sj3=[];SjM=[];SjR=[]
                Ktt_3,Kpt_3,Kpp_3,Ktt_M,Kpt_M,Kpp_M,Ktt_R,Kpt_R,Kpp_R=KernelMat(tau,sigma,xt,xp,inlog)
                try
                    cLinv3,llh3= Mchole(Ktt_3,lt)
                    muj3,Sj3 = calcSj(cLinv3,Ktt_3,Kpt_3,Kpp_3,yt,muy,muyp)
                    PSD_3 = isposdef(Sj3)
                    shiftKtt = Ktt_3 + epsilon * Matrix(1.0I,lt,lt)
                    cLinv,llh = Mchole(shiftKtt,lt)
                    M3_mu,M3_sig = calcSj(cLinv,shiftKtt,Kpt_3,Kpp_3,yt,muy,muyp)
                    dif= (muj3-M3_mu)*sigy
                    push!(difs_M3,[tau, sigma, dif ])
                catch
                    PSD_3 = 3.0
                end

                try
                    cLinvM,llhM= Mchole(Ktt_M,lt)
                    mujM,SjM = calcSj(cLinvM,Ktt_M,Kpt_M,Kpp_M,yt,muy,muyp)
                    PSD_M = isposdef(SjM)
                    shiftKtt = Ktt_M + epsilon * Matrix(1.0I,lt,lt)
                    cLinv,llh = Mchole(shiftKtt,lt)
                    M_mu,M_sig = calcSj(cLinv,shiftKtt,Kpt_M,Kpp_M,yt,muy,muyp)
                    dif = (mujM-M_mu)*sigy
                    push!(difs_M,[tau, sigma, dif])
                catch
                    PSD_M = 3.0
                end
                try
                    cLinvR,llhR= Mchole(Ktt_R,lt)
                    mujR,SjR = calcSj(cLinvR,Ktt_R,Kpt_R,Kpp_R,yt,muy,muyp)
                    shiftKtt = Ktt_R + epsilon * Matrix(1.0I,lt,lt)
                    cLinv,llh = Mchole(shiftKtt,lt)
                    R_mu,R_sig = calcSj(cLinv,shiftKtt,Kpt_R,Kpp_R,yt,muy,muyp)
                    dif = (mujR-R_mu)*sigy
                    push!(difs_R,[tau, sigma, dif])
                    PSD_R = isposdef(SjR)
                catch
                    PSD_R = 3.0
                end

                println("tau=$tau, sigma=$sigma, PSD Mat32:", PSD_3," Mat52:",PSD_M,"  RBF:",PSD_R)
                push!(Vs,[tau, sigma, PSD_R, PSD_M, PSD_3])
            end
        end
        TFplot(Vs,inlog)
        difs=[difs_R,difs_M,difs_M3]
        difplot(difs,inlog,sigy)
    end
end

main()
