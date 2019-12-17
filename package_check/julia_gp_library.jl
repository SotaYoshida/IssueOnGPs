#using Pkg
#Pkg.add("GaussinProcesses")
using GaussianProcesses

function main()
    xt = [6.0,8.0]
    yt= [-1.0,1.0]
    xp = [10.0,12.0]

    sigmas = [1.0]
    for sigma in sigmas
        println("sigma=$sigma")
        mZero = MeanZero() # Zero mean function
        kRBF = SE(0.0,log(sigma)) ##hyperparameters are in logscale
        logObsNoise = -1.e+100
        gp = GP(xt,yt,mZero,kRBF,logObsNoise)
        mu,sigma = predict_y(gp,xp)
        println("sigma[2]", sigma[2])
        println("mu $mu", " sigma ", sigma[1], " ", sigma[2])
    end
end

main()
