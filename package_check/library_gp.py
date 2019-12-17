import GPy
import numpy as np

def main():

    ### Gpy
    xt =np.array([6.0,8.0])[:, np.newaxis]
    yt =np.array([-1.0,1.0])[:, np.newaxis]
    xp = np.array([10.0,12.0])[:, np.newaxis]

    tau=1.0; sigma=1.0
    rbf= GPy.kern.RBF(1,variance=tau, lengthscale=np.sqrt(sigma))
    model=GPy.models.GPRegression(xt,yt, kernel=rbf)
    print("model,predict(xp)\n",model.predict(xp,full_cov=True))
    print("model,predict_noiseless(xp)\n",model.predict_noiseless(xp,full_cov=True))

    #print("model.predict_quantiles(xp)\n",model.predict_quantiles(xp,full_cov=True))
    
    
if __name__ == '__main__':
    main()


