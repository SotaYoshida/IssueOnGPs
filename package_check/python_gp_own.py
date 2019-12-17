import numpy as np

def Mat52(tau,sigma,xi,xj,inlog):
    if inlog:
        theta_r = np.sqrt(5.0)/sigma * abs(np.log(xi)-np.log(xj))
    else:
        theta_r = np.sqrt(5.0)/sigma * abs(xi-xj)
    return tau * (1.0 + theta_r + theta_r**2 /3) * np.exp(-theta_r)

def RBF(tau,sigma,xi,xj,inlog):
    if inlog :
        r = np.log(xi)-np.log(xj)
    else:
        r = xi-xj
    return tau * np.exp( - r**2 / (2.0* sigma*sigma) )


def  KernelMat(tau,sigma,xt,xp,inlog):
    lt =len(xt); lp=len(xp)
    #Ktt_M = np.zeros((lt,lt)); Kpt_M = np.zeros((lp,lt)); Kpp_M = np.zeros((lp,lp))
    Ktt_R = np.zeros((lt,lt)); Kpt_R = np.zeros((lp,lt)); Kpp_R = np.zeros((lp,lp))
    for j in range(lt):
        for i in range(lt):
            #tmp = Mat52(tau,sigma,xt[i],xt[j],inlog)
            #Ktt_M[i,j] = tmp; Ktt_M[j,i] = tmp 
            tmp = RBF(tau,sigma,xt[i],xt[j],inlog)
            Ktt_R[i,j] = tmp; Ktt_R[j,i] = tmp 
        for i in range(lp):
            #Kpt_M[i,j] = Mat52(tau,sigma,xp[i],xt[j],inlog)
            Kpt_R[i,j] = RBF(tau,sigma,xp[i],xt[j],inlog)
    for j in range(lp):
        for i in range(lp):
            # tmp  = Mat52(tau,sigma,xp[i],xp[j],inlog)
            # Kpp_M[i,j] = tmp; Kpp_M[j,i] = tmp
            tmp  = RBF(tau,sigma,xp[i],xp[j],inlog)
            Kpp_R[i,j] = tmp; Kpp_R[j,i] = tmp
    #    return Ktt_M,Kpt_M,Kpp_M,Ktt_R,Kpt_R,Kpp_R
    return Ktt_R,Kpt_R,Kpp_R

def calcSj(cLinv,Ktt,Kpt,Kpp,yt):
    tKtp= np.dot(cLinv , Kpt.T)
    return np.dot(Kpt,np.dot(cLinv.T, np.dot(cLinv,yt))), Kpp - np.dot(tKtp.T,tKtp)


def Mchole(tmpA,ln) :
    cLL= np.linalg.cholesky(tmpA)
    logLii=0.0
    for i in range(ln):
        logLii += np.log(cLL[i,i])
    return np.linalg.inv(cLL), 2.0*logLii

def isposdef(A, tol=1e-8):
  E = np.linalg.eigvalsh(A)
  return np.all(E > -tol)

def choleinv(A):
    cLinv,tmpllh = Mchole(A)
    return np.dot(cLinv.T,cLinv)

def main():
    inlog=False
    xt = np.array([6.0,8.0])
    yt = np.array([-1.0,1.0])
    muy,sigy=np.mean(yt),np.std(yt) ## mean zero
    yt = (yt-muy)/sigy
    xp = np.array([10.0,12.0])
    lt = len(xt);lp = len(xp)
    
    tau=1.0; sigma=1.0

    Ktt,Kpt,Kpp=KernelMat(tau,sigma,xt,xp,inlog)
    cLinv,llh= Mchole(Ktt,lt)
    #Kttinv = np.dot(cLinv.T,cLinv)    
    #print("Kttinv", Kttinv,"\n")
    #print("Kttinv*Ktp",np.dot(Kttinv,Kpt.T),"\n")
    #print("Kpt*Kttinv*Ktp",np.dot(Kpt,np.dot(Kttinv,Kpt.T)),"\n")
    muj,Sj = calcSj(cLinv,Ktt,Kpt,Kpp,yt)
 
    print("muj",muj*sigy+muy,"Sj", Sj)


if __name__ == '__main__':
    main()
