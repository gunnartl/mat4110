import numpy as np

def qr_solve(x,b,deg):
    A = polynomial_this(x,deg)
    Q,R=np.linalg.qr(A)
    QTB = Q.T.dot(b)
    for i in range(deg-1,0,-1):    #Backpropagation
        for j in range(i-1,-1,-1):
            coef      = R[j][i]/R[i][i]
            R[j,:]   -= coef*R[i,:]
            QTB[j,:] -= coef*QTB[i,:]
    beta = QTB[:,0]/R.diagonal()
    
    return A.dot(beta)

def cholesky(A): #Cholesky Factorisation
    L = np.zeros(A.shape)
    D = np.zeros(A.shape)

    for i in range(A.shape[0]):
        L[:,i] = A[:,i]/A[i,i]
        D[i,i] = A[i,i]
        A = A-(D[i,i]*np.c_[L[:,i]].dot(np.c_[L[:,i]].T))
    Droot = np.sqrt(D)
    R = L.dot(Droot)
    return R

def cholesky_solve(x,b,deg):
    A = polynomial_this(x,deg)
    R = cholesky(A.T.dot(A))
    Rt = R.T
    beta_temp = np.zeros(deg)
    ATb = A.T.dot(b)

    for i in range(len(beta_temp)): #foreward propagation
        beta_temp[i] = ATb[i]
        for j in range(i):
            beta_temp[i]-=R[i,j]*beta_temp[j]
        beta_temp[i] /= R[i,i]
    beta = np.zeros(deg)
    
    for i in reversed(range(len(beta_temp))): # backwardpropagation
        beta[i] = beta_temp[i]
        for j in (range(i+1,len(beta_temp))):
            beta[i]=beta[i]-Rt[i,j]*beta[j]
        beta[i] =beta[i] / Rt[i,i]
    
    return A.dot(beta)
    
def polynomial_this(x,n):
    X = np.c_[np.ones(len(x))]
    for i in range(1,n):
        X = np.c_[X,x**(i)]
    return X

n = 30
deg = 8
start = -2;
stop = 2;
x = np.linspace(start,stop,n);
eps = 1;
r = np.random.rand(1,n) * eps;
b   = (x*(np.cos(r+0.5*x**3)+np.sin(0.5*x**3))).T; 


approx = qr_solve(x,b,deg)

thommas = cholesky_solve(x,b,deg)

b2  = (4*x**5 - 5*x**4 - 20*x**3 + 10*x**2 + 40*x + 10 + r).T


import matplotlib.pyplot as plt
plt.scatter(x,b)
plt.plot(x,thommas,x,approx)
plt.show()
