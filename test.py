import numpy as np

def qr_solve(x,b,deg):
    A = polynomial_this(x,deg)
    Q,R=np.linalg.qr(A)
    QTB = Q.T.dot(b)
    for i in range(deg-1,0,-1):
        for j in range(i-1,-1,-1):

            coef      = R[j][i]/R[i][i]
            R[j,:]   -= coef*R[i,:]
            QTB[j,:] -= coef*QTB[i,:]
    beta = QTB[:,0]/R.diagonal()

    approx = np.zeros_like(x)
    for i in range(deg):
        approx += beta[i]*x**i
    return approx

def cholesky(A):
    L = np.zeros(A.shape)
    D = np.zeros(A.shape)
    for i in range(A.shape[0]):
        L[:,i] = A[:,i]/A[i,i]
        D[i,i] = A[i,i]
        #print(L)
        #print(D)
        
        A = A-D[i,i]*np.c_[L[:,i]].dot(np.c_[L[:,i]].T)
        
    return L,D
    
            
def polynomial_this(x,n):
    X = np.c_[np.ones(len(x))]
    for i in range(1,n):
        X = np.c_[X,x**(i)]
    return X

n = 30
start = -2;
stop = 2;
x = np.linspace(start,stop,n);
eps = 1;
r = np.random.rand(1,n) * eps;
b   = (x*(np.cos(r+0.5*x**3)+np.sin(0.5*x**3))).T; 


approx = qr_solve(x,b,3)
A = np.array([[3,4],[4,6]])
L,D = cholesky(A)
print(L)
print(D)

#eps = 1
#r   = np.random.rand(1,n) * eps
b2  = (4*x**5 - 5*x**4 - 20*x**3 + 10*x**2 + 40*x + 10 + r).T

approx2 = qr_solve(x,b2,3)

import matplotlib.pyplot as plt
plt.scatter(x,b)
plt.plot(x,approx,'orangered')
plt.show()
