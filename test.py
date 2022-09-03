
import numpy as np
from numpy import linalg
import math
import matplotlib.pyplot as plt
#matplotlib inline

#matrice A
A = np.array([[6,-1],[2,3]])
A

# egval
linalg.eigvals(A)

#Eigenvalues and eigenvectors computation:
# v, p = linalg.eig(A) egval v  egvect p
v, p = linalg.eig(A)
print( "v=", v )
print( "p=", p )
print ("v0=", v[0])
print ("p0=", p[:,0])
print ("v1=", v[1])
print ("p1=", p[:,1])

#Check eigenvectors property ($Av=\lambda v$) is verified for each eigenvalue, eigenvector pair $(\lambda,v)$:
# np.dot produit matriciel 
# pour egval 5 et egvect p corresp
# [: , 0] means [first_row:last_row , column_0]

print( np.dot(A,p[:,0]) )

print( v[0]*p[:,0] )
# pour egval 5 et egvect p corresp
print( np.dot(A,p[:,1]) )
print( v[1]*p[:,1] )

# Diagonalisation computation:
#avec p les egvect de A P; npdiag(v) matrice D comprenant les egval de A; linalg.inv(p) matrice inverse de P
# a@b equivaut à np.dot(a,b); produit matriciel
p @ np.diag(v) @ linalg.inv(p)


#Compute matrix powers using diagonalisation: let's calculate $A^{10}$
p @ np.diag(v**10) @ linalg.inv(p)


#Check result using loop:
result = A.copy()
for _ in range(1,10):
    result = result @ A
print( result )


# Compute square root using diagonalisation: let's find a matrix $B$ such that $B^2=A$
B = p @ np.diag( np.sqrt(v) ) @ linalg.inv(p)
print ("B", B)
C = p @ np.sqrt( np.diag(v) ) @ linalg.inv(p)
print ("C", C)
B @ B

# It gets tricky when size increases:
#%%timeit
#linalg.eig( np.random.random((5000,5000)) )
#Hence a need for approximate computations.
#Define a function returning the $n$-th root of a matrix, using diagonalisation.
def nth_root(A,n):
    v, p = linalg.eig(A)
    return p @ np.diag( v**(1/n) ) @ linalg.inv(p)
nth_root(A,2)
C = nth_root(A,3)
C.dot(C).dot(C)
# C @ C @ C


# Eigenvalue stability: if two real matrices $M$ and $N$ are almost equal, are the eigenvalues of $M$ and $N$ almost equal ?
M = np.array([[1,1000],[0,1]])
N = np.array([[1,1000],[-0.001,1]])
linalg.eigvals(M), linalg.eigvals(N)
##In this example, $M$ has 1 as only eigenvalue and $N$ has no real eigenvalue; $N$ has however two complex eigenvalues: 1+$i$ and 1-$i$.



# Diagonalisation for population dynamics
#In this section, we look into how diagonalization can help computations with an example of population dynamics. We consider the dynamics of a squirrel population. Each year (indexed by $t$), squirrels can be partitioned into three age groups:
#- newborns, with count $P_0(t)$
#- adults, with count $P_1(t)$
#- elderly squirrels, with count $P_2(t)$
#Any squirrel that survived to the end of the year moves to the next age group. Each age is characterized by its reproduction and survival properties:
#- newborns: they are too young to reproduce; 80% of them survive to the next state
#- adults: they produce a an average of 1.5 squirrel each year; 30% of them survive to the next state
#- elderly squirrels: they produce an average of 0.7 squirrel each year. 
#At any given year, the total population can be represented as vector $P(t) = \begin{pmatrix} P_0(t) \\ P_1(t) \\ P_2(t) \end{pmatrix}$, with initial population $P(0) = \begin{pmatrix} 5 \\ 10 \\7 \end{pmatrix}$.
#<b>Exercise:</b><i> What is the long-term behavior of the squirrel population ?</i>
#The population dynamics is characterized by the following system:
#$\begin{pmatrix} P_0(t+1) \\ P_1(t+1) \\ P_2(t+1) \end{pmatrix} = \begin{pmatrix} 0 & 1.5 & 0.7 \\ 0.8 & 0 & 0 \\ 0 & 0.3 & 0 \end{pmatrix} \begin{pmatrix} P_0(t) \\ P_1(t) \\ P_2(t) \end{pmatrix} $, hence $\begin{pmatrix} P_0(t+k) \\ P_1(t+k) \\ P_2(t+k) \end{pmatrix} = \begin{pmatrix} 0 & 1.5 & 0.7 \\ 0.8 & 0 & 0 \\ 0 & 0.3 & 0 \end{pmatrix}^k \begin{pmatrix} P_0(t) \\ P_1(t) \\ P_2(t) \end{pmatrix}$

M = np.array([[0,1.5,0.7],[0.8,0,0],[0,0.3,0]])
M

print (M**10)
P0 = np.array([5,10,7])
print (np.dot(M**10, P0))


#The following function returns the population counts after $k$ years:
def dynamics(M,P0,k):
    v, p = linalg.eig(M)
    Mk = p @ np.diag(v**k) @ linalg.inv(p)
    print( "v=",v )
    print( "p=",p )
    
    print( v**k )
    return np.dot(Mk,P0)
dynamics(M,P0,10)


#The following function returns all population counts until year $k$:
# append ajoute liste en parentheses a liste avant .
def all_dynamics(M,P0,k):
    P = [P0]
    v, p = linalg.eig(M)
    for i in range(1,k+1):
        Mi = p @ np.diag(v**i) @ linalg.inv(p)
        P.append( np.dot(Mi,P0) )
    return np.array(P)
all_dynamics(M,P0,10)


#Graphically, long term population counts yield:
k_max = 50
result = all_dynamics(M,P0,k_max)
plt.plot(np.arange(0,k_max+1), result[:,0], label='Newborns')
plt.plot(np.arange(0,k_max+1), result[:,1], label='Adults')
plt.plot(np.arange(0,k_max+1), result[:,2], label='Elderly')
plt.plot(np.arange(0,k_max+1), np.sum(result,axis=1), label='Total population')
plt.legend()
plt.title('Squirrel population counts over '+str(k_max)+' years')
plt.show()


# Approximate eigenvalue computations</h3>
#As seen above, direct computations of eigenvalues for large matrix sizes may be difficult and time-consuming. In this section, we will look into methods to approximate eigenvalues, in particular the eigenvalue of largest module. Although this restriction may seem severe, eigenvalues of largest module are of great interest in many physical applications, as well as in the analysis of numerical methods.
#<b> Step 1: Power iteration</b>
#The power method is an iterative technique for computing the <i>eigenvalue of largest module</i> of a diagonalizable matrix $A \in \mathcal{M}_n(\mathbb{R})$ and corresponding eigenvector (also called <i>dominant</i> eigenvalue-eigenvector pair), as long as the eigenvalues of $A$ are ordered as
#$$ \lvert \lambda_1 \rvert > \lvert \lambda_2 \rvert \geq \lvert \lambda_3 \rvert ... \geq \lvert \lambda_n \rvert $$
#with $\lvert\lambda_2\rvert,...,\lvert\lambda_n\rvert\neq\lvert\lambda_1\rvert$.

#The algorithm is the following:
#- Start: choose a nonzero initial vector $v_0$
#- Iterate: for $k=1,2,...$ until convergence, compute
#$$v_k = \frac{1}{\alpha_k}Av_{k-1}$$
#where $\alpha_k$ is the component of the vector $Av_{k-1}$ with maximum modulus (also called infinity norm).
#- Return: the eigenvalue corresponding to $v_k$ is $\lambda = \frac{A v_k \cdot v_k}{v_k \cdot v_k}$ (also called Rayleigh quotient).

#<b>Exercise: </b> <i>Implement the power method. How fast does it seem to converge ? Point out some failure cases.</i>
#The following function implements power iteration. What are the parameters $v_0, error, lambd, mu, k, K$ ?

def poweriteration(A,v0,tol):
    error = math.inf # relative error of consecutive lambda values
    u = v0 # store previous eigenvector
    v = u/linalg.norm(u,math.inf) # eigenvector estimate
    mu = 1 # store previous lambda
    k = 0 # count iterations
    K = 1000 # maximum iteration number
    while error > tol and k <= K:
        k = k+1
        u = A.dot(v)
        lambd = u.dot(v)/v.dot(v)
        v = u/linalg.norm(u,math.inf)
        error = abs((lambd-mu)/mu)
        mu = lambd
    if k <= K:
        print(str(k)+" steps performed")
        print("Largest eigenvalue: "+str(lambd))
        print("Eigenvector: "+str(v))
    else:
        print("Power method does not converge in "+str(K)+" steps")
    return lambd, v

#Parameters:
#- $lambd$ stores the largest eigenvalue of $A$ (also denoted lambda)
#- $error$ stores the relative error between two consecutive values of lambda; 
#- $u$ stores the eigenvector corresponding to lambda, computed at the previous step; 
#- $v_0$ is the initialization value of the eigenvector;
#- $mu$ stores the value of lambda computed at the previous step;
#- $k$ stores the count of iterations;
#- $K$ is the maximal iterations number.

#Testing on running 2-by-2 example:

v0 = np.random.rand(2)
tol = 1e-6
poweriteration(A, v0 ,tol)
#Testing on 4-by-4 example:
P = np.random.rand(4,4)
M = linalg.inv(P).dcdot(np.diag((-7,3,4,1))).dot(P)
v0 = np.random.rand(M.shape[0])
tol = 1e-6
poweriteration(M,v0,tol)

#Testing when non-diagonalizable: there is no guarantee for the power method to converge.
M = np.array([[1,-3],[0,1]])
v0 = np.random.rand(M.shape[0])
tol = 1e-6
poweriteration(M,v0,tol)


#Testing when non-unique dominant eigenvalue: convergence of the power method depends on the sign of dominant eigenvalues.
P = np.random.rand(4,4)
M = linalg.inv(P).dot(np.diag((-7,7,4,1))).dot(P)
v0 = np.random.rand(M.shape[0])
tol = 1e-6
poweriteration(M,v0,tol)
P = np.random.rand(4,4)
M = linalg.inv(P).dot(np.diag((-7,-7,4,1))).dot(P)
v0 = np.random.rand(M.shape[0])
tol = 1e-6
poweriteration(M,v0,tol)

#Adding convergence plots:
def poweriteration_plots(A,v0,tol):
    error = math.inf # relative error of consecutive lambda values
    u = v0 # store previous eigenvector
    v = u/linalg.norm(u,math.inf) # eigenvector estimate
    mu = 1 # store previous lambda
    k = 0 # count iterations
    K = 1000 # maximum iteration number
    errors = []
    while error > tol and k <= K:
        k = k+1
        u = A.dot(v)
        lambd = u.dot(v)/v.dot(v)
        v = u/linalg.norm(u,math.inf)
        error = abs((lambd-mu)/mu)
        errors.append(error)
        mu = lambd
    if k <= K:
        print(str(k)+" steps performed")
        print("Largest eigenvalue: "+str(lambd))
        print("Eigenvector: "+str(v))
    else:
        print("Power method does not converge in "+str(K)+" steps")
    return lambd, v, errors
v0 = np.random.rand(2)
tol = 1e-6
lambd, v, errors = poweriteration_plots(A,v0,tol)
plt.plot(np.arange(0,len(errors)), errors, label='Errors')
plt.plot(np.arange(0,len(errors)), tol*np.ones(len(errors)), label='Tolerance')
plt.ylim(0,1e-4)
plt.legend()
plt.title('Convergence towards eigenvalue with largest module')
plt.show()

#How does the number of iterations required vary when the second largest eigenvalue varies?
step_nb = []
for second_eig in np.arange(-6.9,6.9,0.5):
    P = np.random.rand(4,4)
    M = linalg.inv(P).dot(np.diag((-7,second_eig,0,0))).dot(P)
    v0 = np.random.rand(M.shape[0])
    tol = 1e-6
    lambd, v, errors = poweriteration_plots(M,v0,tol)
    step_nb.append( len(errors) )
plt.plot(np.arange(-6.9,6.9,0.5), step_nb)
plt.xlabel("Eigenvalue of second largest module")
plt.ylabel("Iterations")
plt.title('Number of iterations needed to converge')
plt.show()

#<b> References </b>

#Prince, T. and Angulo, N., "Application of Eigenvalues and Eigenvectors and Diagonalization to Environmental Science", Applied Ecology and Environmental Sciences 2(4):106-109, 2014
#Larson, R. and Edwards, B. H. and Falvo, D. C., "Elementary Linear Algebra", Fifth edition, Houghton Mifflin Company, 2004