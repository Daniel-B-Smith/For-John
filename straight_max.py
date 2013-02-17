#!/usr/bin/env python

from __future__ import division
import numpy as np
import model
from copy import copy, deepcopy
from scipy.optimize import fmin_tnc, fmin_l_bfgs_b, anneal, brute
import John

from time import clock

# Reflection function to implement reflecting boundary conditions:
def reflect(x):
    if x>=1:
        return reflect(2-x)
    elif x<0:
        return reflect(-x)
    else:
        return x

def delt(i,j):
    if i==j:
        return 1
    else:
        return 0

def classify(x, c, b):
    """
    Test classification function. Takes test result x, cut-off c and bandwidth
    b.
    """
    if x<c-b:
        return 0
    elif x>c+b:
        return 1
    else:
        if b>10**-7:
            return (x-c+b)/2/b
        else:
            return 0.5
        
def to_max_log_like(x,c1,c2,b1,b2,test):
    P1,P2,Se1,Se2,Sp1,Sp2 = x
    test = np.asarray(test)
    tmp, D = John.log_like(P1,P2,Se1,Se2,Sp1,Sp2,c1,c2,b1,b2,test)
    return (-tmp, -D)

def to_Se_Sp(x,data,guess=None):
    c1,c2,b1,b2 = x
    return -Se_Sp_funct(c1,c2,b1,b2,data,guess)

def to_Se_Sp_zero(x,data,guess=None):
    c1,c2 = x
    return -Se_Sp_funct(c1,c2,0,0,data,guess)

def band_like(c1,c2,b1,b2,data,guess=None):
    """
    Function of cutoff and bandwidths which returns likelihood.
    """
    if guess==None:
        guess = [0.5,0.5,0.75,0.75,0.75,0.75]

def Se_Sp_funct(c1,c2,b1,b2,data,guess=None):
    """
    Function of cutoff and bandwidth which returns product of test sensitivies
    and specificities to be maximized.  Returns Se1*Se2*Sp1*Sp2
    """
    if guess==None:
        guess = [0.5,0.5,0.75,0.75,0.75,0.75]

    EPS = 10**-12
    bound = [(EPS,1-EPS)]*6
    (X, j, k)  = fmin_tnc(to_max_log_like,guess, args=(c1,c2,b1,b2,data,), 
                          bounds=bound,disp=0)
    # Return Se1/(1-Sp1)+Se2/(1-Sp2)
    # return (X[2]/(1-X[4])+X[3]/(1-X[5]))
    return np.sum((1-X[2:])**2)
    # return np.sum(((1-x)**2 for x in X[2:]))
    # return (X[2]-X[4])**2+(X[3]-X[5])**2-np.prod(X[2:])

def for_Se_Sp_band(x,c1,c2,data,guess=None):
    b1, b2 = x
    return Se_Sp_funct(c1,c2,b1,b2,data,guess=None)

def Se_Sp_band(c1, c2, data, guess=None, bound=None):
    """
    Function of test cutoffs which optimizes bandwidths and returns product
    of test sensitivies and specificies to be maximized. 
    Returns Se1*Se2*Sp1*Sp2
    """
    if guess==None:
        guess = [0.25,0.25]
    if bound==None:
        bound = [(0, 1), (0, 1)]

    (X, f, d) = fmin_l_bfgs_b(for_Se_Sp_band, guess, args=(c1, c2, data),
                              approx_grad=True, bounds=bound, pgtol=1e-08)
    return f, X

def to_Se_Sp_band(x,data,guess=None):
    c1, c2 = x
    return Se_Sp_band(c1,c2,data,guess=None)

def like_term():
    return  {
        0 : lambda P,Se1,Se2,Sp1,Sp2 : P*Se1*Se2+(1-P)*(1-Sp1)*(1-Sp2),
        1 : lambda P,Se1,Se2,Sp1,Sp2 : P*(1-Se1)*Se2+(1-P)*Sp1*(1-Sp2),
        2 : lambda P,Se1,Se2,Sp1,Sp2 : P*Se1*(1-Se2)+(1-P)*(1-Sp1)*Sp2,
        3 : lambda P,Se1,Se2,Sp1,Sp2 : P*(1-Se1)*(1-Se2)+(1-P)*Sp1*Sp2
        }

def log_like(P1,P2,Se1,Se2,Sp1,Sp2,c1,c2,b1,b2,test):
    """
    Log likelihood function for the two-test fuzzy Hui-Walter.
    Parameters:
    P1, P2:   two prevalences
    Se1, Se2: two sensitivities
    Sp1, Sp2: two specificities
    c1, c1:   two test cut-offs
    b1, b2:   two test bandwidths
    test[i,j,iN]:    iN'th sample of test i performed on population j
    """
    N = max(test.shape)
    tmp = 0
    sums = np.zeros((4,2))
    P = [P1,P2]
    c = [c1,c2]
    b = [b1,b2]

    cla = np.zeros(test.shape)
    for i in xrange(2):
        for j in xrange(2):
                cla[i,j,:] = JClass.classify(c[i],b[i],test[i,j,:],)

    for i in xrange(2):
        sums[0,i] += sum((cla[0,i,iN]*cla[1,i,iN] for iN in xrange(N)))
        sums[1,i] += sum(((1-cla[0,i,iN])*cla[1,i,iN] for iN in xrange(N)))
        sums[2,i] += sum((cla[0,i,iN]*(1-cla[1,i,iN]) for iN in xrange(N)))
        sums[3,i] += sum(((1-cla[0,i,iN])*(1-cla[1,i,iN])  for iN in xrange(N)))

    term_dict = like_term()

    for i in xrange(2):
        for j in xrange(4):
            tmp += np.log(term_dict[j](P[i],Se1,Se2,Sp1,Sp2))*sums[j,i]
    return tmp, sums

def grad_log_like(P1,P2,Se1,Se2,Sp1,Sp2,c1,c2,b1,b2,test,sums):
    """
    Calculates the gradient of the log-likelihood function.
    See log_like for details.
    """
    N = max(test.shape)
    tmp = 0
    P = [P1,P2]
    Se = [Se1,Se2]
    Sp = [Sp1,Sp2]
    c = [c1,c2]
    b = [b1,b2]

    term_dict = like_term()
    ret = np.zeros(6)
    for i in xrange(2):
        ret[i] += sums[0,i]*(Se1*Se2-(1-Sp1)*(1-Sp2))\
            /term_dict[0](P[i],Se1,Se2,Sp1,Sp2)
        ret[i] += sums[1,i]*((1-Se1)*Se2-Sp1*(1-Sp2))\
            /term_dict[1](P[i],Se1,Se2,Sp1,Sp2)
        ret[i] += sums[2,i]*(Se1*(1-Se2)-(1-Sp1)*Sp2)\
            /term_dict[2](P[i],Se1,Se2,Sp1,Sp2)
        ret[i] += sums[3,i]*((1-Se1)*(1-Se2)-Sp1*Sp2)\
            /term_dict[3](P[i],Se1,Se2,Sp1,Sp2)

    coeff_dict = {
        (0,0) : lambda p : p*Se2,         (0,1) : lambda p : p*Se1, 
        (1,0) : lambda p : -p*Se2,        (1,1) : lambda p : p*(1-Se1), 
        (2,0) : lambda p : p*(1-Se2),     (2,1) : lambda p : -p*Se1,
        (3,0) : lambda p : -p*(1-Se2),    (3,1) : lambda p : -p*(1-Se1),

        (0,2) : lambda p : (p-1)*(1-Sp2), (0,3) : lambda p : (p-1)*(1-Sp1),
        (1,2) : lambda p : (1-p)*(1-Sp2), (1,3) : lambda p : (p-1)*Sp1,
        (2,2) : lambda p : (p-1)*Sp2,     (2,3) : lambda p : (1-p)*(1-Sp1),
        (3,2) : lambda p : (1-p)*Sp2,     (3,3) : lambda p : (1-p)*Sp1
        }

    for j in xrange(4):                 # For Se1, Se2, Sp1, Sp2
        for i in xrange(2):             # Population
            for k in xrange(4):         # Sum number
                ret[j+2] += sums[k,i]*coeff_dict[(k,j)](P[i])\
                    /term_dict[k](P[i],Se1,Se2,Sp1,Sp2)
    return ret

def generate_sim_data(sigma,mu,P,N):
    """
    Generates two test, two population samples based on the following 
    parameters:
    sigma (2,2): sigma[i,j] is the std dev of test i with outcome j
    mu    (2,2): mu[i,j] is the mean of test i with outcome j
    P     (2):   P[i] is the prevalence for population i
    N:           The number of samples

    Returns test a (2,2,N) array with:
    test[i,j,iN] = iN'th sample of test i performed on population j 
    """
    m = [int(p*N) for p in P]
    tmp = np.zeros((2,2,N))
    # test 1 population 1
    tmp[0,0,:] = np.hstack((sigma[0][0]*np.random.randn(N-m[0])+mu[0][0],
                            sigma[0][1]*np.random.randn(m[0])+mu[0][1]))
    # test 2 population 1
    tmp[1,0,:] = np.hstack((sigma[1][0]*np.random.randn(N-m[0])+mu[1][0],
                            sigma[1][1]*np.random.randn(m[0])+mu[1][1]))
    # test 1 population 2
    tmp[0,1,:] = np.hstack((sigma[0][0]*np.random.randn(N-m[1])+mu[0][0],
                            sigma[0][1]*np.random.randn(m[1])+mu[0][1]))
    # test 2 population 1
    tmp[1,1,:] = np.hstack((sigma[1][0]*np.random.randn(N-m[1])+mu[1][0],
                            sigma[1][1]*np.random.randn(m[1])+mu[1][1]))

    return tmp

def main(init=None, sigma=None, mu=None, P=None, zero_b=None, N=None,
         bound=None, **kwargs):
    """
    Runs a brute-force optimization using simulated two test, two population
    data generated based on the following parameters:
    sigma (2,2): sigma[i,j] is the std dev of test i with outcome j
    mu    (2,2): mu[i,j] is the mean of test i with outcome j
    P     (2):   P[i] is the prevalence for population i
    N:           The number of samples
    """

    start = clock()

    if sigma==None:
        sigma = np.ones((2,2))
    if mu==None:
        mu = [[-1,1],[-1,1]]
    if P==None:
        P = [0.2,0.8]
    if not zero_b:
        funct = to_Se_Sp_zero
    else:
        funct = to_Se_Sp_band
    if init==None:
        if zero_b:
            init = [0,0]
        else:
            init = [0,0,0.01,0.01]
    if N==None:
        N=1000
    if bound==None:
        bound = [(-1,1)]*2

    data = generate_sim_data(sigma,mu,P,N)
    newdata = ['']*4
    newdata[0] = data[0][0]
    newdata[1] = data[0][1]
    newdata[2] = data[1][0]
    newdata[3] = data[1][1]
#    X, f, g, J = brute(funct, bound, full_output=True, args=(data,), **kwargs)
    X = [0,0]
    print "Optimal centers found: "
    print 'Test 1: '+str(X[0])
    print 'Test 2: '+str(X[1])
    restart = clock()
    print 'First part took: '+str(restart-start)+' sec'

    if zero_b:
        (Y, f, d) = fmin_l_bfgs_b(for_Se_Sp_band, [0.5,0.5], 
                                  bounds=[(0,None)]*2, 
                                  args=(X[0], X[1], data), 
                                  approx_grad=True)
        print 'Bandwidths for optimal centers:'
        print 'Test 1: '+str(Y[0])
        print 'Test 2: '+str(Y[1])

    print 'Second part took '+str((clock()-restart))+' sec'
    return X 
# , f, g, J
    
if __name__ == "__main__":
    sys.exit(main(sys.argv))

