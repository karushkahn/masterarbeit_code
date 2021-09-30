"""
This code is written for the numerical part of the master thesis "Optimal
Stopping with Signatures". This is a new approach for optimal stopping problem
solved with the signatures of Rough path. Two numerical methods are implemented
here. This code is written in pure Python, based on numpy, fbm and iisignature.
"""

__author__ = "Zhongdi Xu"
__copyright__ = "Copyright 2021, Optimal Stopping with Signatures"
__credits__ = ["Zhongdi Xu"]
__version__ = "1.0.0"
__maintainer__ = "Zhongdi Xu"
__email__ = "dodoixu@outlook.de"
__status__ = "Development"


# import the modules
import iisignature as iisig
import numpy as np
from fbm import FBM, fbm, times
import time
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import pickle
import scipy.linalg
import json


np.random.seed(234198)

# T: maturity
# n: number of discretization steps
# M: number of samples
# d: dimension of FBM
# h: Hurst parameter

#################################################### Sample generator ###################################################

def gen_fbm(T,n,M,d,h):
    '''
    This function generates a d-dimensional fractional brownian motion and adds the
    time dimension into it.

    Input:
        T = maturity
        n = number of time step
        M = number of data
        d = dimension of FBM
        h = Hurst parameter

    Output:
        1+d dimensional FBM
    '''
    result = np.zeros((n+1,M,d+1))
    f = FBM(n=n, hurst=h, length=T, method='daviesharte')
    for i in range(M):
        path = f.fbm()
        time = f.times()

        # add time to the process
        result[:,i,:] = np.column_stack((time,path))
    return result

class fbm_sampling:
    '''
    a class for FBM sampling
    '''
    def __init__(self, length, n, M, d, h):
        self.length=length
        self.n=n
        self.M=M
        self.d=d
        self.h=h

    def sampling(self):
        s = gen_fbm(self.length,self.n,self.M,self.d,self.h)
        return s

#################################################### utils functions ###################################################

def save_hist(h,myarray,method="default"):
    '''
    This funciton makes the plot for relative distribution of the computed
    stopping times and save it with given method name
    '''
    weights = np.ones_like(myarray) / len(myarray)
    plt.figure(figsize=(15,10))
    plt.hist(myarray, bins=101,weights=weights)
    plt.xticks(times_list[::10])
    plt.title('Verteilung der Stoppzeiten für Fraktionale Brownsche Bewegung mit H={}'.format(str(round(h,2))))
    plt.ylabel('relative Häufigkeit')
    plt.xlabel('Stoppzeit')
    plt.savefig('./'+default+'_method_pics/histogram_{}_20210928.png'.format(round(h,2)))

def compute_sig(fbm,level):
    '''
    This function compute the signature of path X_0,t until time t, where
    X reaches its maximum at time t.
    '''
    n, d = fbm.shape
    time = fbm[:,0]
    path = fbm[:,1]
    sample_dev_norm = path

    # find the time stamp where the process reachs its maximum
    index_max = sample_dev_norm.argmax()

    # compute the signature until t_max
    # 0: numpy array of shape (...,siglength(d,m))
    # 1: the output for a single path is given as a list of numpy arrays, one for each level, for the moment
    # 2: we return not just the signature of the whole path, but the signature of all the partial paths from the start to each point of the path

    max_value = path[index_max]
    signature = iisig.sig(fbm[:index_max+1],level,0)
    return signature

def compute_l(sample,level):
    '''
    This function computes the l for given sample and level of truncated
    signature
    '''
    sig_list = []
    a = time.time()
    try:
        n,M,d = sample.shape
    except:
        M = 1
    for i in range(M):
        sample_dev = sample[:,i,:]
        sig = compute_sig(sample_dev,level)
        sig_list.append(sig)
    #print('start solving linear equation...')
    matrix = list_to_df(sig_list)
    x = solve_lineq(matrix)
    l = x[0]
    residual = x[1]
    b = time.time()
    #print('time used for solving linear equation: ',b-a)
    return l , residual, matrix

def make_plot(x,y,method="default",verbose=False):
    '''
    This funciton makes the plot for "sup E[X] vs. Hurst parameter" and save it
    with given method name
    '''
    plt.figure(figsize=(15,10))
    plt.plot(x,y,label = "predicted expected value")
    #plt.plot(x,signatur_real_expec, label = "real expected value")
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", label = "trendline")
    plt.legend()
    plt.xticks(x)
    plt.grid()
    plt.savefig(method+'_method.png')
    if verbose:
        plt.show()

def list_to_df(X):
    '''
    This function is only needed for computational efficiency
    '''
    feat_cols = [ 'feat'+str(i) for i in range(len(X[0])) ]
    df = pd.DataFrame(X,columns=feat_cols)
    return df

#################################################### Functions for Signature method ###################################################
def get_level(h):
    '''
    This function computes the level of the signature which should be chosen
    for a fbm with Hurst parameter h.
    '''
    a = 1/h
    b = 1+1/h
    p = np.random.uniform(a,b)
    return int(p)

def solve_lineq(matrix):
    '''
    This function solves the lineare equation <l,X>=1 for l with least square
    method from scipy.linalg.lstsq
    Output: x = (x,residuals, rank, singular values of A)
    '''
    M,level = matrix.shape
    print(matrix.shape,'the shape of A')
    A = matrix
    b = pd.Series([1]*M)
    x = scipy.linalg.lstsq(A,b)
    return x

def signature_compute_sup_expec(X_test,level,l):
    '''
    This function compute the sup E[X] with the signature stopping time given l
    and level of truncated signature
    '''
    X_mc = []
    X_real = []
    t_mc = []
    a = time.time()
    n,M,d = X_test.shape
    # for each sample
    for i in range(M):
        # start at time t_i, i = 0,...,n
        for j in range(n):
            path = X_test[:j+1,i,:].reshape((j+1,2))
            # compute the signature until time t_i
            sig = iisig.sig(path,8,0)
            # compute the condition
            cond = l.dot(sig)
            # check the condition
            if cond >= 1:
                break
        # by fulfilling or the 2nd for loop terminates:
        # save the value at this time
        sup_value = X_test[j,i,1]
        # save the real sup value for this path
        real_sup_value = X_test[:,i,1].max()
        X_mc.append(sup_value)
        X_real.append(real_sup_value)
        t_mc.append(j)
    b = time.time()
    #print('time used for testing: ',b-a)
    #print('the predicted supremum expected value is: ',sum(X_mc)/len(X_mc))
    #print('the real supremum expected value is: ',sum(X_real)/len(X_real))
    return X_mc, t_mc , X_real

def signature_whole_run(length,n,M,M_test,d,h):
    '''
    This function runs the whole pipeline to generate the plot for "sup E[X] vs.
    Hurst parameter"; with generating function
    '''
    # do it once, then don't touch it!
    data_train = fbm_sampling(length,n,M,d,h)
    X_train = data_train.sampling()
    # save sample_train
    with open('./data/sample_train_'+'_'+str(h)+'_'+str(M)+'.npy','wb') as f:
        np.save(f,X_train)
    data_test = fbm_sampling(length,n,M_test,d,h)
    X_test = data_test.sampling()
    # save sample_test
    with open('./data/sample_test_'+str(h)+'_'+str(M_test)+'.npy','wb') as f:
        np.save(f,X_test)

    print('sample created!')
    #level = get_level(h)
    #time.sleep(3)
    #print('the level is:',level)
    level = 8 #min(level,8)
    l, residual, matrix= compute_l(X_train, level)

    # save l
    with open('./data/computed_l_'+str(level)+'_'+str(h)+'_'+str(M)+'.npy','wb') as f:
        np.save(f,l)

    # save residual
    with open('./data/computed_res_'+str(level)+'_'+str(h)+'_'+str(M)+'.npy','wb') as f:
        np.save(f,residual)

    X_mc, t_mc, X_real = signature_compute_sup_expec(X_test,level,l)

    return sum(X_mc)/len(X_mc), sum(X_real)/len(X_real), residual, t_mc

def whole_run_read(length,n,M,M_test,d,h):
    '''
    This function runs the whole pipeline to generate the plot for "sup E[X] vs.
    Hurst parameter"; without generating function
    '''
    #X_train = np.load('./data/sample_train_'+'_'+str(h)+'_'+str(M)+'.npy')
    X_test = np.load('./data/sample_test_'+str(h)+'_'+str(M_test)+'.npy')

    #print('sample loaded!')
    level = 8
    #l, residual, matrix= compute_l(X_train, level)

    # load l
    l = np.load('./data/computed_l_'+str(level)+'_'+str(h)+'_'+str(M)+'.npy')

    # load residual
    residual = np.load('./data/computed_res_'+str(level)+'_'+str(h)+'_'+str(M)+'.npy')

    print("l loaded for h={}!".format(h))

    X_mc, t_mc, X_real = signature_compute_sup_expec(X_test,level,l)

    print("predicted expectation computed for h={}!".format(h))

    return sum(X_mc)/len(X_mc), sum(X_real)/len(X_real), residual, t_mc

#################################################### Functions for Hybrid method ###################################################

def sigmoid(x):
    return 1/(1+math.exp(-x))

def F(sig,l):
    res = l.dot(sig)
    return sigmoid(res-1)


def g(t,X):
    index = times_list.index(t)
    # g darf nur reellwertig sein
    result = X[index,1]
    return result


def L(X,sig,tn,tau,l):
    '''
    Monte Carlo for the expectation

    X: sample matrix of dimension n, M, d
    sig: signature matrix of the sample of dimension M, (d)*((d)**level-1)/(d-1)
    tn: the n-th time stamp
    tau: numpy array contains the tau_n+1 stopping times
    l: the object to find

    g(tn,X)-g(tau,X) is a matrix of samples of size nxM
    F(sig,l)(1-F(sig,l)) is a scalar
    sig is a matrix of size Mxd*(d**level-1)/(d-1)
    '''
    n,M,d = X.shape
    level = 2
    V_n = np.zeros(M)
    d_V_n = np.zeros((M,int((d)*((d)**level-1)/(d-1))))
    for i in range(M):
        V_n[i] = g(tn[i],X[:,i,:])*F(sig.iloc[i],l) + g(tau[i],X[:,i,:])*(1-F(sig.iloc[i],l))
        d_V_n[i,:] = ((g(tn[i],X[:,i,:])-g(tau[i],X[:,i,:]))*F(sig.iloc[i],l)*(1-F(sig.iloc[i],l)))*(sig.iloc[i])
    return V_n.mean(), d_V_n.mean(axis=0)

def hybrid_calculate_l(X,sig,tn,tau):
    '''
    Gradient Descent
    '''
    n, M, d = X.shape
    level = 2

    # initial value, random set
    l = np.random.rand(int((d)*((d)**level-1)/(d-1)))
    V_old = 0

    rate = 1 #get_learning_rate(0) # learning rate
    epochs = 30 # the number of iterations to perform gradient descent

    # performing gradient descent
    for i in range(epochs):
        V_new, d_V = L(X,sig,tn,tau,l)
        l = l + d_V*rate
        diff = abs(V_new - V_old)
        if diff<=0.0001:
            break
        V_old = V_new
        # report progress
        #print('>%d with V = %.5f' % (i, V_old))
    return l

def hybrid_calculate_l_adam(X,sig,tn,tau):
    '''
    Adam
    '''
    n, M, d = X.shape
    level = 2
    # define the total iterations
    n_iter = 20
    # steps size
    alpha = 1
    # factor for average gradient
    beta1 = 0.8
    # factor for average squared gradient
    beta2 = 0.999
    # epsilon
    eps=1e-8

    # initial value
    l = np.random.rand(int((d)*((d)**level-1)/(d-1)))
    V_old, derivate = L(X,sig,tn,tau,l)

    m = [0.0 for _ in range(len(l))]
    v = [0.0 for _ in range(len(l))]
    # run the gradient descent updates
    for t in range(n_iter):
        # calculate the functional and derivate values by using Monte Carlo
        V_new, derivate = L(X,sig,tn,tau,l)
        V_new = V_new*(-1)
        derivate = derivate*(-1)
        # build a solution one variable at a time
        for i in range(len(l)):
            # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
            m[i] = beta1 * m[i] + (1.0 - beta1) * derivate[i]
            # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
            v[i] = beta2 * v[i] + (1.0 - beta2) * derivate[i]**2
            # mhat(t) = m(t) / (1 - beta1(t))
            mhat = m[i] / (1.0 - beta1**(t+1))
            # vhat(t) = v(t) / (1 - beta2(t))
            vhat = v[i] / (1.0 - beta2**(t+1))
            # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
            l[i] = l[i] - alpha * mhat / (sqrt(vhat) + eps)
        # evaluate candidate point
        if abs(V_old - V_new) <=0.01:
            break
        V_old = V_new
        # report progress
        #print('>%d with V = %.5f' % (i, V_old))
    return l

def hybrid_compute_sig_multi(X):
    '''
    This function computes the signatures of given samples and returns the data
    in the DataFrame type for computational efficiency.
    '''
    signatures = []
    level = 8
    n,M,d = X.shape
    for i in range(M):
        path = X[:,i,:]
        sig = iisig.sig(path,level,0)
        signatures.append(sig)
    signatures = list_to_df(signatures)
    return signatures

def hybrid_whole_run(length,n,M,M_test,d,h):
    '''
    This function runs the whole pipeline to generate the plot for "sup E[X] vs.
    Hurst parameter"; with generating function
    '''
    # do it once, then don't touch it!
    data_train = fbm_sampling(length,n,M,d,h)
    X_train = data_train.sampling()
    # save sample_train
    #with open('./data/sample_train_'+'_'+str(h)+'_'+str(M)+'.npy','wb') as f:
    #    np.save(f,X_train)
    data_test = fbm_sampling(length,n,M_test,d,h)
    X_test = data_test.sampling()
    # save sample_test
    #with open('./data/sample_test_'+str(h)+'_'+str(M_test)+'.npy','wb') as f:
    #    np.save(f,X_test)

    print('sample created!')
    #level = get_level(h)
    #time.sleep(3)
    #print('the level is:',level)
    level = 8 #min(level,8)
    l, tau_mat, f_mat= hybrid_method_train(X_train)

    # save l
    #with open('./data/computed_l_'+str(level)+'_'+str(h)+'_'+str(M)+'.npy','wb') as f:
    #    np.save(f,l)


    tau_mat_test,f_mat_test,V_mat_test = hybridmethod_test(X_test,l,tau_mat,f_mat)
    V_est = np.mean(V_mat_test,axis=1)[0]

    return V_est, tau_mat_test

def hybrid_whole_run_read(length,n,M,M_test,d,h):
    '''
    This function runs the whole pipeline to generate the plot for "sup E[X] vs.
    Hurst parameter"; without generating function
    '''
    # load sample_train
    X_train = np.load('./data/sample_train_'+'_'+str(h)+'_'+str(M)+'.npy')

    # save sample_test
    X_test = np.load('./data/sample_test_'+str(h)+'_'+str(M_test)+'.npy')

    print('sample LOADED!')
    #level = get_level(h)
    #time.sleep(3)
    #print('the level is:',level)
    level = 8 #min(level,8)
    l, tau_mat, f_mat= hybridmethod_train(X_train)

    # save l
    #with open('./data/computed_l_'+str(level)+'_'+str(h)+'_'+str(M)+'.npy','wb') as f:
    #    np.save(f,l)


    tau_mat_test,f_mat_test,V_mat_test = hybridmethod_test(X_test,l,tau_mat,f_mat)
    V_est = np.mean(V_mat_test,axis=1)[0]

    return V_est, tau_mat_test

def hybrid_method_train(X_train):
    '''
    This method computes the candidate for the optimal stopping time in the
    hybrid method
    '''
    n,M,d = X_train.shape
    level=8
    # initialization
    length = 1
    # stopping rules
    l = [None]*n
    l[-1] = np.array([1]*int((d)*((d)**level-1)/(d-1)))

    # stopping times
    tau_mat = np.zeros((n,M))
    tau_mat[n-1,:]= length

    # stopping decisions
    f_mat = np.zeros((n,M))
    f_mat[n-1,:]=1

    # time stamps
    times_list = [i/(n-1) for i in range(n)]

    # the pipeline
    a = time.time()
    for t in range(n-2,-1,-1):
        #print('start at time',t+1)
        paths = X_train[:t+1,:,:]   # cut the path until time stamp i (included)
        signatures = hybrid_compute_sig_multi(paths)   # compute the signatures for all samples
        #print('signature computed!')
        tau_n_plus_1 = tau_mat[t+1]   # get the optimal stopping time until i (included)
        tn = [times_list[t]]*len(tau_n_plus_1)
        # run Monte Carlo + gradient descent to compute l_n and store l_n in l
        l[t]= hybrid_calculate_l_adam(X_train,signatures,tn,tau_n_plus_1)
        l_n = l[t]
        # set f_n_ln for each signature
        f_mat[t,:] = (l_n.dot(signatures.T) >= 1)*1.0
        # store tau_n
        tau_mat[t,:] = np.array([times_list[j] for j in np.argmax(f_mat,axis=0)])
    b = time.time()
    print('time used for training: ',b-a)
    return l, tau_mat, f_mat

def hybrid_method_test(X_test,l,tau_mat,f_mat):
    '''
    This function uses the computed l of the optimal stopping time and test the
    quality by computing E[X]
    '''
    l_loaded = l
    # question: how to do the prediction step
    level = 8
    Expec = []
    n,M,d = X_test.shape
    # stopping times
    tau_mat_test = np.zeros((n,M))
    tau_mat_test[n-1,:]= length

    # stopping decisions
    f_mat_test = np.zeros((n,M))
    f_mat_test[n-1,:]=1

    # for each time stamp, compute the mean value function, stored n V_est_test

    V_mat_test=np.zeros((n,M))
    V_est_test=np.zeros(n)

    for m in range(M):
        V_mat_test[n-1,m]=X_test[:,m,1].max()

    V_est_test[n-1]=np.mean(V_mat_test[n-1,:])

    for i in range(n-2,-1,-1):
        paths = X_test[:i+1,:,:]   # cut the path until time stamp i (included)
        signatures = hybrid_compute_sig_multi(paths)   # compute the signatures for all samples
        l_n = l_loaded[i]
        # set f_n_ln for each signature
        f_mat_test[i,:] = (l_n.dot(signatures.T) >= 1)*1.0
        # store tau_n
        tau_mat_test[i,:] = np.array([times_list[j] for j in np.argmax(f_mat_test,axis=0)])
        for m in range(M):
            V_mat_test[i,m] = X_test[times_list.index(tau_mat_test[i,m]),m,1]
    print('Done')
    return tau_mat_test,f_mat_test,V_mat_test

#################################################### Parameters ###################################################
T = 1
n = 100
M = 3000
M_test = 500
d = 1
times_list = [i/(n) for i in range(n+1)]
hurst_list = np.arange(0.05,1,0.05)

#################################################### Signature method ###################################################
signature_pred_expec = []
signature_real_expec = []
signature_residuals = []
a = time.time()
for h in hurst_list:
    h = round(h,2)
    print('start with:',h)
    V_pred,V_real, residual, t_mc = signature_whole_run(T,n,M,M_test,d,h)
    #pred,real, residual,t_mc = signature_whole_run_read(T,n,M,M_test,d,h)
    signature_pred_expec.append(V_pred)
    signature_real_expec.append(V_real)
    signature_residuals.append(residual)
    #save_hist(h,t_mc,"signature")
b=time.time()
print('time used: ',b-a)

# make and save the "sup E[X_t] vs. Hurst" plot
x = hurst_list
y = signature_pred_expec
make_plot(x,y,method="signature")

# make and save the "sup E[X_t] vs. Hurst" plot
#plt.figure(figsize=(15,10))
#plt.plot(x,signatur_residuals,label = 'Residual vs. Hurst parameter')
#plt.legend()
#plt.xticks(x)
#plt.grid()
#plt.show()

#################################################### Hybrid method ###################################################
hybrid_pred_expec = []
a = time.time()
for h in hurst_list:
    print('start with:',h)
    V_pred,stopping_times = whole_run(T,n,M,M_test,d,round(h,2))
    #pred,stopping_times = whole_run_read(T,n,M,M_test,d,round(h,2))
    hybrid_pred_expec.append(V_pred)
    #save_hist(h,stopping_times[0],"hybrid")
b=time.time()
print('time used: ',b-a)

# make and save the "sup E[X_t] vs. Hurst" plot
y = hybrid_pred_expec
make_plot(x,y,method="hybrid")


# save the computed data in a JSON file
with open("hybridmethod_pred_expec.json", 'w') as f:
    # indent=2 is not needed but makes the file human-readable
    json.dump(pred_expec, f, indent=2)
