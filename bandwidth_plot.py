#!/usr/bin/env python

from __future__ import division
import straight_max
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from argparse import RawTextHelpFormatter
from scipy.optimize import brute
from time import clock

def go(x,y,z,data):
    return straight_max.for_Se_Sp_band(x,y,z,data)

def main(Ns=None, bound=None, sigma=None, mu=None, P=None, N=None, **kwargs):
    if Ns==None:
        Ns = 11
    if sigma==None:
        sigma = np.ones((2,2))
    if mu==None:
        mu = [[-1,1],[-1,1]]
    if P==None:
        P = [0.2,0.8]
    if N==None:
        N=1000
    if bound==None:
        bound = [(0,1)]*2

    start = clock()

    data = straight_max.generate_sim_data(sigma,mu,P,N)
    newdata = ['']*4
    newdata[0] = data[0][0]
    newdata[1] = data[1][0]
    newdata[2] = data[0][1]
    newdata[3] = data[1][1]

    x1 = np.linspace(0, bound[0][1], num=Ns)
    x2 = np.linspace(0, bound[1][1], num=Ns)
    
    X, f, g, J = brute(go, bound, full_output=True, finish=None,
                       args=(0, 0, newdata), Ns=Ns, **kwargs)

    f = np.array(J).min()

    f2, Y = straight_max.Se_Sp_band(0, 0, newdata, bound=bound)
    print "Brute force min = " + str(f)
    print X
    print "Minimizer min = " + str(f2)
    print Y
    np.savetxt("bandwidth.dat",J)

    print 'Simulation took: '+str(clock()-start)+' sec'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("$\sum(1-S_j)^2$",size=36)

    cax = ax.matshow(J)

    M = len(ax.get_xticklabels())
    xticks = ['',str(0)]+['']*(M-4)+[str(bound[0][1]),'']
    yticks = ['',str(0)]+['']*(M-4)+[str(bound[1][1]),'']
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(24)

    ax.set_xlabel('Test 1',size=32)
    ax.set_ylabel('Test 2',size=32)
    
    col = fig.colorbar(cax)
    for label in col.ax.get_yticklabels():
        label.set_fontsize(18)
        
    fig.set_figheight(10)
    fig.set_figwidth(12)
    fig.savefig('bandwidth.pdf')
    fig.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Input parameters to generalized"+
        " Hui-Walter model.", formatter_class=RawTextHelpFormatter)
    parser.add_argument('-s','--stdev', metavar='S', type=float, nargs=4, 
                    help='Input the four test standard deviations:\n'
                    +'test1pos test1neg test2pos test2neg')
    parser.add_argument('-m','--mean', metavar='x', type=float, nargs=4,
                        help='Input the four test means:\n'+
                        'test1pos test1neg test2pos test2neg')
    parser.add_argument('-p','--prev', metavar='p', type=float, nargs=2,
                        help='Input the two population prevalences:\n'+
                        'pop1 pop2')
    parser.add_argument('-n','--nsamp', metavar='N', type=int, nargs=1,
                        help='Number of samples in each population.')
    parser.add_argument('-g','--grid', metavar='N', type=int, nargs=1,
                        help='Number of samples in the brute-force grid.')
    parser.add_argument('-r','--range', metavar='x', type=float, nargs=2,
                        help='Range to perform brute-force search:\n'+
                        'band1max band2max')
    
    args = parser.parse_args()
    stdev = args.stdev
    mean = args.mean
    bounds = args.range
    if stdev:
        sigma=[[stdev[0], stdev[1]], [stdev[2], stdev[3]]]
    else:
        sigma = None
    if mean:
        m = [[mean[0], mean[1]], [mean[2], mean[3]]]
    else:
        m = None
    nsamp = args.nsamp[0] if args.nsamp else None
    grid = args.grid[0] if args.grid else None
    if bounds:
        bounds = [(0,bounds[0]),(0,bounds[1])]

    sys.exit(main(sigma=sigma, mu=m, P=args.prev, N=nsamp, Ns=grid,
                  bound=bounds))
