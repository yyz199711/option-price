#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 07:22:54 2020

@author: shousakai
"""

import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm

def B_infty(r, delta, sigma, K):
    b = delta - r + 0.5 * sigma**2
    f = (b**2 + 2*r*sigma**2)**0.5
    
    return K*(b+f)/(b+f-sigma**2)


def alpha(r, delta, sigma):
    return r - delta + sigma ** 2 / 2

def D(S, K, r, delta, sigma, t, T):
    tau = T - t
    x = log(S / K) + alpha(r, delta, sigma) * tau
    
    return x / (sigma * sqrt(tau))

def C_infty(S,K,r,delta, sigma):
    B = B_infty(r, delta, sigma, K)
    b = delta - r + 0.5 * sigma**2
    f = (b**2 + 2*r*sigma**2)**0.5
    alpha = 0.5*(b+f)
    
    return (B - K)*(S/B)**(2*alpha/sigma**2)


def auto_cap(S,K,L,r,delta,sigma,t,T):
    tau = T - t   
    b = delta - r + 0.5 * sigma**2
    f = (b**2 + 2*r*sigma**2)**0.5
    alpha = 0.5*(b+f)
    phi = 0.5*(b-f)
    lam = S/L
    d1_minus_L = (-log(lam)-log(L)+log(L)+b*tau)/(sigma*sqrt(tau))
    d1_minus_K = (-log(lam)-log(L)+log(K)+b*tau)/(sigma*sqrt(tau))
    d1_plus_L = (log(lam)-log(L)+log(L)+b*tau)/(sigma*sqrt(tau))
    d1_plus_K = (log(lam)-log(L)+log(K)+b*tau)/(sigma*sqrt(tau))
    
    d_0 = (log(lam) - f*tau)/(sigma*sqrt(tau))
    
    part1 = (L-K)*(lam**(2*phi/sigma**2)*norm.cdf(d_0)+lam**(2*alpha/sigma**2)*norm.cdf(d_0+2*f*sqrt(tau)/sigma))
    
    part2 = S*exp(-delta*tau)*(norm.cdf(d1_minus_L-sigma*sqrt(tau))-norm.cdf(d1_minus_K-sigma*sqrt(tau)))
    
    part3 = lam**(-2*(r-delta)/sigma**2)*L*exp(-delta*tau)*(norm.cdf(d1_plus_L-sigma*sqrt(tau))-norm.cdf(d1_plus_K-sigma*sqrt(tau)))
    
    part4 = K*exp(-r*tau)*(norm.cdf(d1_minus_L)-norm.cdf(d1_minus_K)-lam**(1-2*(r-delta)/sigma**2)*(norm.cdf(d1_plus_L)-norm.cdf(d1_plus_K)))
    
    
    return part1 + part2 - part3 - part4
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    