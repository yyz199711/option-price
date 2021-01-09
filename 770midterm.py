#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:14:53 2020

@author: yuyang zhang
"""
import numpy as np

from math import exp, log, sqrt, pi

from scipy.stats import norm, multivariate_normal


def alpha(r, delta, sigma):
    return r - delta + sigma ** 2 / 2


def D(S, K, r, delta, sigma, t, T):
    tau = T - t
    x = log(S / K) + alpha(r, delta, sigma) * tau
    
    return x / (sigma * sqrt(tau))

def AON(S, K, r, delta, sigma, t, T):
    tau = T - t
    return S * exp(-delta * tau) * norm.cdf(D(S, K, r, delta, sigma, t, T))

def CON(S, K, Q, r, delta, sigma, t, T):
    tau = T - t
    return Q * exp(- r * tau) * norm.cdf(D(S, K, r, delta, sigma, t, T) - sigma * sqrt(tau))

def Euro_call(S, K, r, delta, sigma, t, T) :
    tau = T - t
    H = S * exp(- delta * tau) * norm.cdf(D(S, K, r, delta, sigma, t, T))
    B = K * exp(- r * tau) * norm.cdf(D(S, K, r, delta, sigma, t, T) - sigma * sqrt(tau))
    
    return H - B


def Euro_put(S, K, r, delta, sigma, t, T):
    return Euro_call(K, S, delta, r, sigma, t, T)

def A(r, delta, sigma, tau):
    d = sqrt(tau) * alpha(r, delta, sigma) / sigma 
    H = exp(-delta * tau) * norm.cdf(d)
    B = exp(-r * tau) * norm.cdf(d - sigma * sqrt(tau))
    
    return H - B

def A_p(r, delta, sigma, tau):
    d = sqrt(tau) * alpha(r, delta, sigma) / sigma
    H = exp(-r * tau) * norm.cdf(-d + sigma * sqrt(tau))
    B = exp(-delta * tau) * norm.cdf(-d)
    
    return H - B

def FS_call(S, r, delta, sigma, t, T1, T2) :
    tau = T2 - T1
    return S * exp(-delta * (T1 - t)) * A(r, delta, sigma, tau)

def FS_put(S, r, delta, sigma, t, T1, T2) :
    tau1 = T1 - t
    # tau2 = T2 - t
    tau12 = T2 - T1
    
    return S * exp(-delta * tau1) * A_p(r, delta, sigma, tau12)
    
    # FScall = FS_call(S, r, delta, sigma, t, T1, T2)
    
    # B1 = S * exp(-delta * tau1) * exp(-r * tau12)
    # B2 = S * exp(-delta * tau2)
    
    # return FScall + B1 - B2 Here is Put-Call parity.

def RSO_call(S, H, K, r, delta, sigma, t, T1, T2) :
    tau1 = T1 - t
    tau2 = T2 - t
    tau12 = T2 - T1
    B1 = S * exp(-delta * tau1) * norm.cdf(-D(S, H, r, delta, sigma, t, T1)) * A(r, delta, sigma, tau12)
    
    tau2 = T2 - t
    rho = sqrt(tau1 / tau2)
    cov1 = np.array([[1, rho],[rho, 1]])
    mean1 = np.array([0,0])
    x = np.array([D(S, H, r, delta, sigma, t, T1), D(S, K, r, delta, sigma, t, T2)])
    N2 = multivariate_normal.cdf(x, mean = mean1, cov = cov1)
    B2 = S * exp(-delta * tau2) * N2
    
    y = np.array([D(S, H, r, delta, sigma, t, T1) - sigma * sqrt(tau1), D(S, K, r, delta, sigma, t, T2) - sigma * sqrt(tau2)])
    N3 = multivariate_normal.cdf(y, mean = mean1, cov = cov1)
    B3 = K * exp(-r * tau2) * N3
    
    return B1 + B2 - B3

def RSO_put(S, H, K, r, delta, sigma, t, T1, T2):
    tau1 = T1 - t
    tau2 = T2 - t
    tau12 = T2 - T1
    mean1 = np.zeros(2)
    rho = sqrt(tau1 / tau2)
    cov1 = np.array([[1, rho], [rho, 1]])
    A_ptau12 = A_p(r, delta, sigma, tau12)
    
    B1 = S * exp(-delta * tau1) * norm.cdf(D(S,K,r,delta,sigma,t, T1)) * A_ptau12
    
    x = [-D(S,H,r,delta,sigma,t,T1) + sigma * sqrt(tau1), -D(S,K,r,delta,sigma,t,T2) + sigma * sqrt(tau2)]
    N2 = multivariate_normal.cdf(x, mean=mean1, cov=cov1)
    B2 = K * exp(-r * tau2) * N2
    
    y = [-D(S,H,r,delta,sigma,t,T1),-D(S,K,r,delta,sigma,t,T2)]
    N3 = multivariate_normal.cdf(y, mean=mean1, cov=cov1)
    B3 = S * exp(-delta * tau1) * N3
    
    return B1 + B2 - B3

def reverse_BS_call(a, b, K, r, delta, sigma, t, T, C_target, epsilon):
    x = (b + a) / 2
    C = Euro_call(x, K, r, delta, sigma, t, T)

    while abs(C - C_target) > epsilon:
        C = Euro_call(x, K, r, delta, sigma, t, T)
        if C > C_target :
            b = x
        else:
            a = x
        x = (b + a) / 2
    return x

def reverse_BS_put(a, b, K, r, delta, sigma, t, T, P_target, epsilon):
    x = (b + a) / 2
    P = Euro_put(x, K, r, delta, sigma, t, T)

    while abs(P - P_target) > epsilon:
        P = Euro_put(x, K, r, delta, sigma, t, T)
        if P > P_target :
            a = x
        else:
            b = x
        x = (b + a) / 2
    return x



def CC(S, K1, K2, r, delta, sigma, t, T1, T2, a, b, epsilon):
    S_star = reverse_BS_call(a, b, K2, r, delta, sigma, T1, T2, K1, epsilon)
    print(S_star)
    
    tau1 = T1 - t
    tau2 = T2 - t
    tau12 = T2 - T1
    mean1 = np.array([0,0])
    rho = sqrt(tau1 / tau2)
    cov1 = np.array([[1, rho], [rho, 1]])
    
    x = [D(S, S_star, r, delta, sigma, t, T1), D(S, K2, r, delta, sigma, t, T2)]
    N1 = multivariate_normal.cdf(x, mean = mean1, cov = cov1)
    B1 = S * exp(-delta * tau2) * N1
    
    y = [D(S, S_star, r, delta, sigma, t, T1) - sigma * sqrt(tau1), D(S, K2, r, delta, sigma, t, T2) - sigma * sqrt(tau2)]
    N2 = multivariate_normal.cdf(y, mean = mean1, cov = cov1)
    B2 = K2 * exp(-r * tau2) * N2
    
    B3 = K1 * exp(-r * tau1) * norm.cdf(D(S, S_star, r, delta, sigma, t, T1) - sigma * sqrt(tau1))
    
    return B1 - B2 - B3

def PC(S, K1, K2, r, delta, sigma, t, T1, T2, a, b, epsilon):
    S_star = reverse_BS_call(a, b, K2, r, delta, sigma, T1, T2, K1, epsilon)
    tau1 = T1 - t
    tau12 = T2 - T1
    tau2 = T2 - t
    B1 = K1 * exp(-r * tau1) * norm.cdf(-D(S,S_star,r,delta,sigma,t,T1) + sigma * sqrt(tau1))
    mean1 = np.array([0,0])
    rho = sqrt(tau1 / tau2)
    cov1 = np.array([[1, -rho], [-rho, 1]])
    
    x = [-D(S, S_star, r, delta, sigma, t, T1) + sigma * sqrt(tau1), D(S, K2, r, delta, sigma, t, T2) - sigma * sqrt(tau2)]
    N1 = multivariate_normal.cdf(x, mean=mean1, cov=cov1)
    B2 = K2 * exp(-r * tau2) * N1
    
    y = [-D(S, S_star, r, delta, sigma, t, T1), D(S, K2, r, delta, sigma, t, T2)]
    N2 = multivariate_normal.cdf(y, mean=mean1, cov=cov1)
    B3 = S * exp(-delta * tau2) * N2
    
    return B1 + B2 - B3
    
    
    # B1 = CC(S,K1,K2,r,delta,sigma,t,T1,T2,a,b,epsilon)
    # B2 = K1 * exp(-r * tau1)
    # B3 = Euro_call(S,K2,r,delta,sigma,t,T2)
    
    # return B1 + B2 - B3 This is put-call parity.

def CP(S, K1, K2, r, delta, sigma, t, T1, T2, a, b, epsilon):
    S_star = reverse_BS_put(a, b, K2, r, delta, sigma, T1, T2, K1, epsilon)
    
    tau1 = T1 - t
    tau2 = T2 - t
    tau12 = T2 - T1
    mean1 = np.array([0,0])
    rho = sqrt(tau1 / tau2)
    cov1 = np.array([[1, rho], [rho, 1]])
    
    x = [-D(S, S_star, r, delta, sigma, t, T1) + sigma * sqrt(tau1), -D(S, K2, r, delta, sigma, t, T2) + sigma * sqrt(tau2)]
    N1 = multivariate_normal.cdf(x, mean = mean1, cov = cov1)
    B1 = K2 * exp(-r * tau2) * N1
    
    y = [-D(S, S_star, r, delta, sigma, t, T1), -D(S, K2, r, delta, sigma, t, T2)]
    N2 = multivariate_normal.cdf(y, mean = mean1, cov = cov1)
    B2 = S * exp(-delta * tau2) * N2
    
    B3 = K1 * exp(-r * tau1) * norm.cdf(-D(S, S_star, r, delta, sigma, t, T1) + sigma * sqrt(tau1))
    
    return B1 - B2 - B3

def PP(S, K1, K2, r, delta, sigma, t, T1, T2, a, b, epsilon):
    S_star = reverse_BS_put(a, b, K2, r, delta, sigma, T1, T2, K1, epsilon)
    
    tau1 = T1 - t
    tau2 = T2 - t
    mean1 = np.array([0,0])
    rho = sqrt(tau1 / tau2)
    cov1 = np.array([[1, -rho], [-rho, 1]])
    
    B1 = K1 * exp(-r * tau1) * norm.cdf(D(S,S_star,r,delta,sigma,t,T1)-sigma*sqrt(tau1))
    
    x = [D(S, S_star, r, delta, sigma, t, T1) - sigma * sqrt(tau1), -D(S, K2, r, delta, sigma, t, T2) + sigma * sqrt(tau2)]
    
    N1 = multivariate_normal.cdf(x, mean = mean1, cov = cov1)
    B2 = K2 * exp(-r * tau2) * N1
    
    y = [D(S, S_star, r, delta, sigma, t, T1), -D(S, K2, r, delta, sigma, t, T2)]
    N2 = multivariate_normal.cdf(y, mean = mean1, cov = cov1)
    
    B3 = S * exp(-delta * tau2) * N2
    
    return B1 - B2 + B3
    
    # B1 = CP(S, K1, K2, r, delta, sigma, t, T1, T2, a, b, epsilon)
    # B2 = K1 * exp(-r * tau1)
    # B3 = Euro_put(S,K2,r,delta,sigma,t,T2)
    
    # return B1 + B2 - B3

def CH_same_Strike(S, K, r, delta, sigma, t, T1, T2):
    tau12 = T2 - T1
    B1 = Euro_call(S, K, r, delta, sigma, t, T2)
    K1 = K * exp(-(r-delta)*tau12)
    B2 = exp(-delta * tau12) * Euro_put(S, K1, r, delta, sigma, t, T1)
    
    return B1 + B2

def reverse_call_put(a, b, Kc, Kp, r, delta, sigma, t, Tc, Tp, epsilon):
    x = (a + b) / 2
    cp = Euro_call(x, Kc, r, delta, sigma, t, Tc) - Euro_put(x, Kp, r, delta, sigma, t, Tp)
    
    while abs(cp) > epsilon:
        cp = Euro_call(x, Kc, r, delta, sigma, t, Tc) - Euro_put(x, Kp, r, delta, sigma, t, Tp)
        if cp > 0 :
            b = x
        else:
            a = x
        x = (a + b) / 2
    
    return x
    

def CH(S, Kc, Kp, r, delta, sigma, t, T1, Tc, Tp, a, b, epsilon):
    S_star = reverse_call_put(a, b, Kc, Kp, r, delta, sigma, T1, Tc, Tp, epsilon)
    
    tauc = Tc - t
    taup = Tp - t
    tau1 = T1 - t
    mean1 = np.zeros(2)
    rhoc = sqrt(tau1 / tauc)
    rhop = sqrt(tau1 / taup)
    covc = np.array([[1, rhoc], [rhoc, 1]])
    covp = np.array([[1, rhop], [rhop, 1]])
    
    x = [D(S, S_star, r, delta, sigma, t, T1), D(S, Kc, r, delta, sigma, t, Tc)]
    N1 = multivariate_normal.cdf(x, mean=mean1, cov=covc)
    B1 = S * exp(-delta * tauc) * N1
    
    y = [D(S, S_star, r, delta, sigma, t, T1) - sigma * sqrt(tau1), D(S, Kc, r, delta, sigma, t, Tc) - sigma * sqrt(tauc)]
    N2 = multivariate_normal.cdf(y, mean=mean1, cov=covc)
    B2 = Kc * exp(-r * tauc) * N2
    
    z = [-D(S, S_star, r, delta, sigma, t, T1) + sigma * sqrt(tau1), -D(S, Kp, r, delta, sigma, t, Tp) + sigma * sqrt(taup)]
    N3 = multivariate_normal.cdf(z, mean=mean1, cov=covp)
    B3 = Kp * exp(-r * taup) * N3
    
    m = [-D(S, S_star, r, delta, sigma, t, T1), -D(S, Kp, r, delta, sigma, t, Tp)]
    N4 = multivariate_normal.cdf(m, mean=mean1, cov=covp)
    B4 = S * exp(-delta * taup) * N4
    
    return B1 - B2 + B3 - B4

def up_and_in_call(S, K, H, r, delta, sigma, t, T):
    tau = T - t
    
    if S < H and K <= H :
        alpha = (r - delta + 0.5 * sigma ** 2) / sigma ** 2
        B1 = S * exp(-delta * tau) * norm.cdf(D(S, H, r, delta, sigma, t, T))
        B2 = K * exp(-r * tau) * norm.cdf(D(S, H, r, delta, sigma, t, T) - sigma * sqrt(tau))
    
        B3 = S * exp(-delta * tau) * (S / H) ** (-2 * alpha) * \
            (norm.cdf(-D(H**2/S, K, r, delta, sigma, t, T)) - norm.cdf(-D(H,S,r,delta,sigma,t,T)))
    
        B4 = K * exp(-r * tau) * (S / H) ** (2 - 2 * alpha) * \
            (norm.cdf(-D(H**2/S, K, r, delta, sigma, t, T) + sigma * sqrt(tau)) - norm.cdf(-D(H,S,r,delta,sigma,t,T)+ sigma * sqrt(tau)))

        return B1 - B2 - B3 + B4
    else :
        return Euro_call(S, K, r, delta, sigma, t, T)


def up_and_in_put(S, K, H, r, delta, sigma, t, T):
    if S < H:
        tau = T - t
        alpha = (r - delta + 0.5 * sigma ** 2) / sigma ** 2
        B1 = up_and_in_call(S, K, H, r, delta, sigma, t, T)
        B2 = K * exp(-r * tau) * (norm.cdf(D(S,H,r,delta,sigma,t,T) - sigma * sqrt(tau)) + \
                                  (S / H) ** (2 - 2 * alpha) * norm.cdf(-D(H, S, r, delta, sigma, t, T) + sigma * sqrt(tau)))
        
        B3 = S * exp(-delta * tau) * (norm.cdf(D(S,H,r,delta,sigma,t,T)) + \
                                  (S / H) ** (-2 * alpha) * norm.cdf(-D(H, S, r, delta, sigma, t, T)))
        
        return B1 + B2 - B3
    
    else:
        return Euro_put(S, K, r, delta, sigma, t, T)

def up_and_out_call(S, K, H, r, delta, sigma, t, T):
    c_uo = Euro_call(S,K,r,delta,sigma,t,T) - up_and_in_call(S,K,H,r,delta,sigma,t,T)
    
    return c_uo

def up_and_out_put(S, K, H, r, delta, sigma, t, T):
    if S < H:
        tau = T - t
        B1 = up_and_out_call(S, K, H, r, delta, sigma, t, T)
        alpha = (r - delta + 0.5 * sigma ** 2) / sigma ** 2
        
        B2 = K * exp(-r * tau) * (norm.cdf(-D(S, H, r, delta, sigma, t, T) + sigma * sqrt(tau)) - \
                                  (S / H) ** (2 - 2 * alpha) * norm.cdf(-D(H, S, r, delta, sigma, t, T) + sigma * sqrt(tau)))
        
        B3 = S * exp(-delta * tau) * (norm.cdf(-D(S, H, r, delta, sigma, t, T)) - (S / H) ** (-2 * alpha) * \
                                      norm.cdf(-D(H, S, r, delta, sigma, t, T)))
        
        return B1 + B2 - B3
    
    else:
        return 0


def down_and_in_call(S, K, H, r, delta, sigma, t, T):
    alpha = (r - delta + 0.5 * sigma ** 2) / sigma ** 2
    tau = T - t
    if S > H and K >= H :
        B1 = S * exp(-delta * tau) * (S / H) ** (-2 * alpha) * norm.cdf(D(H**2/S, K, r, delta, sigma, t, T))
        
        B2 = K * exp(-r * tau) * (S / H) ** (2 - 2 * alpha) * norm.cdf(D(H**2/S, K, r, delta, sigma, t, T) - sigma * sqrt(tau))
        
        return B1 - B2
    if S <= H :
        return Euro_call(S, K, r, delta, sigma, t, T)
    
    else:
        B1 = Euro_put(S, K, r, delta, sigma, t, T)
        
        B2 = S * exp(-delta * tau) * (norm.cdf(-D(S, H, r, delta, sigma, t, T)) + (S / H) ** (-2 * alpha) * norm.cdf(D(H, S, r, delta, sigma, t, T)))
        
        B3 = K * exp(-r * tau) * (norm.cdf(-D(S, H, r, delta, sigma, t, T) + sigma * sqrt(tau)) + (S / H) ** (2-2 * alpha) * norm.cdf(D(H, S, r, delta, sigma, t, T) - sigma * sqrt(tau)))
        
        return B1 + B2 - B3

def down_and_in_put(S, K, H, r, delta, sigma, t, T):
    if S > H :
        tau = T - t
        alpha = (r - delta + 0.5 * sigma ** 2) / sigma ** 2
        B1 = down_and_in_call(S, K, H, r, delta, sigma, t, T)
    
        B2 = K * exp(-r * tau) * (norm.cdf(-D(S, H, r, delta, sigma, t, T) + sigma * sqrt(tau)) + (S / H) ** (2-2 * alpha) * norm.cdf(D(H, S, r, delta, sigma, t, T) - sigma * sqrt(tau)))
        
        B3 = S * exp(-delta * tau) * (norm.cdf(-D(S, H, r, delta, sigma, t, T)) + (S / H) ** (-2 * alpha) * norm.cdf(D(H, S, r, delta, sigma, t, T)))
        # print(B2, B3)
        return B1 + B2 - B3
    else:
        return Euro_put(S, K, r, delta, sigma, t, T)


def down_and_out_call(S, K, H, r, delta, sigma, t, T) :

    c_do = Euro_call(S, K, r, delta, sigma, t, T) - down_and_in_call(S, K, H, r, delta, sigma, t, T)
    
    return c_do

    

def down_and_out_put(S, K, H, r, delta, sigma, t, T):
    if S > H :
        tau = T - t
        alpha = (r - delta + 0.5 * sigma ** 2) / sigma ** 2
        B1 = down_and_out_call(S, K, H, r, delta, sigma, t, T)
    
        B2 = K * exp(-r * tau) * (norm.cdf(D(S, H, r, delta, sigma, t, T) - sigma * sqrt(tau)) - (S / H) ** (2 - 2 * alpha) * norm.cdf(D(H, S, r, delta, sigma, t, T) - sigma * sqrt(tau)))
    
        B3 = S * exp(-delta * tau) * (norm.cdf(D(S, H, r, delta, sigma, t, T)) - (S / H) ** (-2 * alpha) * norm.cdf(D(H, S, r, delta, sigma, t, T)))
    
        return B1 + B2 - B3
    else:
        return 0
    
def Shout_call(S, K, S_sh, r, delta, sigma, t, T):
    tau = T - t
    B1 = (S_sh - K) * exp(-r * tau) * norm.cdf(-D(S, S_sh, r, delta, sigma, t, T) + sigma * sqrt(tau))
    
    B2 = S * exp(-delta * tau) * norm.cdf(D(S, S_sh, r, delta, sigma, t, T))
    
    B3 = K * exp(-r * tau) * norm.cdf(D(S, S_sh, r, delta, sigma, t, T) - sigma * sqrt(tau))
    
    return B1 + B2 - B3

def Shout_put(S, K, S_sh, r, delta, sigma, t, T):
    tau = T - t
    B1 = K * exp(-r * tau) * norm.cdf(-D(S, S_sh, r, delta, sigma, t, T) + sigma * sqrt(tau))
    
    B2 = S * exp(-delta * tau) * norm.cdf(-D(S, S_sh, r, delta, sigma, t, T)) 
    
    B3 = (K - S_sh) * exp(-r * tau) * norm.cdf(D(S, S_sh, r, delta, sigma, t, T) - sigma * sqrt(tau))
    
    return B1 - B2 + B3


def GAAC(S, K, r, delta, sigma, t, T):
    tau = T - t
    b = r - delta
    b_G = (b - sigma ** 2 / 6) / 2
    sigma_G = sigma / sqrt(3)
    
    d = (log(S / K) + (b_G + 0.5 * sigma_G ** 2) * tau) / (sigma_G * sqrt(tau))
    
    B1 = S * exp((b_G - r) * tau) * norm.cdf(d)
    B2 = K * exp(-r * tau) * norm.cdf(d - sigma_G * sqrt(tau))
    
    return B1 - B2


def GAAP(S, K, r, delta, sigma, t, T):
    tau = T - t
    b = r - delta
    b_G = (b - sigma ** 2 / 6) / 2
    sigma_G = sigma / sqrt(3)
    
    d = (log(S / K) + (b_G + 0.5 * sigma_G ** 2) * tau) / (sigma_G * sqrt(tau))
    
    B1 = K * exp(-r * tau) * norm.cdf(-d + sigma_G * sqrt(tau))
    
    B2 = S * exp((b_G - r)* tau) * norm.cdf(-d)
    
    return B1 - B2


def FLSLC(S, S_min, r, delta, sigma, t, T):
    tau = T - t
    B1 = S * exp(-delta * tau) * (norm.cdf(D(S, S_min, r, delta, sigma, t, T)) -\
                                  sigma ** 2 / (2 * (r - delta)) * norm.cdf(-D(S, S_min, r,delta, sigma, t, T)))
    B2 = S_min * exp(-r * tau) * (norm.cdf(D(S, S_min, r, delta, sigma, t, T) - sigma * sqrt(tau)) -\
                                  sigma ** 2 / (2 * (r - delta)) * (S/S_min) ** (1 - 2*(r-delta)/sigma**2) * norm.cdf(D(S_min, S, r,delta, sigma, t, T) - sigma * sqrt(tau)))

    
    
    return B1 - B2

def FISLP(S, S_min, K, r, delta, sigma, t, T):
    if S_min <= K :
        tau = T - t 
        B1 = FLSLC(S, S_min, r, delta, sigma, t, T)
        B2 = S * exp(-delta * tau)
        B3 = K * exp(-r * tau)
        
        return B1 - B2 + B3
    else:
        print("S_min > K")

def FLSLP(S, S_max, r, delta, sigma, t, T):
    tau = T - t
    B1 = S_max * exp(-r * tau) * (norm.cdf(-D(S,S_max,r,delta,sigma,t,T) + sigma * sqrt(tau))-\
                                  sigma ** 2 / (2 * (r - delta)) * (S / S_max) ** (1 - 2 * (r-delta)/sigma**2) * norm.cdf(-D(S_max, S, r, delta, sigma, t, T)+sigma * sqrt(tau)))
 
    B2 = S * exp(-delta * tau) * (norm.cdf(-D(S, S_max, r, delta, sigma, t, T)) -\
                                  sigma ** 2 / (2 * (r-delta)) * norm.cdf(D(S, S_max, r,delta, sigma, t, T)))
    
    
    return B1 - B2

def FISLC(S, S_max, K, r, delta, sigma, t, T):
    if S_max >= K :
        tau = T - t
        B1 = FLSLP(S, S_max, r, delta, sigma, t, T)
        B2 = S * exp(-delta * tau)
        B3 = K * exp(-r * tau)
        
        return B1 + B2 - B3
    else:
        print("S_max < K")
    

def EO(S1,S2, delta1, delta2, sigma1, sigma2, t, T, rho):
    tau = T - t
    sigma_R = sqrt(sigma2**2 + sigma1**2 - 2*sigma1*sigma2*rho)
    # print(sigma_R)
    d = (log(S2/S1) + (delta1 - delta2 + 0.5 * sigma_R**2) * tau) / (sigma_R * sqrt(tau))
    
    B1 = S2 * exp(-delta2 * tau) * norm.cdf(d)
    
    B2 = S1 * exp(-delta1 * tau) * norm.cdf(d - sigma_R * sqrt(tau))
    # print(B1, B2)
    
    return B1 - B2

def max_call(S1, S2, K, r, delta1, delta2, sigma1, sigma2, t, T, rho):
    tau = T - t
    sigma = sqrt(sigma1 ** 2 + sigma2 ** 2 - 2 * rho * sigma1 * sigma2)
    d = (log(S1/S2) + (delta2 - delta1 + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    
    y1 = D(S1, K, r, delta1, sigma1, t, T)
    y2 = D(S2, K, r, delta2, sigma2, t, T)
    rho1 = (sigma1 - sigma2 * rho) / sigma
    rho2 = (sigma2 - sigma1 * rho) / sigma
    mean1 = np.zeros(2)
    cov1 = np.array([[1,rho1],[rho1, 1]])
    x1 = [y1, d]
    N1 = multivariate_normal.cdf(x1, mean = mean1, cov=cov1)
    
    B1 = S1 * exp(-delta1 * tau) * N1
    
    cov2 = np.array([[1,rho2],[rho2,1]])
    x2 = [y2, -d + sigma * sqrt(tau)]
    N2 = multivariate_normal.cdf(x2, mean=mean1, cov=cov2)
    B2 = S2 * exp(-delta2 * tau) * N2
    
    cov3 = np.array([[1,rho],[rho,1]])
    x3 = [-y1 + sigma1 * sqrt(tau), -y2 + sigma2 * sqrt(tau)]
    N3 = multivariate_normal.cdf(x3, mean=mean1, cov=cov3)
    B3 = K * exp(-r * tau) * (1 - N3)
    
    return B1 + B2 - B3

def min_call(S1, S2, K, r, delta1, delta2, sigma1, sigma2, t, T, rho):
    tau = T - t
    sigma = sqrt(sigma1 ** 2 + sigma2 ** 2 - 2 * rho * sigma1 * sigma2)
    d = (log(S1/S2) + (delta2 - delta1 + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    
    y1 = D(S1, K, r, delta1, sigma1, t, T)
    y2 = D(S2, K, r, delta2, sigma2, t, T)
    rho1 = (sigma1 - sigma2 * rho) / sigma
    rho2 = (sigma2 - sigma1 * rho) / sigma
    mean1 = np.zeros(2)
    
    cov1 = np.array([[1,-rho1],[-rho1, 1]])
    x1 = [y1, -d]
    N1 = multivariate_normal.cdf(x1, mean=mean1, cov=cov1)
    B1 = S1 * exp(-delta1 * tau) * N1
    
    cov2 = np.array([[1,-rho2],[-rho2, 1]])
    x2 = [y2, d - sigma * sqrt(tau)]
    N2 = multivariate_normal.cdf(x2, mean=mean1, cov=cov2)
    B2 = S2 * exp(-delta2 * tau) * N2
    
    cov3 = np.array([[1, rho], [rho, 1]])
    x3 = [y1 - sigma1 * sqrt(tau), y2 - sigma2 * sqrt(tau)]
    N3 = multivariate_normal.cdf(x3, mean=mean1, cov=cov3)
    B3 = K * exp(-r * tau) * N3
    
    return B1 + B2 - B3
    
def max_put(S1, S2, K, r, delta1, delta2, sigma1, sigma2, t, T, rho):
    B1 = max_call(S1, S2, K, r, delta1, delta2, sigma1, sigma2, t, T, rho)
    tau = T - t
    B2 = K * exp(-r * tau)
    
    B3 = S1 * exp(-delta1 * tau)
    
    B4 = EO(S1, S2, delta1, delta2, sigma1, sigma2, t, T, rho)
    
    return B1 + B2 - B3 - B4


def min_put(S1, S2, K, r, delta1, delta2, sigma1, sigma2, t, T, rho):
    tau = T - t
    B1 = K * exp(-r * tau)
    
    B2 = S2 * exp(-delta2 * tau)
    
    B3 = EO(S1, S2, delta1, delta2, sigma1, sigma2, t, T, rho)
    
    B4 = min_call(S1, S2, K, r, delta1, delta2, sigma1, sigma2, t, T, rho)
    
    return B1 - B2 + B3 + B4
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    