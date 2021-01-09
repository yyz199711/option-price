#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 02:31:53 2020

@author: shousakai
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import fsolve


def simulate_dw(steps):
    dw = np.random.normal(0, 1, size=steps)
    return dw

def simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw):
    sigma_list = np.array([sigma])
    for c in dw:
        sigma = sigma + (sigma_bar - sigma) * kappa * h + gamma * c * np.sqrt(h)
        sigma_list = np.append(sigma_list, sigma)
    return sigma_list


def get_d(S, K, r, delta, tau, rho, h, sigma_list, dw):
    v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
    d = (np.log(S/K)+(r-delta+0.5*v_bar)*tau+rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))) / (np.sqrt((1-rho **2)*v_bar*tau))
    return d

def conditional_call(S, K, r, delta, tau, rho, h, sigma_list, dw):
    v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
    N1 = norm.cdf(get_d(S, K, r, delta, tau, rho, h, sigma_list, dw) - rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
    N2 = norm.cdf(get_d(S, K, r, delta, tau, rho, h, sigma_list, dw) - np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
    c = S * np.exp(- delta * tau) * eta * N1 - K * np.exp(-r * tau) * N2
    return c


def general_call(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau, rho_s_sigma, rho_s_y):
    h = tau/steps
    c_list = np.array([])
    rho = np.sqrt(rho_s_sigma**2+rho_s_y**2)
    for i in range(M):
        dw_sigma = simulate_dw(steps)
        dw_y = simulate_dw(steps)
        dw = (rho_s_sigma * dw_sigma + rho_s_y * dw_y)/rho
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw_sigma)
        c = conditional_call(S, K, r, delta, tau, rho, h, sigma_list, dw)
        c_list = np.append(c_list, c)
    
    return np.mean(c_list)

def call(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau, rho):
    h = tau / steps
    c_list = np.array([])
    for i in range(M):
        dw = simulate_dw(steps)
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        c = conditional_call(S, K, r, delta, tau, rho, h, sigma_list, dw)
        c_list = np.append(c_list, c)
    return np.mean(c_list)



def conditional_put(S, K, r, delta, tau, rho, h, sigma_list, dw):
    v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
    N1 = norm.cdf(-get_d(S, K, r, delta, tau, rho, h, sigma_list, dw) + rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
    N2 = norm.cdf(-get_d(S, K, r, delta, tau, rho, h, sigma_list, dw) + np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
    p = -S * np.exp(- delta * tau) * eta * N1 + K * np.exp(-r * tau) * N2
    return p


def put(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau, rho):
    h = tau / steps
    p_list = np.array([])
    for i in range(M):
        dw = simulate_dw(steps)
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        p = conditional_put(S, K, r, delta, tau, rho, h, sigma_list, dw)
        p_list = np.append(p_list, p)
    return np.mean(p_list)


'''Heston'''

def simulate_v(kappa, v, v_bar, eta, h, dw):
    v_list = np.array([v])
    for c in dw:
        if v < 0:
            v = 0
            v = v + (v_bar - v) * kappa * h + eta * np.sqrt(v) * c * np.sqrt(h)
        v_list = np.append(v_list, v)
    return v_list

def call2(M, steps, kappa, v, v_bar, eta, S, K, r, delta, tau, rho):
    h = tau / steps
    c_list = np.array([])
    for i in range(M):
        dw = simulate_dw(steps)
        v_list = simulate_v(kappa, v, v_bar, eta, h, dw)
        sigma_list = np.sqrt(v_list)
        c = conditional_call(S, K, r, delta, tau, rho, h, sigma_list, dw)
        c_list = np.append(c_list, c)
    return np.mean(c_list)


def put2(M, steps, kappa, v, v_bar, eta, S, K, r, delta, tau, rho):
    h = tau / steps
    p_list = np.array([])
    for i in range(M):
        dw = simulate_dw(steps)
        v_list = simulate_v(kappa, v, v_bar, eta, h, dw)
        sigma_list = np.sqrt(v_list)
        p = conditional_put(S, K, r, delta, tau, rho, h, sigma_list, dw)
        p_list = np.append(p_list, p)
    return np.mean(p_list)



'''delta for call/put HW'''
def conditional_delta(S, K, r, delta, tau, rho, h, sigma_list, dw):
    v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
    N1 = norm.cdf(get_d(S, K, r, delta, tau, rho, h, sigma_list, dw) - rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
    conditional_delta = np.exp(- delta * tau) * eta * N1
    return conditional_delta


def delta_call(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau, rho):
    h = tau / steps
    delta_list = np.array([])
    for i in range(M):
        dw = simulate_dw(steps)
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        d = conditional_delta(S, K, r, delta, tau, rho, h, sigma_list, dw)
        delta_list = np.append(delta_list, d)
    return np.mean(delta_list)

def delta_put(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau, rho):
    return delta_call(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau, rho) - np.exp(-delta * tau)

def conditional_vega(S, K, r, delta, tau, rho, h, sigma_list, kappa, dw):
    v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
    N1 = norm.cdf(get_d(S, K, r, delta, tau, rho, h, sigma_list, dw) - rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
    n = norm.pdf(get_d(S, K, r, delta, tau, rho, h, sigma_list, dw) - np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
    e_list = np.array([np.exp(-kappa * i) for i in np.arange(0, tau, h)])
    part1 = S * np.exp(- delta * tau) * rho * eta * N1 * sum(e_list * dw)
    part2 = S * np.exp(- delta * tau) * rho ** 2 * eta * N1 * sum(e_list * sigma_list[:-1] * h)
    part3 = K * np.exp(- r * tau) * n * np.sqrt(1-rho**2) * sum(e_list * sigma_list[:-1] * h) / np.sqrt(v_bar * tau)
    vega = part1 - part2 + part3
    return vega

def vega(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau, rho):
    h = tau / steps
    vega_list = np.array([])
    for i in range(M):
        dw = simulate_dw(steps)
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        vega = conditional_vega(S, K, r, delta, tau, rho, h, sigma_list, kappa, dw)
        vega_list = np.append(vega_list, vega)
    return np.mean(vega_list)


'''Forward Start Option HW'''
def conditional_A(S, K, r, delta, tau, rho, h, sigma_list, dw):
    v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
    N1 = norm.cdf(get_d(S, K, r, delta, tau, rho, h, sigma_list, dw) - rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
    N2 = norm.cdf(get_d(S, K, r, delta, tau, rho, h, sigma_list, dw) - np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
    A = np.exp(- delta * tau) * eta * N1 - np.exp(-r * tau) * N2
    return A

def conditional_B(S, K, r, delta, tau, rho, h, sigma_list, dw):
    v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
    N1 = norm.cdf(-get_d(S, K, r, delta, tau, rho, h, sigma_list, dw) + rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
    N2 = norm.cdf(-get_d(S, K, r, delta, tau, rho, h, sigma_list, dw) + np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
    B = -np.exp(- delta * tau) * eta *N1 + np.exp(-r * tau) * N2
    return B


def fs_call(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau1, tau2, rho):   
    h1 = tau1 / steps
    tau12 = tau2 - tau1
    h12 = tau12/ steps
    fs_c_list = np.array([])
    for i in range(M):
        dw1 = simulate_dw(steps)
        dw1s = rho*dw1 + np.sqrt(1-rho**2)*simulate_dw(steps)
        dw12 = simulate_dw(steps)
        sigma_list1 = simulate_sigma(kappa, sigma, sigma_bar, gamma, h1, dw1)
        sigma_1 = sigma_list1[-1]
        sigma_list12 = simulate_sigma(kappa, sigma_1, sigma_bar, gamma, h12, dw12)
        a = conditional_A(S, S, r, delta, tau12, rho, h12, sigma_list12, dw12)
        v_bar = sum(sigma_list1[:-1] ** 2 * h1) / tau1
        eta = np.exp((sum(sigma_list1[:-1]*dw1s*np.sqrt(h1)))-0.5*v_bar*tau1)
        fs_c = S * np.exp(- delta * tau1) * eta * a
        fs_c_list = np.append(fs_c_list, fs_c)
    return np.mean(fs_c_list)

def fs_put(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau1, tau2, rho):
    h1 = tau1 / steps
    tau12 = tau2 - tau1
    h12 = tau12 / steps
    fs_p_list = np.array([])
    for i in range(M):
        dw1 = simulate_dw(steps)
        dw1s = rho*dw1 + np.sqrt(1-rho**2)*simulate_dw(steps)
        dw12 = simulate_dw(steps)
        sigma_list1 = simulate_sigma(kappa, sigma, sigma_bar, gamma, h1, dw1)
        sigma_1 = sigma_list1[-1]
        sigma_list12 = simulate_sigma(kappa, sigma_1, sigma_bar, gamma, h12, dw12)
        b = conditional_B(S, S, r, delta, tau12, rho, h12, sigma_list12, dw12)
        v_bar = sum(sigma_list1[:-1] ** 2 * h1) / tau1
        eta = np.exp((sum(sigma_list1[:-1]*dw1s*np.sqrt(h1)))-0.5*v_bar*tau1)
        fs_p = S * np.exp(- delta * tau1) * eta * b
        fs_p_list = np.append(fs_p_list, fs_p)
    return np.mean(fs_p_list)


'''Cliquet HW'''
def cliquet_call(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau_list, rho):
    cliquet = 0
    for i in range(len(tau_list) - 1):
        tau1 = tau_list[i]
        tau2 = tau_list[i + 1]
        cliquet += fs_call(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau1, tau2, rho)
    return cliquet

def cliquet_put(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau_list, rho):
    cliquet = 0
    for i in range(len(tau_list) - 1):
        tau1 = tau_list[i]
        tau2 = tau_list[i + 1]
        cliquet += fs_put(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau1, tau2, rho)
    return cliquet



'''Reset HW'''
def conditional_RSC(S, K, r, delta, tau1, tau2, rho, h1, h12, h2, sigma_list1, sigma_list12, dw1, dw12):
    dw2 = np.append(dw1, dw12)
    sigma_list2 = np.append(sigma_list1, sigma_list12[1:])
    v_bar1 = sum(sigma_list1[:-1] ** 2 * h1) / tau1
    v_bar2 = sum(sigma_list2[:-1] ** 2 * h2) / tau2
    d11 = get_d(S, K, r, delta, tau1, rho, h1, sigma_list1, dw1) - rho**2*np.sqrt(v_bar1*tau1)/np.sqrt(1-rho**2)
    d21 = get_d(S, K, r, delta, tau2, rho, h2, sigma_list2, dw2) - rho**2*np.sqrt(v_bar2*tau2)/np.sqrt(1-rho**2)
    d12 = get_d(S, K, r, delta, tau1, rho, h1, sigma_list1, dw1) - np.sqrt(v_bar1*tau1)/np.sqrt(1-rho**2)
    d22 = get_d(S, K, r, delta, tau2, rho, h2, sigma_list2, dw2) - np.sqrt(v_bar2*tau2)/np.sqrt(1-rho**2)  
    rho12 = np.sqrt(v_bar1 * tau1 / (v_bar2 * tau2))
    N1 = norm.cdf(-d11)
    N2 = multivariate_normal.cdf(np.array([d11, d21]), mean = np.array([0,0]), cov = np.array([[1, rho12],[rho12, 1]]))
    N3 = multivariate_normal.cdf(np.array([d12, d22]), mean = np.array([0,0]), cov = np.array([[1, rho12],[rho12, 1]]))
    eta1 = np.exp((sum(sigma_list1[:-1] * dw1 * np.sqrt(h1))) - 0.5 * v_bar1 * tau1)
    eta2 = np.exp((sum(sigma_list2[:-1] * dw2 * np.sqrt(h2))) - 0.5 * v_bar2 * tau2)
    a = conditional_A(S, K, r, delta, tau2 - tau1, rho, h12, sigma_list12, dw12)
    rsc = S * np.exp(- delta * tau1) * eta1 * N1 * a + S * np.exp(- delta * tau2) * eta2 * N2 - K * np.exp(- r * tau2) * N3
    return rsc

def RSC(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau1, tau2, rho):
    h1 = tau1 / steps
    h2 = tau2 / (2 * steps)
    h12 = (tau2 - tau1) / steps
    rsc_list = np.array([])
    for i in range(M):
        dw1 = simulate_dw(steps)
        dw12 = simulate_dw(steps)
        sigma_list1 = simulate_sigma(kappa, sigma, sigma_bar, gamma, h1, dw1)
        sigma_1 = sigma_list1[-1]
        sigma_list12 = simulate_sigma(kappa, sigma_1, sigma_bar, gamma, h12, dw12)
        rsc = conditional_RSC(S, K, r, delta, tau1, tau2, rho, h1, h12, h2, sigma_list1, sigma_list12, dw1, dw12)
        rsc_list = np.append(rsc_list, rsc)
    return np.mean(rsc_list)


def conditional_RSP(S, K, r, delta, tau1, tau2, rho, h1, h12, h2, sigma_list1, sigma_list12, dw1, dw12):
    dw2 = np.append(dw1, dw12)
    sigma_list2 = np.append(sigma_list1, sigma_list12[1:])
    v_bar1 = sum(sigma_list1[:-1] ** 2 * h1) / tau1
    v_bar2 = sum(sigma_list2[:-1] ** 2 * h2) / tau2
    d11 = get_d(S, K, r, delta, tau1, rho, h1, sigma_list1, dw1) - rho**2*np.sqrt(v_bar1*tau1)/np.sqrt(1-rho**2)
    d21 = get_d(S, K, r, delta, tau2, rho, h2, sigma_list2, dw2) - rho**2*np.sqrt(v_bar2*tau2)/np.sqrt(1-rho**2)
    d12 = get_d(S, K, r, delta, tau1, rho, h1, sigma_list1, dw1) - np.sqrt(v_bar1*tau1)/np.sqrt(1-rho**2)
    d22 = get_d(S, K, r, delta, tau2, rho, h2, sigma_list2, dw2) - np.sqrt(v_bar2*tau2)/np.sqrt(1-rho**2)  
    rho12 = np.sqrt(v_bar1 * tau1 / (v_bar2 * tau2))
    N1 = norm.cdf(d11)
    N2 = multivariate_normal.cdf(np.array([-d11, -d21]), mean = np.array([0,0]), cov = np.array([[1, rho12],[rho12, 1]]))
    N3 = multivariate_normal.cdf(np.array([-d12, -d22]), mean = np.array([0,0]), cov = np.array([[1, rho12],[rho12, 1]]))
    eta1 = np.exp((sum(sigma_list1[:-1] * dw1 * np.sqrt(h1))) - 0.5 * v_bar1 * tau1)
    eta2 = np.exp((sum(sigma_list2[:-1] * dw2 * np.sqrt(h2))) - 0.5 * v_bar2 * tau2)
    b = conditional_B(S, K, r, delta, tau2 - tau1, rho, h12, sigma_list12, dw12)
    rsp = S * np.exp(- delta * tau1) * eta1 * N1 * b - S * np.exp(- delta * tau2) * eta2 * N2 + K * np.exp(- r * tau2) * N3
    return rsp


def RSP(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau1, tau2, rho):
    h1 = tau1 / steps
    h2 = tau2 / (2 * steps)
    h12 = (tau2 - tau1) / steps
    rsp_list = np.array([])
    for i in range(M):
        dw1 = simulate_dw(steps)
        dw12 = simulate_dw(steps)
        sigma_list1 = simulate_sigma(kappa, sigma, sigma_bar, gamma, h1, dw1)
        sigma_1 = sigma_list1[-1]
        sigma_list12 = simulate_sigma(kappa, sigma_1, sigma_bar, gamma, h12, dw12)
        rsp = conditional_RSP(S, K, r, delta, tau1, tau2, rho, h1, h12, h2, sigma_list1, sigma_list12, dw1, dw12)
        rsp_list = np.append(rsp_list, rsp)
    return np.mean(rsp_list)


'''delta Heston'''
def delta_call2(M, steps, kappa, v, v_bar, eta, S, K, r, delta, tau):
    h = tau / steps
    delta_list = np.array([])
    for i in range(M):
        dw = simulate_dw(steps)
        v_list = simulate_v(kappa, v, v_bar, eta, h, dw)
        sigma_list = np.sqrt(v_list)
        d = conditional_delta(S, K, r, delta, tau, 0, h, sigma_list, dw)
        delta_list = np.append(delta_list, d)
    return np.mean(delta_list)


def delta_put2(M, steps, kappa, v, v_bar, eta, S, K, r, delta, tau):
    return delta_call2(M, steps, kappa, v, v_bar, eta, S, K, r, delta, tau) - np.exp(-delta * tau)


'''FSO heston'''
def fs_call2(M, steps, kappa, v, v_bar, eta, S, K, r, delta, tau1, tau2, rho):   
    h1 = tau1 / steps
    tau12 = tau2 - tau1
    h12 = tau12 / steps
    fs_c_list = np.array([])
    for i in range(M):
        dw1 = simulate_dw(steps)
        dw12 = simulate_dw(steps)
        v_list1 = simulate_v(kappa, v, v_bar, eta, h1, dw1)
        sigma_list1 = np.sqrt(v_list1)
        v = v_list1[-1]
        v_list12 = simulate_v(kappa, v, v_bar, eta, h12, dw12)
        sigma_list12 = np.sqrt(v_list12)
        a = conditional_A(S, S, r, delta, tau12, rho, h12, sigma_list12, dw12)
        v_bar1 = sum(sigma_list1[:-1] ** 2 * h1) / tau1
        eta1 = np.exp((sum(sigma_list1[:-1]*dw1*np.sqrt(h1)))-0.5*v_bar1*tau1)
        fs_c = S * np.exp(- delta * tau1) * eta1 * a
        fs_c_list = np.append(fs_c_list, fs_c)
    return np.mean(fs_c_list)

def fs_put2(M, steps, kappa, v, v_bar, eta, S, K, r, delta, tau1, tau2, rho):
    h1 = tau1 / steps    
    tau12 = tau2 - tau1
    h12 = tau12 / steps
    fs_p_list = np.array([])
    for i in range(M):
        dw1 = simulate_dw(steps)
        dw12 = simulate_dw(steps)
        v_list1 = simulate_v(kappa, v, v_bar, eta, h1, dw1)
        sigma_list1 = np.sqrt(v_list1)
        v = v_list1[-1]
        v_list12 = simulate_v(kappa, v, v_bar, eta, h12, dw12)
        sigma_list12 = np.sqrt(v_list12)
        b = conditional_B(S, S, r, delta, tau12, rho, h12, sigma_list12, dw12)
        v_bar1 = sum(sigma_list1[:-1] ** 2 * h1) / tau1
        eta1 = np.exp((sum(sigma_list1[:-1]*dw1*np.sqrt(h1)))-0.5*v_bar1*tau1)
        fs_p = S * np.exp(- delta * tau1) * eta1 * b
        fs_p_list = np.append(fs_p_list, fs_p)
    return np.mean(fs_p_list)

'''Cliquet Heston'''
def cliquet_call2(M, steps, kappa, v, v_bar, eta, S, K, r, delta, tau_list, rho):
    cliquet = 0
    for i in range(len(tau_list) - 1):
        tau1 = tau_list[i]
        tau2 = tau_list[i + 1]
        cliquet += fs_call2(M, steps, kappa, v, v_bar, eta, S, K, r, delta, tau1, tau2, rho)
    return cliquet


def cliquet_put2(M, steps, kappa, v, v_bar, eta, S, K, r, delta, tau_list, rho):
    cliquet = 0
    for i in range(len(tau_list) - 1):
        tau1 = tau_list[i]
        tau2 = tau_list[i + 1]
        cliquet += fs_put2(M, steps, kappa, v, v_bar, eta, S, K, r, delta, tau1, tau2, rho)
    return cliquet


'''Reset Heston'''

def RSC2(M, steps, kappa, v, v_bar, eta, S, K, r, delta, tau1, tau2, rho):
    h1 = tau1 / steps
    h2 = tau2 / (2 * steps)
    h12 = (tau2 - tau1) / steps
    rsc_list = np.array([])
    for i in range(M):
        dw1 = simulate_dw(steps)
        dw12 = simulate_dw(steps)
        v_list1 = simulate_v(kappa, v, v_bar, eta, h1, dw1)
        sigma_list1 = np.sqrt(v_list1)
        v = v_list1[-1]
        v_list12 = simulate_v(kappa, v, v_bar, eta, h12, dw12)
        sigma_list12 = np.sqrt(v_list12)
        rsc = conditional_RSC(S, K, r, delta, tau1, tau2, rho, h1, h12, h2, sigma_list1, sigma_list12, dw1, dw12)
        rsc_list = np.append(rsc_list, rsc)
    return np.mean(rsc_list)

def RSP2(M, steps, kappa, v, v_bar, eta, S, K, r, delta, tau1, tau2, rho):
    h1 = tau1 / steps
    h2 = tau2 / (2 * steps)
    h12 = (tau2 - tau1) / steps
    rsp_list = np.array([])
    for i in range(M):
        dw1 = simulate_dw(steps)
        dw12 = simulate_dw(steps)
        v_list1 = simulate_v(kappa, v, v_bar, eta, h1, dw1)
        sigma_list1 = np.sqrt(v_list1)
        v = v_list1[-1]
        v_list12 = simulate_v(kappa, v, v_bar, eta, h12, dw12)
        sigma_list12 = np.sqrt(v_list12)
        rsp = conditional_RSP(S, K, r, delta, tau1, tau2, rho, h1, h12, h2, sigma_list1, sigma_list12, dw1, dw12)
        rsp_list = np.append(rsp_list, rsp)
    return np.mean(rsp_list)

'''Asian Option HW'''

def GAAC(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau, rho):
    h = tau / steps
    gaac_list = np.array([])
    
    for i in range(M):
        dw = simulate_dw(steps)
        w = np.array([sum(dw[:i+1] * np.sqrt(h)) for i in range(steps-1)])
        w = np.append(0,w)
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        A_list = np.array([sum(sigma_list[:i+1] * h) / tau for i in range(steps)])
        A_list = np.append(0,A_list)
        v_bar = (A_list[-1])**2 + sum((A_list[:-1])**2 * h)/tau - 2*A_list[-1]*sum((A_list[:-1]) * h)/tau
        t = np.array([i*h for i in range(steps)])
        b_G = 0.5 * v_bar + 0.5 * (r - delta) - sum((sigma_list[:-1])**2 * t * h)/(2 * tau**2)
        
        # part_1 = rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))
        part_2 = rho * sum(sigma_list[:-1] * w * h)/tau
        eta = np.exp((b_G - 0.5 * rho**2 * v_bar) * tau + part_2)
        
        d_sigma = (np.log(S/K) + (b_G + 0.5*v_bar)*tau)/(np.sqrt((1-rho**2)*v_bar*tau))
        
        N1 = norm.cdf(d_sigma - (rho**2 * v_bar * tau - part_2)/(np.sqrt((1-rho**2) * v_bar * tau)))
        
        N2 = norm.cdf(d_sigma - (v_bar * tau - part_2)/(np.sqrt((1-rho**2) * v_bar * tau)))
        
        conditional_gaac = S * np.exp(-r*tau) * eta * N1 - K * np.exp(-r*tau) * N2
        
        gaac_list = np.append(gaac_list, conditional_gaac)
        
    return np.mean(gaac_list)

def GAAP(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau, rho):
    h = tau / steps
    gaap_list = np.array([])
    
    for i in range(M):
        dw = simulate_dw(steps)
        w = np.array([sum(dw[:i+1] * np.sqrt(h)) for i in range(steps-1)])
        w = np.append(0,w)
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        A_list = np.array([sum(sigma_list[:i+1] * h) / tau for i in range(steps)])
        A_list = np.append(0,A_list)
        v_bar = (A_list[-1])**2 + sum((A_list[:-1])**2 * h)/tau - 2*A_list[-1]*sum((A_list[:-1]) * h)/tau
        t = np.array([i*h for i in range(steps)])
        b_G = 0.5 * v_bar + 0.5 * (r - delta) - sum((sigma_list[:-1])**2 * t * h)/(2 * tau**2)
        
        # part_1 = rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))
        part_2 = rho * sum(sigma_list[:-1] * w * h)/tau
        eta = np.exp((b_G - 0.5 * rho**2 * v_bar) * tau + part_2)
        
        d_sigma = (np.log(S/K) + (b_G + 0.5*v_bar)*tau)/(np.sqrt((1-rho**2)*v_bar*tau))
        
        N1 = norm.cdf(-d_sigma + (rho**2 * v_bar * tau - part_2)/(np.sqrt((1-rho**2) * v_bar * tau)))
        
        N2 = norm.cdf(-d_sigma + (v_bar * tau - part_2)/(np.sqrt((1-rho**2) * v_bar * tau)))
        
        conditional_gaap = -S * np.exp(-r*tau) * eta * N1 + K * np.exp(-r*tau) * N2
        
        gaap_list = np.append(gaap_list, conditional_gaap)
        
    return np.mean(gaap_list)


'''Heston Asian'''
def GAAC_heston(M, steps, kappa, v, v_bar, eta, S, K, r, delta, tau, rho):
    h = tau / steps
    gaac_list = np.array([])
    
    for i in range(M):
        dw = simulate_dw(steps)
        w = np.array([sum(dw[:i+1] * np.sqrt(h)) for i in range(steps-1)])
        w = np.append(0,w)
        v_list = simulate_v(kappa, v, v_bar, eta, h, dw)
        sigma_list = np.sqrt(v_list)
        A_list = np.array([sum(sigma_list[:i+1] * h) / tau for i in range(steps)])
        A_list = np.append(0,A_list)
        v_bar = (A_list[-1])**2 + sum((A_list[:-1])**2 * h)/tau - 2*A_list[-1]*sum((A_list[:-1]) * h)/tau
        t = np.array([i*h for i in range(steps)])
        b_G = 0.5 * v_bar + 0.5 * (r - delta) - sum((sigma_list[:-1])**2 * t * h)/(2 * tau**2)
        
        # part_1 = rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))
        part_2 = rho * sum(sigma_list[:-1] * w * h)/tau
        eta_1 = np.exp((b_G - 0.5 * rho**2 * v_bar) * tau + part_2)
        
        d_sigma = (np.log(S/K) + (b_G + 0.5*v_bar)*tau)/(np.sqrt((1-rho**2)*v_bar*tau))
        
        N1 = norm.cdf(d_sigma - (rho**2 * v_bar * tau - part_2)/(np.sqrt((1-rho**2) * v_bar * tau)))
        
        N2 = norm.cdf(d_sigma - (v_bar * tau - part_2)/(np.sqrt((1-rho**2) * v_bar * tau)))
        
        conditional_gaac = S * np.exp(-r*tau) * eta_1 * N1 - K * np.exp(-r*tau) * N2
        
        gaac_list = np.append(gaac_list, conditional_gaac)
        
    return np.mean(gaac_list)

def GAAP_heston(M, steps, kappa, v, v_bar, eta, S, K, r, delta, tau, rho):
    h = tau / steps
    gaap_list = np.array([])
    
    for i in range(M):
        dw = simulate_dw(steps)
        w = np.array([sum(dw[:i+1] * np.sqrt(h)) for i in range(steps-1)])
        w = np.append(0,w)
        v_list = simulate_v(kappa, v, v_bar, eta, h, dw)
        sigma_list = np.sqrt(v_list)
        A_list = np.array([sum(sigma_list[:i+1] * h) / tau for i in range(steps)])
        A_list = np.append(0,A_list)
        v_bar = (A_list[-1])**2 + sum((A_list[:-1])**2 * h)/tau - 2*A_list[-1]*sum((A_list[:-1]) * h)/tau
        t = np.array([i*h for i in range(steps)])
        b_G = 0.5 * v_bar + 0.5 * (r - delta) - sum((sigma_list[:-1])**2 * t * h)/(2 * tau**2)
        
        # part_1 = rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))
        part_2 = rho * sum(sigma_list[:-1] * w * h)/tau
        eta_1 = np.exp((b_G - 0.5 * rho**2 * v_bar) * tau + part_2)
        
        d_sigma = (np.log(S/K) + (b_G + 0.5*v_bar)*tau)/(np.sqrt((1-rho**2)*v_bar*tau))
        
        N1 = norm.cdf(-d_sigma + (rho**2 * v_bar * tau - part_2)/(np.sqrt((1-rho**2) * v_bar * tau)))
        
        N2 = norm.cdf(-d_sigma + (v_bar * tau - part_2)/(np.sqrt((1-rho**2) * v_bar * tau)))
        
        conditional_gaap = -S * np.exp(-r*tau) * eta_1 * N1 + K * np.exp(-r*tau) * N2
        
        gaap_list = np.append(gaap_list, conditional_gaap)
        
    return np.mean(gaap_list)


'''HW model LOOKBACK option'''
def FLSC(M, steps, kappa, sigma, sigma_bar, gamma, S, S_min, r, delta, tau, rho):
    h = tau / steps
    flsc_list = np.array([])
    
    for i in range(M):
        dw = simulate_dw(steps)
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
        N1 = norm.cdf(get_d(S, S_min, r, delta, tau, rho, h, sigma_list, dw) - rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
        N3 = norm.cdf(get_d(S, S_min, r, delta, tau, rho, h, sigma_list, dw) - np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
        N2 = 1 - N1
        N4 = norm.cdf(get_d(S_min, S, r, delta, tau, rho, h, sigma_list, dw) - np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
        eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
        
        mu = (2*(r-delta)*tau+2*rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-rho**2*v_bar*tau)/((1-rho**2)*v_bar*tau)
        
        conditional_flsc = S * np.exp(-delta*tau)*eta*(N1-N2/mu) - S_min * np.exp(-r*tau)*(N3-(S/S_min)**(1-mu)*N4/mu)
        
        flsc_list = np.append(flsc_list, conditional_flsc)
        
    return np.mean(flsc_list)

def FLSP(M, steps, kappa, sigma, sigma_bar, gamma, S, S_max, r, delta, tau, rho):
    h = tau / steps
    flsp_list = np.array([])
    
    for i in range(M):
        dw = simulate_dw(steps)
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
        N1 = norm.cdf(-get_d(S, S_max, r, delta, tau, rho, h, sigma_list, dw) + np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
        N2 = norm.cdf(-get_d(S_max, S, r, delta, tau, rho, h, sigma_list, dw) + np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
        N3 = norm.cdf(-get_d(S, S_max, r, delta, tau, rho, h, sigma_list, dw) + rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
        
        N4 = 1 - N3
        eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
        
        mu = (2*(r-delta)*tau+2*rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-rho**2*v_bar*tau)/((1-rho**2)*v_bar*tau)
        
        conditional_flsp = -S * np.exp(-delta*tau)*eta*(N3-N4/mu) + S_max * np.exp(-r*tau)*(N1-(S/S_max)**(1-mu)*N2/mu)
        
        flsp_list = np.append(flsp_list, conditional_flsp)
        
    return np.mean(flsp_list)

def FISC(M, steps, kappa, sigma, sigma_bar, gamma, S, S_max, K, r, delta, tau, rho):
    return S*np.exp(-delta*tau)-K*np.exp(-r*tau)+FLSP(M, steps, kappa, sigma, sigma_bar, gamma, S, S_max, r, delta, tau, rho)


def FISP(M, steps, kappa, sigma, sigma_bar, gamma, S, S_min, K, r, delta, tau, rho):
    return -S*np.exp(-delta*tau)+K*np.exp(-r*tau)+FLSC(M, steps, kappa, sigma, sigma_bar, gamma, S, S_min, r, delta, tau, rho)

'''Heston model LOOKBACK option'''
def FLSC_heston(M, steps, kappa, v, v_bar, eta, S, S_min, r, delta, tau, rho):
    h = tau / steps
    flsc_list = np.array([])
    
    for i in range(M):
        dw = simulate_dw(steps)
        v_list = simulate_v(kappa, v, v_bar, eta, h, dw)
        sigma_list = np.sqrt(v_list)
        v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
        N1 = norm.cdf(get_d(S, S_min, r, delta, tau, rho, h, sigma_list, dw) - rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
        N3 = norm.cdf(get_d(S, S_min, r, delta, tau, rho, h, sigma_list, dw) - np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
        N2 = 1 - N1
        N4 = norm.cdf(get_d(S_min, S, r, delta, tau, rho, h, sigma_list, dw) - np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
        eta_1 = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
        
        mu = (2*(r-delta)*tau+2*rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-rho**2*v_bar*tau)/((1-rho**2)*v_bar*tau)
        
        conditional_flsc = S * np.exp(-delta*tau)*eta_1*(N1-N2/mu) - S_min * np.exp(-r*tau)*(N3-(S/S_min)**(1-mu)*N4/mu)
        
        flsc_list = np.append(flsc_list, conditional_flsc)
        
    return np.mean(flsc_list)

def FLSP_heston(M, steps, kappa, v, v_bar, eta, S, S_max, r, delta, tau, rho):
    h = tau / steps
    flsp_list = np.array([])
    
    for i in range(M):
        dw = simulate_dw(steps)
        v_list = simulate_v(kappa, v, v_bar, eta, h, dw)
        sigma_list = np.sqrt(v_list)
        v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
        N1 = norm.cdf(-get_d(S, S_max, r, delta, tau, rho, h, sigma_list, dw) + np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
        N2 = norm.cdf(-get_d(S_max, S, r, delta, tau, rho, h, sigma_list, dw) + np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
        N3 = norm.cdf(-get_d(S, S_max, r, delta, tau, rho, h, sigma_list, dw) + rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
        
        N4 = 1 - N3
        eta_1 = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
        
        mu = (2*(r-delta)*tau+2*rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-rho**2*v_bar*tau)/((1-rho**2)*v_bar*tau)
        
        conditional_flsp = -S * np.exp(-delta*tau)*eta_1*(N3-N4/mu) + S_max * np.exp(-r*tau)*(N1-(S/S_max)**(1-mu)*N2/mu)
        
        flsp_list = np.append(flsp_list, conditional_flsp)
        
    return np.mean(flsp_list)

def FISC_heston(M, steps, kappa, v, v_bar, eta, S, S_max, K, r, delta, tau, rho):
    return S*np.exp(-delta*tau)-K*np.exp(-r*tau)+FLSP_heston(M, steps, kappa, v, v_bar, eta, S, S_max, r, delta, tau, rho)


def FISP_heston(M, steps, kappa, v, v_bar, eta, S, S_min, K, r, delta, tau, rho):
    return -S*np.exp(-delta*tau)+K*np.exp(-r*tau)+FLSC_heston(M, steps, kappa, v, v_bar, eta, S, S_min, r, delta, tau, rho)


'''Compound Option'''
'''Hull White'''
'''Call on Call'''
def N2(x,y,rho):
    return multivariate_normal.cdf([x,y],cov=[[1,rho],[rho,1]])

def conditional_CC(S, K1, K2, r, delta, T1, T2, rho, h, dw, sigma_list):
    steps1 = int(T1/h)
    v_bar_1 = sum(sigma_list[:steps1] ** 2 * h) / T1
    v_bar_2 = sum(sigma_list[:-1] ** 2 * h) / T2
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar_2*T2)
    rho1_2 = np.sqrt(v_bar_1*T1/(v_bar_2*T2))
    '''solve for sstar'''
    def C_T1(S):
        return conditional_call(S, K2, r, delta, T2-T1, rho, h, sigma_list[steps1:], dw[steps1:])-K1
    S_star = fsolve(C_T1,K1+K2)
    
    d1 = get_d(S, K2, r, delta, T2, rho, h, sigma_list, dw) - rho**2*np.sqrt(v_bar_2*T2/(1-rho**2))
    d2 = get_d(S, S_star, r, delta, T1, rho, h, sigma_list[:steps1+1], dw[:steps1]) - rho**2*np.sqrt(v_bar_1*T1/(1-rho**2))
    temp1 = S*np.exp(-delta*T2)*eta * N2(d1, d2, rho1_2)
    
    d1 = get_d(S, K2, r, delta, T2, rho, h, sigma_list, dw) - np.sqrt(v_bar_2*T2/(1-rho**2))
    d2 = get_d(S, S_star, r, delta, T1, rho, h, sigma_list[:steps1+1], dw[:steps1]) - np.sqrt(v_bar_1*T1/(1-rho**2))
    temp2 = K2*np.exp(-r*T2) * N2(d1, d2, rho1_2)
    
    d3 = get_d(S, S_star, r, delta, T1, rho, h, sigma_list[:steps1+1], dw[:steps1]) - np.sqrt(v_bar_1*T1/(1-rho**2))
    temp3 = K1*np.exp(-r*T1) * norm.cdf(d3)
    
    return (temp1 - temp2 - temp3)

def CC_HW(M, steps, kappa, sigma, sigma_bar, gamma, S, K1, K2, r, delta, T1, T2, rho):
    h = T2 / steps
    c_list = np.array([])
    for i in range(M):
        dw = simulate_dw(steps)
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        c = conditional_CC(S, K1, K2, r, delta, T1, T2, rho, h, dw, sigma_list)
        c_list = np.append(c_list, c)
    return np.mean(c_list)

def conditional_PC(S, K1, K2, r, delta, T1, T2, rho, h, dw, sigma_list):
    steps1 = int(T1/h)
    v_bar_1 = sum(sigma_list[:steps1] ** 2 * h) / T1
    v_bar_2 = sum(sigma_list[:-1] ** 2 * h) / T2
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar_2*T2)
    rho1_2 = np.sqrt(v_bar_1*T1/(v_bar_2*T2))
    '''solve for sstar'''
    def C_T1(S):
        return conditional_call(S, K2, r, delta, T2-T1, rho, h, sigma_list[steps1:], dw[steps1:])-K1
    S_star = fsolve(C_T1,K1+K2)
    
    
    d1 = -get_d(S, S_star, r, delta, T1, rho, h, sigma_list[:steps1+1], dw[:steps1]) + rho**2*np.sqrt(v_bar_1*T1/(1-rho**2))
    d2 = get_d(S, K2, r, delta, T2, rho, h, sigma_list, dw) - rho**2*np.sqrt(v_bar_2*T2/(1-rho**2))
    temp3 = - S*np.exp(-delta*T2)*eta * N2(d1, d2, -rho1_2)
    
    d1 = -get_d(S, S_star, r, delta, T1, rho, h, sigma_list[:steps1+1], dw[:steps1]) + np.sqrt(v_bar_1*T1/(1-rho**2))
    d2 = get_d(S, K2, r, delta, T2, rho, h, sigma_list, dw) - np.sqrt(v_bar_2*T2/(1-rho**2))
    temp2 = K2*np.exp(-r*T2) * N2(d1, d2, -rho1_2)
    
    d3 = - get_d(S, S_star, r, delta, T1, rho, h, sigma_list[:steps1+1], dw[:steps1]) + np.sqrt(v_bar_1*T1/(1-rho**2))
    temp1 = K1*np.exp(-r*T1) * norm.cdf(d3)
    
    return (temp1 + temp2 + temp3)

def PC_HW(M, steps, kappa, sigma, sigma_bar, gamma, S, K1, K2, r, delta, T1, T2, rho):
    h = T2 / steps
    c_list = np.array([])
    for i in range(M):
        dw = simulate_dw(steps)
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        c = conditional_PC(S, K1, K2, r, delta, T1, T2, rho, h, dw, sigma_list)
        c_list = np.append(c_list, c)
    return np.mean(c_list)

def conditional_CP(S, K1, K2, r, delta, T1, T2, rho, h, dw, sigma_list):
    steps1 = int(T1/h)
    v_bar_1 = sum(sigma_list[:steps1] ** 2 * h) / T1
    v_bar_2 = sum(sigma_list[:-1] ** 2 * h) / T2
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar_2*T2)
    rho1_2 = np.sqrt(v_bar_1*T1/(v_bar_2*T2))
    '''solve for sstar'''
    def C_T1(S):
        return conditional_put(S, K2, r, delta, T2-T1, rho, h, sigma_list[steps1:], dw[steps1:])-K1
    S_star = fsolve(C_T1,K1+K2)
    
    d1 = -get_d(S, S_star, r, delta, T1, rho, h, sigma_list[:steps1+1], dw[:steps1]) + rho**2*np.sqrt(v_bar_1*T1/(1-rho**2))
    d2 = -get_d(S, K2, r, delta, T2, rho, h, sigma_list, dw) + rho**2*np.sqrt(v_bar_2*T2/(1-rho**2))
    temp2 = - S*np.exp(-delta*T2)*eta * N2(d1, d2, rho1_2)
    
    d1 = -get_d(S, S_star, r, delta, T1, rho, h, sigma_list[:steps1+1], dw[:steps1]) + np.sqrt(v_bar_1*T1/(1-rho**2))
    d2 = -get_d(S, K2, r, delta, T2, rho, h, sigma_list, dw) + np.sqrt(v_bar_2*T2/(1-rho**2))
    temp1 = K2*np.exp(-r*T2) * N2(d1, d2, rho1_2)
    
    d3 = - get_d(S, S_star, r, delta, T1, rho, h, sigma_list[:steps1+1], dw[:steps1]) + np.sqrt(v_bar_1*T1/(1-rho**2))
    temp3 = - K1*np.exp(-r*T1) * norm.cdf(d3)
    
    return (temp1 + temp2 + temp3)

def CP_HW(M, steps, kappa, sigma, sigma_bar, gamma, S, K1, K2, r, delta, T1, T2, rho):
    h = T2 / steps
    c_list = np.array([])
    for i in range(M):
        dw = simulate_dw(steps)
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        c = conditional_CP(S, K1, K2, r, delta, T1, T2, rho, h, dw, sigma_list)
        c_list = np.append(c_list, c)
    return np.mean(c_list)

def conditional_PP(S, K1, K2, r, delta, T1, T2, rho, h, dw, sigma_list):
    steps1 = int(T1/h)
    v_bar_1 = sum(sigma_list[:steps1] ** 2 * h) / T1
    v_bar_2 = sum(sigma_list[:-1] ** 2 * h) / T2
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar_2*T2)
    rho1_2 = np.sqrt(v_bar_1*T1/(v_bar_2*T2))
    '''solve for sstar'''
    def C_T1(S):
        return conditional_put(S, K2, r, delta, T2-T1, rho, h, sigma_list[steps1:], dw[steps1:])-K1
    S_star = fsolve(C_T1,K1+K2)
    
    
    d1 = get_d(S, S_star, r, delta, T1, rho, h, sigma_list[:steps1+1], dw[:steps1]) - rho**2*np.sqrt(v_bar_1*T1/(1-rho**2))
    d2 = -get_d(S, K2, r, delta, T2, rho, h, sigma_list, dw) + rho**2*np.sqrt(v_bar_2*T2/(1-rho**2))
    temp3 = S*np.exp(-delta*T2)*eta * N2(d1, d2, -rho1_2)
    
    d1 = get_d(S, S_star, r, delta, T1, rho, h, sigma_list[:steps1+1], dw[:steps1]) - np.sqrt(v_bar_1*T1/(1-rho**2))
    d2 = -get_d(S, K2, r, delta, T2, rho, h, sigma_list, dw) + np.sqrt(v_bar_2*T2/(1-rho**2))
    temp2 = - K2*np.exp(-r*T2) * N2(d1, d2, -rho1_2)
    
    d3 = get_d(S, S_star, r, delta, T1, rho, h, sigma_list[:steps1+1], dw[:steps1]) - np.sqrt(v_bar_1*T1/(1-rho**2))
    temp1 = K1*np.exp(-r*T1) * norm.cdf(d3)
    
    return (temp1 + temp2 + temp3)

def PP_HW(M, steps, kappa, sigma, sigma_bar, gamma, S, K1, K2, r, delta, T1, T2, rho):
    h = T2 / steps
    c_list = np.array([])
    for i in range(M):
        dw = simulate_dw(steps)
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        c = conditional_PP(S, K1, K2, r, delta, T1, T2, rho, h, dw, sigma_list)
        c_list = np.append(c_list, c)
    return np.mean(c_list)


'''Chooser'''
def conditional_CH(S, K, r, delta, T1, T2, rho, h, dw, sigma_list):
    steps1 = int(T1/h)
    v_bar_1 = sum(sigma_list[:steps1] ** 2 * h) / T1
    v_bar_2 = sum(sigma_list[:-1] ** 2 * h) / T2
    eta1 = np.exp(rho*(sum(sigma_list[:steps1]*dw[:steps1]*np.sqrt(h)))-0.5*rho**2*v_bar_1*T1)
    eta2 = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar_2*T2)
    
    d1 = get_d(S, K*np.exp(-(r-delta)*T1), r, delta, T1, rho, h, sigma_list[:steps1+1], dw[:steps1])
    temp1 = K*np.exp(-r*T2)*norm.cdf(-d1 + np.sqrt(v_bar_1*T1/(1-rho**2)) )
    temp2 = -S*np.exp(-delta*T2)*eta1*norm.cdf(-d1 + rho**2*np.sqrt(v_bar_1*T1/(1-rho**2) ))
    #print(temp1+temp2)
    #print(np.exp(-delta*(T2-T1))*conditional_put(S, K*np.exp(-(r-delta)*(T2-T1)), r, delta, T1, rho, h, sigma_list[:steps1+1], dw[:steps1]))
    d2 = get_d(S, K, r, delta, T2, rho, h, sigma_list, dw)
    temp3 = S*np.exp(-delta*T2)*eta2*norm.cdf(d2 - rho**2*np.sqrt(v_bar_2*T2/(1-rho**2) ))
    temp4 = -K*np.exp(-r*T2)*norm.cdf(d2 - np.sqrt(v_bar_2*T2/(1-rho**2) ))
    #print(temp3+temp4)
    #print(conditional_call(S, K, r, delta, T2, rho, h, sigma_list, dw))
    return temp1 + temp2 + temp3 + temp4

def CH_HW(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, T1, T2, rho):
    h = T2 / steps
    c_list = np.array([])
    for i in range(M):
        dw = simulate_dw(steps)
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        c = conditional_CH(S, K, r, delta, T1, T2, rho, h, dw, sigma_list)
        c_list = np.append(c_list, c)
    return np.mean(c_list)

def conditional_CH_General(S, Kc, Kp, r, delta, T1, Tc, Tp, rho, h, dw, sigma_list):
    steps1 = int(T1/h)
    stepsc = int(Tc/h)
    stepsp = int(Tp/h)
    v_bar_1 = sum(sigma_list[:steps1] ** 2 * h) / T1
    v_bar_c = sum(sigma_list[:stepsc] ** 2 * h) / Tc
    v_bar_p = sum(sigma_list[:stepsp] ** 2 * h) / Tp
    etac = np.exp(rho*(sum(sigma_list[:stepsc]*dw[:stepsc]*np.sqrt(h)))-0.5*rho**2*v_bar_c*Tc)
    etap = np.exp(rho*(sum(sigma_list[:stepsp]*dw[:stepsc]*np.sqrt(h)))-0.5*rho**2*v_bar_p*Tp)
   
    rhoc = np.sqrt(v_bar_1*T1/(v_bar_c*Tc))
    rhop = np.sqrt(v_bar_1*T1/(v_bar_p*Tp))
    '''solve for SSTAR'''
    def C_P_T1(S):
        return conditional_call(S, Kc, r, delta, Tc-T1, rho, h, sigma_list[steps1:stepsc+1], dw[steps1:stepsc])-conditional_put(S, Kp, r, delta, Tp-T1, rho, h, sigma_list[steps1:stepsp+1], dw[steps1:stepsp])
    S_star = fsolve(C_P_T1,Kc)
    
    d1 = get_d(S,S_star, r, delta, T1, rho, h, sigma_list[:steps1+1], dw[:steps1])
    dc = get_d(S, Kc, r, delta, Tc, rho, h, sigma_list[:stepsc+1], dw[:stepsc])
    dp = get_d(S, Kp, r, delta, Tp, rho, h, sigma_list[:stepsp+1], dw[:stepsp])
    
    temp1 = S*np.exp(-delta*Tc)*etac*N2(d1 - rho**2*np.sqrt(v_bar_1*T1/(1-rho**2)),
        dc - rho**2*np.sqrt(v_bar_c*Tc/(1-rho**2)),
        rhoc)
    temp2 = - Kc*np.exp(-r*Tc)*N2(d1 - np.sqrt(v_bar_1*T1/(1-rho**2)),
        dc - np.sqrt(v_bar_c*Tc/(1-rho**2)),
        rhoc)
    temp3 =  Kp*np.exp(-r*Tp)*N2(-d1 + np.sqrt(v_bar_1*T1/(1-rho**2)),
        -dp + np.sqrt(v_bar_p*Tp/(1-rho**2)),
        rhop)
    temp4 = - S*np.exp(-delta*Tp)*etap*N2(-d1 + rho**2*np.sqrt(v_bar_1*T1/(1-rho**2)),
        -dp - rho**2*np.sqrt(v_bar_p*Tp/(1-rho**2)),
        rhop)
    return temp1 + temp2 + temp3 + temp4

def CH_General_HW(M, steps, kappa, sigma, sigma_bar, gamma, S, Kc, Kp, r, delta, T1, Tc, Tp, rho):
    h = max(Tc, Tp) / steps
    c_list = np.array([])
    for i in range(M):
        dw = simulate_dw(steps)
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        c = conditional_CH_General(S, Kc, Kp, r, delta, T1, Tc, Tp, rho, h, dw, sigma_list)
        c_list = np.append(c_list, c)
    return np.mean(c_list)

'''Binary Call'''
def conditional_AON(S, K, r, delta, T, rho, h, dw, sigma_list):
    v_bar = sum(sigma_list[:-1] ** 2 * h) / T
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*T)
    return S*np.exp(-delta*T)*eta*norm.cdf(
            get_d(S, K, r, delta, T, rho, h, sigma_list, dw)
            -rho**2*np.sqrt(v_bar*T/(1-rho**2))
            )
    
def AON_HW(M, steps, kappa, sigma, sigma_bar, gamma, S, K, r, delta, tau, rho):
    h = tau / steps
    c_list = np.array([])
    for i in range(M):
        dw = simulate_dw(steps)
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        c = conditional_AON(S, K, r, delta, tau, rho, h, dw, sigma_list)
        c_list = np.append(c_list, c)
    return np.mean(c_list)


def conditional_CON(S, K, Q, r, delta, T, rho, h, dw, sigma_list):
    v_bar = sum(sigma_list[:-1] ** 2 * h) / T
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*T)
    return Q*np.exp(-r*T)*eta*norm.cdf(
            get_d(S, K, r, delta, T, rho, h, sigma_list, dw)
            -np.sqrt(v_bar*T/(1-rho**2))
            )
    
def CON_HW(M, steps, kappa, sigma, sigma_bar, gamma, S, K, Q, r, delta, tau, rho):
    h = tau / steps
    c_list = np.array([])
    for i in range(M):
        dw = simulate_dw(steps)
        sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        c = conditional_CON(S, K, Q, r, delta, tau, rho, h, dw, sigma_list)
        c_list = np.append(c_list, c)
    return np.mean(c_list)

def european_call(S, K, r, delta, tau, rho, h, sigma_list, dw):
    v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
    N1 = norm.cdf(get_d(S, K, r, delta, tau, rho, h, sigma_list, dw) - rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
    N2 = norm.cdf(get_d(S, K, r, delta, tau, rho, h, sigma_list, dw) - np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
    c = S * np.exp(- delta * tau) * eta * N1 - K * np.exp(-r * tau) * N2
    return c

def european_put(S, K, r, delta, tau, rho, h, sigma_list, dw):
    v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
    N1 = norm.cdf(-get_d(S, K, r, delta, tau, rho, h, sigma_list, dw) + rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
    N2 = norm.cdf(-get_d(S, K, r, delta, tau, rho, h, sigma_list, dw) + np.sqrt(v_bar*tau)/np.sqrt(1-rho**2))
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
    p = -S * np.exp(- delta * tau) * eta * N1 + K * np.exp(-r * tau) * N2
    return p

'''Barrier Option'''

def up_in_call(S, K, H,r, delta, tau, rho, h, sigma_list, dw):
    v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
    alpha = ((r - delta + 0.5*v_bar)*tau + rho*(sum(sigma_list[:-1]*dw*np.sqrt(h))) - rho**2*v_bar*tau)/((1-rho**2)*v_bar*tau)
    
    d1 = get_d(S, H, r, delta, tau, rho, h, sigma_list, dw) -  rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    d2 = get_d(S, H, r, delta, tau, rho, h, sigma_list, dw) - np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    d3 = -get_d(H**2/S,K, r, delta, tau, rho, h, sigma_list, dw) +  rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    d4 = -get_d(H,S, r, delta, tau, rho, h, sigma_list, dw) + rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    d5 = -get_d(H**2/S,K, r, delta, tau, rho, h, sigma_list, dw) + np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    d6 = -get_d(H,S, r, delta, tau, rho, h, sigma_list, dw) + np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    
    n1 = norm.cdf(d1)
    n2 = norm.cdf(d2)
    n3 = norm.cdf(d3)
    n4 = norm.cdf(d4)
    n5 = norm.cdf(d5)
    n6 = norm.cdf(d6)
    
    df_d = np.exp(- delta * tau)
    df_r = np.exp(- r * tau)
    
    c = S* df_d*eta*n1 - K*df_r*n2 - S*df_d*(S/H)**(-2*alpha)*eta*(n3-n4) + K*df_r*(S/H)**(2-2*alpha)*(n5-n6)
    
    return c

def up_out_call(S, K, H,r, delta, tau, rho, h, sigma_list, dw):
    
    return european_call(S, K, r, delta, tau, rho, h, sigma_list, dw)- up_in_call(S, K, H,r, delta, tau, rho, h, sigma_list, dw)

def down_in_call(S, K, H,r, delta, tau, rho, h, sigma_list, dw):
    
    v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
    alpha = ((r - delta + 0.5*v_bar)*tau + rho*(sum(sigma_list[:-1]*dw*np.sqrt(h))) - rho**2*v_bar*tau)/((1-rho**2)*v_bar*tau)
    
    d1 = get_d(H**2/S,K, r, delta, tau, rho, h, sigma_list, dw) - rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    d2 = get_d(H**2/S,K, r, delta, tau, rho, h, sigma_list, dw) - np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    n1 = norm.cdf(d1)
    n2 = norm.cdf(d2)
    
    df_d = np.exp(- delta * tau)
    df_r = np.exp(- r * tau)
    
    c = S*df_d*(S/H)**(-2*alpha)*eta*n1 - K*df_r*(S/H)**(2-2*alpha)*n2
    
    return c

def down_out_call(S, K, H,r, delta, tau, rho, h, sigma_list, dw):
    
    return european_call(S, K, r, delta, tau, rho, h, sigma_list, dw)- down_in_call(S, K, H,r, delta, tau, rho, h, sigma_list, dw)



def up_in_put(S, K, H,r, delta, tau, rho, h, sigma_list, dw):
    c = up_in_call(S, K, H,r, delta, tau, rho, h, sigma_list, dw)
    v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
    alpha = ((r - delta + 0.5*v_bar)*tau + rho*(sum(sigma_list[:-1]*dw*np.sqrt(h))) - rho**2*v_bar*tau)/((1-rho**2)*v_bar*tau)
    
    d1 = get_d(S, H, r, delta, tau, rho, h, sigma_list, dw) - np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    d2 = -get_d(H,S, r, delta, tau, rho, h, sigma_list, dw) + np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    d3 = get_d(S, H, r, delta, tau, rho, h, sigma_list, dw) -  rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    d4 = -get_d(H, S, r, delta, tau, rho, h, sigma_list, dw) +  rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)

    n1 = norm.cdf(d1)
    n2 = norm.cdf(d2)
    n3 = norm.cdf(d3)
    n4 = norm.cdf(d4)
    
    df_d = np.exp(- delta * tau)
    df_r = np.exp(- r * tau)
    
    p = c + K*df_r*n1 + (S/H)**(2-2*alpha)*n2 - S*df_d*eta*(n3 + (S/H)**(-2*alpha)*n4)

    return p


def up_out_put(S, K, H,r, delta, tau, rho, h, sigma_list, dw):
    
    return european_put(S, K, r, delta, tau, rho, h, sigma_list, dw) - up_in_put(S, K, H,r, delta, tau, rho, h, sigma_list, dw)


def down_in_put(S, K, H,r, delta, tau, rho, h, sigma_list, dw):
    c = down_in_call(S, K, H,r, delta, tau, rho, h, sigma_list, dw)
    v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
    alpha = ((r - delta + 0.5*v_bar)*tau + rho*(sum(sigma_list[:-1]*dw*np.sqrt(h))) - rho**2*v_bar*tau)/((1-rho**2)*v_bar*tau)
    
    d1 = -get_d(S, H, r, delta, tau, rho, h, sigma_list, dw) + np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    d2 = get_d(H,S, r, delta, tau, rho, h, sigma_list, dw) - np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    d3 = -get_d(S, H, r, delta, tau, rho, h, sigma_list, dw) +  rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    d4 = get_d(H, S, r, delta, tau, rho, h, sigma_list, dw) - rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    
    n1 = norm.cdf(d1)
    n2 = norm.cdf(d2)
    n3 = norm.cdf(d3)
    n4 = norm.cdf(d4)
    
    df_d = np.exp(- delta * tau)
    df_r = np.exp(- r * tau)
    
    p = c + K*df_r*n1 + (S/H)**(2-2*alpha)*n2 - S*df_d*eta*(n3 + (S/H)**(-2*alpha)*n4)
    
    return p

def down_out_put(S, K, H,r, delta, tau, rho, h, sigma_list, dw):
    
    return european_put(S, K, r, delta, tau, rho, h, sigma_list, dw) - down_in_put(S, K, H,r, delta, tau, rho, h, sigma_list, dw)
    

'''Shout Options'''
    
def shout_call(S, K,  S_sh,r, delta, tau, rho, h, sigma_list, dw):
    
    v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
    
    d1 = -get_d(S,S_sh, r, delta, tau, rho, h, sigma_list, dw) + np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    d2 = get_d(S, S_sh, r, delta, tau, rho, h, sigma_list, dw) -  rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    d3 = -d1
    
    n1 = norm.cdf(d1)
    n2 = norm.cdf(d2)
    n3 = norm.cdf(d3)
    
    df_d = np.exp(- delta * tau)
    df_r = np.exp(- r * tau)
    
    c = df_r*(S_sh - K)*n1+ S* df_d*eta*n2 - K*df_r*n3
    
    return c
    
def shout_put(S, K, S_sh, r, delta, tau, rho, h, sigma_list, dw):
    
    v_bar = sum(sigma_list[:-1] ** 2 * h) / tau
    eta = np.exp(rho*(sum(sigma_list[:-1]*dw*np.sqrt(h)))-0.5*rho**2*v_bar*tau)
    
    d1 = -get_d(S,S_sh, r, delta, tau, rho, h, sigma_list, dw) + np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    d2 = -get_d(S, S_sh, r, delta, tau, rho, h, sigma_list, dw) + rho**2*np.sqrt(v_bar*tau)/np.sqrt(1-rho**2)
    d3 = -d1
    
    n1 = norm.cdf(d1)
    n2 = norm.cdf(d2)
    n3 = norm.cdf(d3)
    
    df_d = np.exp(- delta * tau)
    df_r = np.exp(- r * tau)
    
    p = K *df_r*n1 - S* df_d*eta*n2 +(K-S_sh) * n3
    
    return p 
    
def monte_carlo_sim(M, steps, kappa, sigma,sigma_bar,vol, v_bar,eta, gamma, S, K,H,S_sh, r, delta, tau, rho, typ = '', cp = '', model = ''):
    h = tau / steps
    lst = []  
    
    for i in range(M):
        dw = simulate_dw(steps)
        if model == 'HW':
            sigma_list = simulate_sigma(kappa, sigma, sigma_bar, gamma, h, dw)
        else:
            v_list = simulate_v(kappa, vol, v_bar, eta, h, dw)
            sigma_list = np.sqrt(v_list)
        
        if typ == 'up_in' and cp == 'call':
            v = up_in_call(S, K, H,r, delta, tau, rho, h, sigma_list, dw)
            lst.append(v)
        if typ == 'down_in' and cp == 'call':
            v = down_in_call(S, K, H,r, delta, tau, rho, h, sigma_list, dw)
            lst.append(v)
        if typ == 'up_out' and cp == 'call':
            v = up_out_call(S, K, H,r, delta, tau, rho, h, sigma_list, dw)
            lst.append(v)
        
        if typ == 'down_out' and cp == 'call':
            v = down_out_call(S, K, H,r, delta, tau, rho, h, sigma_list, dw)
            lst.append(v)
            
        if typ == 'up_in' and cp == 'put':
            v = up_in_put(S, K, H,r, delta, tau, rho, h, sigma_list, dw)
            lst.append(v)
        
        if typ == 'up_out' and cp == 'put':
            v = up_out_put(S, K, H,r, delta, tau, rho, h, sigma_list, dw)
            lst.append(v)
            
        if typ == 'down_in' and cp == 'put':
            v = down_in_put(S, K, H,r, delta, tau, rho, h, sigma_list, dw)
            lst.append(v)
            
        if typ == 'down_out' and cp == 'put':
            v = down_out_put(S, K, H,r, delta, tau, rho, h, sigma_list, dw)
            lst.append(v)
        
        if typ == 'shout' and cp == 'call':
            v = shout_call(S, K, S_sh,r, delta, tau, rho, h, sigma_list, dw)
            lst.append(v)
        if typ == 'shout' and cp == 'put':
            v = shout_put(S, K, S_sh,r, delta, tau, rho, h, sigma_list, dw)
            lst.append(v)
            
    return np.mean(lst)

















