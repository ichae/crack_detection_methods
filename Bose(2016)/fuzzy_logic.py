# -*- coding: utf-8 -*-
"""
Tipping problem
Created on Wed Sep 12 16:18:02 2018

@author: Cho
"""
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt 

def fuzzy_model(param, graph = False):
    Area, AxesRatio, AreaRatio = param
    #%% Generate univerce variable 
    x_Area = np.arange(0, 20001, 1)
    x_AxesRatio = np.arange(0, 201, 0.1)
    x_AreaRatio = np.arange(0, 501, 1)
    x_out  = np.arange(0, 1, 0.01)

    #%% Generate Fuzzy membership functions
    Area_lo = fuzz.trapmf(x_Area, [0, 0, 200, 400])
    Area_md = fuzz.trapmf(x_Area, [200, 400, 600, 2000])
    Area_hi = fuzz.trapmf(x_Area, [600, 20000, 20000, 20000])

    AxesRatio_lo = fuzz.trapmf(x_AxesRatio, [0, 0, 4.5, 5.5])
    AxesRatio_hi = fuzz.trapmf(x_AxesRatio, [4.5, 5.5, 200, 200])
    
    AreaRatio_lo = fuzz.trapmf(x_AreaRatio, [0, 0, 5, 6])
    AreaRatio_hi = fuzz.trapmf(x_AreaRatio, [5, 6, 500, 500])
    
    out_lo = fuzz.trapmf(x_out, [0, 0, 0.45, 0.55])
    out_hi = fuzz.trapmf(x_out, [0.45, 0.55, 1, 1])

    #%% We need the activation of our fuzzy membership functions at these values.
    Area_level_lo = fuzz.interp_membership(x_Area, Area_lo, Area)
    Area_level_md = fuzz.interp_membership(x_Area, Area_md, Area)
    Area_level_hi = fuzz.interp_membership(x_Area, Area_hi, Area)
    
    AxesRatio_level_lo = fuzz.interp_membership(x_AxesRatio, AxesRatio_lo, AxesRatio)
    AxesRatio_level_hi = fuzz.interp_membership(x_AxesRatio, AxesRatio_hi, AxesRatio)
    
    AreaRatio_level_lo = fuzz.interp_membership(x_AreaRatio, AreaRatio_lo, AreaRatio)
    AreaRatio_level_hi = fuzz.interp_membership(x_AreaRatio, AreaRatio_hi, AreaRatio)

    #%% Activation Rule
    # And is min
    active_rule1 = np.fmin(Area_level_hi, AxesRatio_level_hi)
    active_rule2 = np.fmin(Area_level_hi, np.fmin(AxesRatio_level_lo, AreaRatio_level_hi))
    active_rule3 = np.fmin(Area_level_md, AxesRatio_level_hi)
    active_rule4 = np.fmin(Area_level_md, np.fmin(AxesRatio_level_lo, AreaRatio_level_hi))
    active_rule5 = np.fmin(Area_level_lo, AxesRatio_level_hi) # noise

    # Now we apply this by clipping the top off the corresponding output
    out_activation_lo = np.fmin(active_rule5, out_lo)
    out_activation_hi = np.fmin(np.max([active_rule1, active_rule2, active_rule3, active_rule4]), out_hi)
    
    out0 = np.zeros_like(x_out)

    # Visualize this
    if graph:
        fig, ax0 = plt.subplots(figsize=(8, 3))
        
        ax0.fill_between(x_out, out0, out_activation_lo, facecolor='b', alpha=0.7)
        ax0.plot(x_out, out_lo, 'b', linewidth=0.5, linestyle='--', )
        ax0.fill_between(x_out, out0, out_activation_hi, facecolor='r', alpha=0.7)
        ax0.plot(x_out, out_hi, 'r', linewidth=0.5, linestyle='--')
        ax0.set_title('Output membership activity')
        
        # Turn off top/right axes
        for ax in (ax0,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
        
        plt.tight_layout()

    # Aggregate all three output membership functions together
    aggregated = np.fmax(out_activation_lo, out_activation_hi)
    if np.sum(aggregated) > 0:
        out = fuzz.defuzz(x_out, aggregated, 'centroid')
    else:
        out = 0

    return out

