#!/usr/bin/env python2

import os
import glob
import json
import pymworks
import re
import datautils
import copy
import math
import time
import optparse
import sys
import pprint

import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
import cPickle as pkl
import scipy.stats as spstats

import utils as util
import assign_phase as ph
import process_datafiles as procd

pp = pprint.PrettyPrinter(indent=4)

import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


def update_fonts(labelsize=24): #big=True):

    # Font params
#     font_params = {'legend.fontsize': 20, #'medium',
#              'axes.labelsize': 24, #'x-large',
#              'axes.titlesize': 24, #'x-large',
#              'xtick.labelsize': 20, #'x-large',
#              'ytick.labelsize': 20} #'x-large'}

    font_params = {'legend.fontsize': 12, #'medium',
             'axes.labelsize': labelsize, #'x-large',
             'axes.titlesize': labelsize, #'x-large',
             'xtick.labelsize': labelsize-4, #'x-large',
             'ytick.labelsize': labelsize-4} #'x-large'}

    # font_params = {'legend.fontsize': 'medium',
    #          'axes.labelsize': 'medium',
    #          'axes.titlesize': 'medium',
    #          'xtick.labelsize': 'medium',
    #          'ytick.labelsize': 'medium'}
    pl.rcParams.update(font_params)


def get_pnas_cmap():
    # combine two color maps for 0-50% and 50-100% as in the PNAS paper
    colors1 = pl.cm.bone(np.linspace(0.95,0.0,128))
    colors2 = pl.cm.hot(np.linspace(0.,0.95,128))
    colors = np.vstack((colors1, colors2))
    pnas_cmap = mcolors.LinearSegmentedColormap.from_list('pnas_map', colors)
    
    return pnas_cmap

def outline_boxplot(ax):
    # iterate over boxes to make them outlines only
    for i,box in enumerate(ax.artists):
        box.set_edgecolor('black')
        box.set_facecolor('white')
        # iterate over whiskers and median lines
        for j in range(6*i,6*(i+1)):
             ax.lines[j].set_color('black')
    return


from matplotlib.patches import Rectangle

def draw_no_feedback(ax, curr_no_fb, defaults, seaborn=True, lw=2):

    default_size = defaults['size']
    default_depth_rotation = defaults['depth_rotation']
    default_planar_rotation = defaults['planar_rotation']

    expected_sizes = defaults['expected_sizes']
    expected_drots = defaults['expected_depth_rotations']

    offset = 0 if seaborn else 0.5

    # Draw default box
    default_size_ix = list(expected_sizes).index(default_size)
    default_drot_ix = list(expected_drots).index(default_depth_rotation)
    ax.add_patch(Rectangle((default_drot_ix-offset, default_size_ix-offset), 1, 1, 
                           fill=False, edgecolor='forestgreen', lw=lw, label='default'))
    
    if len(curr_no_fb) > 0:
        # Draw no feedback box
        nofb_size_min = min([f[0] for f in curr_no_fb])
        nofb_size_max = max([f[0] for f in curr_no_fb])
        nofb_drot_min = min([f[1] for f in curr_no_fb])
        nofb_drot_max = max([f[1] for f in curr_no_fb])
        #print("size - no feedback: [%i, %i]" % (nofb_size_min, nofb_size_max))
        #print("dept rot - no feedback: [%i, %i]" % (nofb_drot_min, nofb_drot_max))
        fb_sz_ixs = (list(expected_sizes).index(nofb_size_min), list(expected_sizes).index(nofb_size_max))
        fb_drot_ixs = (list(expected_drots).index(nofb_drot_min), list(expected_drots).index(nofb_drot_max))

        n_fb_drot = fb_drot_ixs[1]-fb_drot_ixs[0]+1
        n_fb_sz = fb_sz_ixs[1]-fb_sz_ixs[0]+1

        ax.add_patch(Rectangle((fb_drot_ixs[0]-offset, fb_sz_ixs[0]-offset), n_fb_drot, n_fb_sz, 
                               fill=False, edgecolor='cornflowerblue', lw=lw, linestyle=':', label='no feedback'))

    return ax

def format_size_depth_ticks(ax, xvals=[], yvals=[], 
                            xmax=None, ymax=None,
                            minimal=True, seaborn=True):
    
    offset = 0.5 if seaborn else 0.
    
    if minimal:
        xmax = max(xvals) if xmax is None else xmax
        xtick_labels = [int(x) for x in xvals if x%xmax==0 ]
        xticks = [i+offset for i, x in enumerate(xvals) if x%xmax==0 ]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)

        ymax = max(yvals) if ymax is None else ymax
        ytick_labels = [int(x) for x in yvals if x%ymax==0 ]
        yticks = [i+offset for i, x in enumerate(yvals) if x%ymax==0 ]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, rotation=0)
    else:
        ax.set_xticks([i+offset for i in np.arange(0, len(xvals))])
        ax.set_xticklabels(xvals)

        ax.set_yticks([i+offset for i in np.arange(0, len(yvals))])
        ax.set_yticklabels(yvals)
    
    return ax