#!/usr/bin/env python
# simple data analysis attempt for within-session
# Nov 24, 2015 - jyr
# Go/NoGo task, new protocol with optimizers.

# un-crap-ify the crap code from 2013...

# For "logging" mode during debugging/testing new code:
# (These need to be done BEFORE importing pymworks, etc.)

# In [1]: import logging
# In [2]: logging.basicConfig(level=logging.DEBUG)
# In [3]: logging.debug('foo')
# DEBUG:root:foo


import os
import sys

import pymworks
import cPickle as pkl

import itertools
import functools

import logging
import datautils
import numpy as np

import re
import matplotlib.pyplot as plt

###############################################################################
# GENERALLY USEFUL FUNCTIONS
###############################################################################

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def save_dict(var, savepath, fname):
    '''
    Check if dict exists. Append and save, if so. Otherwise, just save.

    var : dict
        dict to save

    savepath : string
        path to save dict (or to look for existent dict)

    fname: string
        name of .pkl dict
    '''
    fn = os.path.join(savepath, fname)
    if not os.path.exists(fn):
        P = var
    else:
        f = open(fn,'rb')
        P = pkl.load(f)
        P.update(var)
        f.close()

    f = open(fn, 'wb')
    pkl.dump(P, f)
    f.close()


###############################################################################
# SESSION ANALYSIS FUNCTIONS:
###############################################################################

def get_outcome_tallies(datadir):

    animals = os.listdir(datadir)
    processed_path = os.path.join(os.path.split(datadir)[0], 'processed')

    outcomes = dict()
    for animal in animals:
        outfiles = [i for i in os.listdir(processed_path) if animal in i]
        fn_trials = open([os.path.join(processed_path, i) for i in outfiles if '_trials' in i][0], 'rb')
        trials = pkl.load(fn_trials)
        fn_trials.close()

        outcomes[animal] = dict()

        for s in trials.keys():
            
            outcomes[animal][s] = dict()

            try:
                if len(trials[s]) > 1:
                    trials[s] = list(itertools.chain.from_iterable(trials[s]))
                else:
                    trials[s] = trials[s][0]

            except IndexError as e:
                print e
                print animal, s, trials[s]
                

            outcomes[animal][s]['hits'] = len([i for i in trials[s] if i['outcome']=='session_correct_lick'])
            outcomes[animal][s]['misses'] = len([i for i in trials[s] if i['outcome']=='session_bad_ignore'])
            outcomes[animal][s]['fas'] = len([i for i in trials[s] if i['outcome']=='session_bad_lick'])
            outcomes[animal][s]['crs'] = len([i for i in trials[s] if i['outcome']=='session_correct_ignore'])

    return outcomes



def get_phase_info(datadir):

    animals = os.listdir(datadir)
    processed_path = os.path.join(os.path.split(datadir)[0], 'processed')
    print processed_path

    phase_info = dict()
    for animal in animals:
        outfiles = [i for i in os.listdir(processed_path) if animal in i]
        fn_summ = open([os.path.join(processed_path, i) for i in outfiles if '_summary' in i][0], 'rb')
        summary = pkl.load(fn_summ)
        fn_summ.close()

        phase_info[animal] = dict()

        for s in summary.keys():
            phase_info[animal][s] = dict()
            # if len(trials[s]) > 1:
            #     summary[s] = list(itertools.chain.from_iterable(summary[s]))
            # else:
            #     summary[s] = summary[s][0]

            phase_info[animal][s]['phases_completed'] = summary[s]['phases_completed']
            phase_info[animal][s]['targetprob'] = summary[s]['targetprob']
            phase_info[animal][s]['targetprob_lower'] = summary[s]['targetprob_lower']
            phase_info[animal][s]['engagement'] = summary[s]['engagement']
            phase_info[animal][s]['curr_contrast'] = summary[s]['curr_contrast']

    return phase_info


###############################################################################
# PLOTTING FUNCTIONS:
###############################################################################

def plot_progress(datadir):

    figpath = os.path.join(os.path.split(datadir)[0], 'figures')
    if not os.path.exists(figpath):
        os.makedirs(figpath)

    session_info = get_phase_info(datadir)
    outcome_info = get_outcome_tallies(datadir)

    animals = session_info.keys()

    for animal in animals:

        total_trials = [float(sum([outcome_info[animal][s][k] for k in outcome_info[animal][s].keys()])) for s in sorted(outcome_info[animal], key=natural_keys)]
        n_sessions = len(total_trials)

        time_headout = np.array([session_info[animal][s]['engagement'][0] for s in session_info[animal]])/1E6
        time_total = np.array([session_info[animal][s]['engagement'][1] for s in session_info[animal]])/1E6
        if not time_total.any():
            print "ZERO DIV ERROR: ", s
            percent_engaged = 0.
        else:
            percent_engaged = time_headout/time_total

        target_probs = [session_info[animal][s]['targetprob'] for s in sorted(outcome_info[animal], key=natural_keys)]
        distractor_contrasts = [session_info[animal][s]['curr_contrast'] for s in sorted(outcome_info[animal], key=natural_keys)]

        phases_completed = [session_info[animal][s]['phases_completed'] for s in sorted(outcome_info[animal], key=natural_keys)]

        try:
            pct_success = [ (outcome_info[animal][s]['hits']+outcome_info[animal][s]['crs']) / total_trials[i] for i,s in enumerate(sorted(outcome_info[animal], key=natural_keys))]
            pct_fail = [ (outcome_info[animal][s]['misses']+outcome_info[animal][s]['fas']) / total_trials[i] for i,s in enumerate(sorted(outcome_info[animal], key=natural_keys))]
        except ZeroDivisionError:
            trials = total_trials[i] for i,s in enumerate(sorted(outcome_info[animal], key=natural_keys))
            #### FIX ZERO DIV ERROR BY SESSION #####

        try:
            pct_hits = [ float(outcome_info[animal][s]['hits']) / (outcome_info[animal][s]['hits'] + outcome_info[animal][s]['misses']) for i,s in enumerate(sorted(outcome_info[animal], key=natural_keys))]
            pct_misses = [ float(outcome_info[animal][s]['misses']) / (outcome_info[animal][s]['hits'] + outcome_info[animal][s]['misses']) for i,s in enumerate(sorted(outcome_info[animal], key=natural_keys))]
        except ZeroDivisionError:
            pct_hits = [0.]*n_sessions
            pct_misses = [0.]*n_sessions

        try:
            pct_crs = [ float(outcome_info[animal][s]['crs']) / (outcome_info[animal][s]['crs'] + outcome_info[animal][s]['fas']) for i,s in enumerate(sorted(outcome_info[animal], key=natural_keys))]
            pct_fas = [ float(outcome_info[animal][s]['fas']) / (outcome_info[animal][s]['crs'] + outcome_info[animal][s]['fas']) for i,s in enumerate(sorted(outcome_info[animal], key=natural_keys))]
        except ZeroDivisionError:
            pct_crs = [0.]*n_sessions
            pct_fas = [0.]*n_sessions


        #------------------------------------------------
        # PLOTTING (by animal):
        #------------------------------------------------
        plt.close('all')

        n_outcomes = 2 #len(outcome_info[animal][s].keys())
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
        fig.suptitle(animal, fontsize=12, fontweight='bold')
        fig.facecolor='0.75'

        apply_standard_plotstyle(ax1)
        xvals = range(1, n_sessions + 1)
        yvals = [pct_success, pct_fail] # zip(*np.array([[pct_success], [pct_fail]]).T) #zip(*performance_vect)        
        colors = ['b', 'r']
        labels = ['success', 'failure']

        ax1.axis([1, n_sessions, 0., 1.])
        ax1.set_title('overall performance')
        ax1.set_ylabel('proportion of trials')

        for n in range(0, n_outcomes, 1):
            ax1.plot(xvals, yvals[n], 'o-', color=colors[n], label=labels[n])
            apply_standard_plotstyle(ax1)
        lgd = ax1.legend()
        lgd.get_frame().set_alpha(0.5)

        ax1.fill_between(xvals, session_info[animal][s]['targetprob_lower'], 1.0, facecolor='gray', \
                edgecolor='None', alpha=0.50, interpolate=True)


        # ax2:  plot adjusted outcome performance:
        xvals = range(1, n_sessions + 1)
        yvals = [pct_hits, pct_misses, pct_crs, pct_fas]#zip(*adj_performance_vect)        
        colors = ['r', 'b', 'm', 'c']
        labels = ['corrLicks', 'badIgnores', 'corrIgnores','badLicks']

        ax2.axis([1, n_sessions, 0., 1.])
        ax2.set_title('adjusted performance')
        ax2.set_ylabel('proportion of trials')

        for n in range(0, len(yvals), 1):
            ax2.plot(xvals, yvals[n], 'o-', color=colors[n], label=labels[n])
            apply_standard_plotstyle(ax2)
        lgd = ax2.legend()
        lgd.get_frame().set_alpha(0.5)


        # ax3:  plot staircase performance
        xvals = range(1, n_sessions + 1)
        yvals = [target_probs, distractor_contrasts]
        styles = ['o-', 'o--']
        markercolors = ['k','g']
        labels = ['target prob', 'distractor contrast']
        maxy=1.0

        ax3.axis([1, n_sessions, 0., 1.])
        ax3.set_title('staircase training performance')
        ax3.set_ylabel('value')

        for n in range(0, len(yvals), 1):
            ax3.plot(xvals, yvals[n], styles[n], color='k', label=labels[n], \
                    markerfacecolor=markercolors[n])
            apply_standard_plotstyle(ax3) 
        lgd = ax3.legend()
        lgd.get_frame().set_alpha(0.5)

        phase_idx = [1, 2, 3, 4, 5]
        phase_colors = ['r','DarkOrange','y', 'g','b']
        for i, p in enumerate(phases_completed):
            ax3.axvline(x=xvals[i], color=phase_colors[int(p)], linewidth=4, \
                alpha=0.4)



        # ax4:  plot trials counts:
        xvals = range(1, n_sessions + 1)
        yvals = total_trials
        maxy = max(yvals)
        
        ax4.axis([1, n_sessions, 0., maxy])
        ax4.set_title('trials per session')
        ax4.set_ylabel('num trials')

        ax4.plot(xvals, yvals, 'o-')
        apply_standard_plotstyle(ax4, nonpercents=1)


        # ax5:  plot engagement vals
        xvals = range(1, n_sessions + 1)
        yvals = percent_engaged
        
        ax5.axis([1, n_sessions, 0., 1.])
        ax5.set_title('engagement')
        ax5.set_ylabel('proportion of time head out')
        
        ax5.plot(xvals, yvals, 'o-')
        apply_standard_plotstyle(ax5)



        # ax6:  phase legend
        phase_colors = ['r','DarkOrange','y', 'g','b']
        nPhases = len(phase_colors)
        xvals = [1,2,3,4,5]
        yvals = [(1,1,1,1,1), (1,1,1,1,1), (1,1,1,1,1), \
                    (1,1,1,1,1), (1,1,1,1,1)]
        labels = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5']
        
        ax6.axis([1, 5, 1, 5])
        # ax6.set_title('phase key')
        
        apply_standard_plotstyle(ax6, default=1)
        for n in range(0, nPhases):
            ax6.plot(xvals, yvals[n], '-', lw=4, color=phase_colors[n], \
                            alpha=0.4, label=labels[n])
        adjust_spines(ax6, [])
        # plt.rc('legend', fontsize=12, loc='center', fancybox=True)
        lgd = ax6.legend(loc='center', prop={'size':12}, title='phase key', \
                            frameon=False)


        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

        figname = '%s_training_summary.png' % animal
        plt.savefig(os.path.join(figpath, figname))
        plt.close('all')




def apply_standard_plotstyle(ax, nonpercents=0, default=0):
    if default == 0:
        # standard_axes = [1, n_sessions, 0, 1.0]
        ax.set_xlabel('sessions')
        
        plt.rc('legend', fontsize=8, loc='upper right', fancybox=True)
        plt.rc('axes', titlesize='medium', labelsize='small')
        plt.rc('xtick', labelsize='small')
        plt.rc('ytick', labelsize='small')
        plt.rc('lines', markersize=3, markeredgewidth=0.3)
        # plt.axis([1, n_sessions, 0, 1.0])

        # if nonpercents==1:
        #     plt.axis([1, n_sessions, 0, maxy])
    elif default == 1:
        plt.xlabel('')
        plt.rcdefaults()


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10)) # outward by 10 pts
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there's no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])



datadir = sys.argv[1]                                                           # data INPUT folder (all others are soft-coded relative to this one)

if __name__ == "__main__":

    plot_progress(datadir)
