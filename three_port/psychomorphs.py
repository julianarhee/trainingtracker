#!/usr/bin/env python2

import os
import multiprocessing
import datetime
import pymworks
import matplotlib.pyplot as plt

import sys
import cPickle as pkl
import numpy as np
import re

from collections import Counter
from pandas import Series
import pandas as pd


import itertools
import functools

import logging
# import datautils

import pypsignifit as psi

file_dir = '/share/coxlab-behavior/mworks-data/three_port_morphs/pnas'


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

def get_session_list(input_path):
    '''
    Returns a dict with animal names as keys, all found sessions as values
        e.g., {'JR003E': ['JR003E_20151120.mwk', 'JR003E_20151123.mwk']}

    input_path : string 
        path to dict containing data files (organized by subject)
        each subject folder should contain .mwk files for all sessions.
    '''
    animals = os.listdir(input_path)
    F = dict()
    for a in animals:
        F[a] = os.listdir(os.path.join(input_path, a))
        F[a] = sorted([f for f in F[a] if os.path.splitext(f)[1]=='.mwk'])
    print "Session list: ", F

    return F


def get_new_sessions(outpath, session_dict):
    '''
    Returns a dict with animal names as keys, all *new* sessions as values
        e.g., {'JR003E': ['JR003E_20151120.mwk', 'JR003E_20151123.mwk']}

    outpath : string 
        path to dict containing previously saved processed data for each subject

    session_dict : dict
        dict containing all found sessions (values) for all animals (keys)
        output of get_new_sessions()
    '''
    new_sessions = dict()                                                               
    if not os.path.exists(outpath):                                             # initiate dict if data from this expt has never been processed before
        print "First analysis!"
        os.makedirs(outpath)
        for animal in session_dict.keys():
            new_sessions[animal] = session_dict[animal]
            print "Adding %s to animal-session dict." % animal                         
    else:                                                                       # only chopse new stuff to be analyzed (animals and/or sessions)
        for animal in session_dict.keys():
            outfiles = [os.path.join(outpath, i) for i in os.listdir(outpath)\
                            if animal+'_trials' in i]                                     # get processed files for loading
            print "Existent outfiles: ", outfiles
            if outfiles:                                                        # only use new sessions if animal has been processed
                fn_trials = open([i for i in outfiles if '_trials.pkl' in i][0],'rb')            
                prev_trials = pkl.load(fn_trials)
                new_sessions[animal] = [s for s in session_dict[animal] if s not in prev_trials.keys()]
                if new_sessions[animal]:
                    print "Analyzing sessions: %s,\n%s" % (animal, new_sessions[animal])
                else:
                    print "No new sessions found."
                fn_trials.close()
            else:
                print "New animal %s detected..." % animal                      # get all sessions if new animal
                new_sessions[animal] = session_dict[animal]

    return new_sessions




def parse_trials(dfns, remove_orphans=True):
    """
    Parse session .mwk files.
    Key is session name values are lists of dicts for each trial in session.
    Looks for all response and display events that occur within session.

    dfns : list of strings
        contains paths to each .mwk file to be parsed
    
    remove_orphans : boolean
        for each response event, best matching display update event
        set this to 'True' to remove display events with unknown outcome events
    """

    trialdata = {}                                                              # initiate output dict
    
    for dfn in dfns:
        df = None
        df = pymworks.open(dfn)                                                 # open the datafile

        sname = os.path.split(dfn)[1]
        trialdata[sname] = []

        modes = df.get_events('#state_system_mode')                             # find timestamps for run-time start and end (2=run)
        run_idxs = np.where(np.diff([i['time'] for i in modes])<20)             # 20 is kind of arbitray, but mode is updated twice for "run"
        bounds = []
        for r in run_idxs[0]:
            try:
                stop_ev = next(i for i in modes[r:] if i['value']==0 or i['value']==1)
            except StopIteration:
                end_event_name = 'trial_end'
                print "NO STOP DETECTED IN STATE MODES. Using alternative timestamp: %s." % end_event_name
                stop_ev = df.get_events(end_event_name)[-1]
                print stop_ev
            bounds.append([modes[r]['time'], stop_ev['time']])

        # print "................................................................"
        print "****************************************************************"
        print "Parsing file\n%s... " % dfn
        print "Found %i start events in session." % len(bounds)
        print "****************************************************************"


        for bidx,boundary in enumerate(bounds):
            # print "................................................................"
            print "SECTION %i" % bidx
            print "................................................................"
            #M1:#tmp_devs = df.get_events('#stimDisplayUpdate')                      # get *all* display update events
            # tmp_devs = [i for i in tmp_devs if i['time']<= boundary[1] and\
            #             i['time']>=boundary[0]]                                 # only grab events within run-time bounds (see above)

            #M1:#devs = [e for e in tmp_devs if not e.value[0]==[None]]


            # Check stimDisplayUpdate events vs announceStimulus:
            stim_evs = df.get_events('#stimDisplayUpdate')
            devs = [e for e in stim_evs if not e.value[0]==None]
            idevs = [i for i in devs for v in i.value if 'png' in v['name']]

            ann_evs = df.get_events('#announceStimulus')

            # SEE: AG8_160705.mwk -- issue with sevs...
            try:
                sevs = [i for i in ann_evs if 'png' in i.value['name']]
            except TypeError as e:
                print dfn
                print e

            if not len(idevs)==len(sevs):
                print "MISMATCH in event counts in DF: %s" % dfn
                print "-------------------------"
                print "#stimDisplayUpdate %i and #announceStimulus %i." % (len(idevs), len(sevs))

            #M1:#devs = [e for e in stim_evs if e.value[0] is not None]
            im_names = sorted([i['name'] for d in idevs for i in d.value if '.png' in i['name']], key=natural_keys)


            #im_names = sorted([i['name'] for d in devs for i in d.value if '.png' in i['name']], key=natural_keys)

            resptypes = ['success','ignore','failure']
                             #,\ # 'aborted_counter']                           # list of response variables to track...
            
            outevs = df.get_events(resptypes)                                   # get all events with these vars
            outevs = [i for i in outevs if i['time']<= boundary[1] and\
                        i['time']>=boundary[0]]
            R = sorted([r for r in outevs if r.value > 0], key=lambda e: e.time)

            T = pymworks.events.display.to_trials(stim_evs, R,\
                        remove_unknown=False)                                   # match stim events to response events

            trials = T
            print "N total response events: ", len(trials)
            if remove_orphans:                                                  # this should always be true, generally...
                orphans = [(i,x) for i,x in enumerate(trials) if\
                            x['outcome']=='unknown']
                trials = [t for t in trials if not t['outcome']=='unknown']     # get rid of display events without known outcome within 'duration_multiplier' time
                print "Found and removed %i orphan stimulus events in file %s"\
                            % (len(orphans), dfn)
                print "N valid trials: %i" % len(trials)

            trialdata[sname].append(trials)

    return trialdata



def get_trials_by_session(datadir):
    F = get_session_list(datadir)                                               # list of all existent datafiles

    outpath = os.path.join(os.path.split(datadir)[0], 'info')                   # set up output path for session info
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    save_dict(F, outpath, 'sessions.pkl')

    processed_path = os.path.join(os.path.split(outpath)[0], 'processed')           # path to previously processed data
    new_sessions = get_new_sessions(processed_path, F)                              # get dict of NEW sessions (and animals)


    # analyze and parse new datafiles:
    for animal in new_sessions.keys():
        tmp_new = new_sessions[animal]
        dfns = [os.path.join(datadir, animal, i) for i in tmp_new]

        # Parse trials for each session:
        curr_trials = parse_trials(dfns)

        outfiles = [os.path.join(processed_path, i)\
                    for i in os.listdir(processed_path) if animal in i]

        if '_trials' in outfiles:
            fn_trials = open([i for i in outfiles if '_trials.pkl' in i][0], 'rb')   
            trials = pkl.load(fn_trials)                                             # if summary file exists for current animal, just append 
            trials.update(curr_trials)
        else:
            trials = curr_trials                                                  # otherwise, create new summary dict for this animal

        fname = '%s_trials.pkl' % animal 
        save_dict(trials, processed_path, fname)                                   # save/overwrite summary file with new session data


        # ## TO DO: FIX THIS
        # # Get summary info for each session:
        # stats = get_session_stats()
        # curr_summary = process_datafiles(dfns, stats)

        # if outfiles:
        #     fn_summ = open([i for i in outfiles if '_summary.pkl' in i][0], 'rb')
        #     summary = pkl.load(fn_summ)                                             # if summary file exists for current animal, just append 
        #     summary.update(curr_summary)
        # else:
        #     summary = curr_summary 

        # fname = '%s_summary.pkl' % animal 
        # save_dict(summary, processed_path, fname)                                   # save/overwrite summary file with new session data


def fix_date(date):
    if len(date) == 6 or 7:  # YYMMDD
        return '20' + date
    elif len(date) == 8:  # YYYYMMDD
        return date
    else:
        logging.error("Invalid date(%s)" % date)
        raise IOError("Invalid date(%s)" % date)


def parse_filename(filename):
    name, ext = os.path.splitext(os.path.basename(filename))
    tokens = name.split('_')
    if len(tokens) != 2:
        print filename
        logging.error("Invalid filename(%s)" % name)
        raise IOError("Invalid filename(%s)" % name)
    animal = tokens[0].upper()
    if tokens[1].isdigit():  # no epoch
        epoch = ''
        date = fix_date(tokens[1])
    else:  # assume tokens[1][-1] is a letter
        epoch = tokens[1][-1]
        date = fix_date(tokens[1][:-1])
    return animal, date, epoch


# fn = '/share/coxlab-behavior/mworks-data/three_port_morphs/pnas/processed/AG6_trials.pkl'

def get_morph_stats(datadir, show_curve=True):
    processed_dir = os.path.join(os.path.split(datadir)[0], 'processed')
    animals = os.listdir(processed_dir)
    animals = [i for i in animals if i.endswith('_trials.pkl')]
    animals = [os.path.splitext(a)[0] for a in animals]

    P = dict()
    for animal in animals:

        P[animal] = dict()

        fn = os.path.join(processed_dir, animal+'.pkl')
        with open(fn, 'rb') as f:
            curr_trials = pkl.load(f)

        session_dates = []
        for t in curr_trials.keys():
            print t
            subject, date, epoch = parse_filename(t)
            session_dates.append(date)

        M = dict()
        for t in sorted(curr_trials.keys(), key=natural_keys):
            print t
            if len(curr_trials[t])==0:
                continue
            elif len(curr_trials[t])==1 or len(curr_trials[t])==2:
                ctrials = []
                check_idx = 0
                while ctrials==[]:
                    ctrials = curr_trials[t][check_idx]
                    check_idx += 1
            else:
                print "LEN: ", len(curr_trials[t])
                ctrials = curr_trials[t]

            df = pymworks.open(os.path.join(datadir, subject, t))

            # if no morphs, move on...
            mtrials = [i for i in ctrials if 'morph' in i['name']]
            if len(mtrials)==0:
                continue

            # Do this funky stuff to get default size:
            sizes = [i['size_x'] for i in ctrials]
            sizes = list(set(sizes))

            tmp_default_size = 40*df.get_events('StimAspectRatio')[0].value
            diffs = np.array(sizes) - tmp_default_size
            find_match = np.where(diffs==min(diffs, key=lambda x: abs(float(x) - tmp_default_size)))
            default_size = sizes[find_match[0][0]]

            anchor_trials = [i for i in ctrials if '_y0' in i['name'] and i['size_x']==default_size]
            anchor_names = list(set([i['name'] for i in anchor_trials]))

            morph_names = sorted(list(set([i['name'] for i in mtrials])), key=natural_keys)
            morphs = dict()
            for m in morph_names:

                morphs[m] = dict()
                morphs[m]['success'] = sum([1 for i in mtrials if i['name']==m and i['outcome']=='success'])
                morphs[m]['failure'] = sum([1 for i in mtrials if i['name']==m and i['outcome']=='failure'])
                morphs[m]['ignore'] = sum([1 for i in mtrials if i['name']==m and i['outcome']=='ignore'])
                morphs[m]['total'] = sum([morphs[m]['success'] + morphs[m]['failure'] + morphs[m]['ignore']])

            for m in anchor_names:

                if '_N1' in m:  # THIS IS SPACESHIP -- AG group, this is RIGHT
                    aname = 'morph0.png'
                elif '_N2' in m:  # THIS IS BUNNY -- this is LEFT
                    aname = 'morph21.png'

                if aname not in morphs.keys():
                    morphs[aname] = dict()
                morphs[aname]['success'] = sum([1 for i in anchor_trials if i['name']==m and i['outcome']=='success'])
                morphs[aname]['failure'] = sum([1 for i in anchor_trials if i['name']==m and i['outcome']=='failure'])
                morphs[aname]['ignore'] = sum([1 for i in anchor_trials if i['name']==m and i['outcome']=='ignore'])
                morphs[aname]['total'] = sum([morphs[aname]['success'] + morphs[aname]['failure'] + morphs[aname]['ignore']])

            M[t] = morphs

        all_morph_names = ['morph%i.png' % int(i) for i in range(22)]


        P[animal]['session'] = dict()
        session_keys = ['percent', 'total', 'success', 'failure', 'ignore']
        # initiate dict() for each session/morph:
        for session_key in session_keys:
            P[animal]['session'][session_key] = dict()
            for msession in M.keys():
                P[animal]['session'][session_key][msession] = dict()

        for mkey in M.keys():

            for morph in all_morph_names:

                morph_num = int(re.findall("[-+]?\d+?\d*", morph)[0])

                if morph not in M[mkey].keys() or M[mkey][morph]['total']==0:
                    for skey in P[animal]['session'].keys():
                        P[animal]['session'][skey][mkey][str(morph_num)] = 0.

                else:
                    for skey in P[animal]['session'].keys():

                        # if just need totals for a given morph in a given session:
                        if skey is not 'percent':
                            P[animal]['session'][skey][mkey][str(morph_num)] = float(M[mkey][morph][skey])

                        # otherwise, need to calculate %-choose-RIGHT (or LEFT) port:
                        elif skey == 'percent':
                            if morph_num <= 11:
                                P[animal]['session'][skey][mkey][str(morph_num)] = float(M[mkey][morph]['success']/float(M[mkey][morph]['total']))
                            elif morph_num > 11:
                                P[animal]['session'][skey][mkey][str(morph_num)] = float(M[mkey][morph]['failure']/float(M[mkey][morph]['total']))

        ALL = dict()
        for morph in sorted(all_morph_names, key=natural_keys):
            ALL[morph] = dict()
            ALL[morph]['success'] = []
            ALL[morph]['failure'] = []
            ALL[morph]['ignore'] = []
            ALL[morph]['total'] = []
            for i in M.keys():
                if morph in M[i].keys():
                    ALL[morph]['success'].append(M[i][morph]['success'])
                    ALL[morph]['failure'].append(M[i][morph]['failure'])
                    ALL[morph]['ignore'].append(M[i][morph]['ignore'])
                    ALL[morph]['total'].append(M[i][morph]['total'])
                else:
                    ALL[morph]['success'].append(0)
                    ALL[morph]['failure'].append(0)
                    ALL[morph]['ignore'].append(0)
                    ALL[morph]['total'].append(0)

        percents = dict()
        totals = dict()
        successes = dict()
        failures = dict()
        ignores = dict()
        for m in sorted(all_morph_names, key=natural_keys):
            morph_num = int(re.findall("[-+]?\d+?\d*", m)[0])
            totals[m] = float(sum(ALL[m]['total']))
            successes[m] = float(sum(ALL[m]['success']))
            failures[m] = float(sum(ALL[m]['failure']))
            ignores[m] = float(sum(ALL[m]['ignore']))
            if m not in ALL.keys() or sum(ALL[m]['total'])==0:
                percents[m] = 0
            else:
                if morph_num <= 11:
                    percents[m] = float(sum(ALL[m]['success'])) / float(sum(ALL[m]['total']))
                elif morph_num > 11:
                    percents[m] = float(sum(ALL[m]['failure'])) / float(sum(ALL[m]['total']))

        P[animal]['percent_right'] = percents
        P[animal]['totals'] = totals
        P[animal]['success'] = successes
        P[animal]['failure'] = failures
        P[animal]['ignore'] = ignores

        x = []
        xlabel = []
        for p in sorted(percents.keys(), key=natural_keys):
            x.append(percents[p])
            xlabel.append(p)

        x = np.flipud(x)
        xlabel = np.flipud(x)

        fig = plt.figure()
        plt.plot(x, 'r*')
        plt.ylabel('percent choose RIGHT port')
        plt.xlabel('morph #')
        subject = animal.split('_')[0]
        plt.title('%s' % subject)

        imname = '%s.png' % subject
        outdir = os.path.join(os.path.split(datadir)[0], 'figures')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        fig.savefig(outdir + '/' + imname)
        print outdir + '/' + imname

        if show_curve is True:
            plt.show()

    return P

def get_morph_counts(datadir):

    processed_dir = os.path.join(os.path.split(datadir)[0], 'processed')
    animals = os.listdir(processed_dir)
    animals = [i for i in animals if i.endswith('.pkl')]

    counts = dict()
    for animal in animals:
        fn = os.path.join(processed_dir, animal)
        with open(fn, 'rb') as f:
            curr_trials = pkl.load(f)

        counts[animal] = dict()
        for session in curr_trials.keys():
            if len(curr_trials[session])==0:
                print session
                continue
            else:
                curr_session = curr_trials[session][0]

            im_names = sorted([trial['name'] for trial in curr_session if '.png' in trial['name']], key=natural_keys)
            morphs = sorted([i for i in im_names if 'morph' in i], key=natural_keys)
            if len(morphs)==0:
                continue

            transforms = [i for i in im_names if i not in morphs]
            print "N morphs: ", len(morphs)
            print "N transforms: ", len(transforms)

            # counts = Counter(im_names)
            # print counts

            s = Series(im_names)
            vc = s.value_counts()
            vc = vc.sort_index()
            print vc
            mkeys = [i for i in vc.keys() if 'morph' in i]
            tkeys = [i for i in vc.keys() if not 'morph' in i]
            morph_total = sum(vc[mkeys])
            trans_total = sum(vc[tkeys])


            df = pd.DataFrame({'name':sorted(vc.keys(), key=natural_keys),
                           'count':[vc[k] for k in sorted(vc.keys(), key=natural_keys)]})

            # ax = df.plot(kind='bar',  title='Scores')
            # ax.set_ylim(0, 100)
            # for i, label in enumerate(list(df.index)):
            #     val = df.ix[label]['count']
            #     ax.annotate(str(val), (i-0.5, val+1))


            ax = df.plot.bar(title="Counts", rot=90,xticks=df.index)
            ax.set_ylim(0,100)
            ax.set_xticklabels(df.name)
            for p in ax.patches:
                ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

            plt.show()

            counts[animal][session] = vc

    return counts


def get_choice_matlab(P):
    nmorphs = 22

    animals = P.keys()
    data = dict()
    for animal in animals:
        data[animal] = np.zeros((nmorphs, 3))
        data[animal][:,0] = range(22)
        # data[animal][:,1] = np.array([P[animal]['percent_right'][i] for i in sorted([P][animal]['percent_right'].keys(), key=natural_keys)]) * np.array([P[animal]['totals'][i] for i in sorted(P[animal]['totals'].keys(), key=natural_keys)])
        data[animal][:,1] = np.array([P[animal]['percent_right'][i] for i in sorted(P[animal]['percent_right'].keys(), key=natural_keys)]) * np.array([P[animal]['totals'][i] for i in sorted(P[animal]['totals'].keys(), key=natural_keys)])

        data[animal][:,2] = np.array([P[animal]['totals'][i] for i in sorted(P[animal]['totals'].keys(), key=natural_keys)])

        data[animal][:,1] = np.flipud(data[animal][:,1])
        data[animal][:,2] = np.flipud(data[animal][:,2])

    return data

    # [ 0.94736842,  0.89189189,  0.92105263,  0.91891892,  0.94594595,
    #     0.89473684,  0.94736842,  0.8974359 ,  0.89473684,  0.71794872,
    #     0.68421053,  0.68421053,  0.62162162,  0.59459459,  0.5       ,
    #     0.60526316,  0.37837838,  0.41935484,  0.42105263,  0.18421053,
    #     0.23684211,  0.10666667]


def get_success_matlab(P):
    nmorphs = 22

    animals = P.keys()
    data = dict()
    for animal in animals:
        data[animal] = np.zeros((nmorphs, 3))
        data[animal][:,0] = range(22)
        # data[animal][:,1] = np.array([P[animal]['percent_right'][i] for i in sorted(P[animal]['percent_right'].keys(), key=natural_keys)]) * np.array([P[animal]['totals'][i] for i in sorted(P[animal]['totals'].keys(), key=natural_keys)])
        data[animal][:,1] = np.array([P[animal]['success'][i] for i in sorted(P[animal]['success'].keys(), key=natural_keys)])

        data[animal][:,2] = np.array([P[animal]['totals'][i] for i in sorted(P[animal]['totals'].keys(), key=natural_keys)])

        data[animal][:,1] = np.flipud(data[animal][:,1])
        data[animal][:,2] = np.flipud(data[animal][:,2])

    return data



from scipy.optimize import curve_fit

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y

# def weib(x,n,a):
#     return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)


def weib(x,a,c):

    return a * c * (1-np.exp(-x**c))**(a-1) * np.exp(-x**c)*x**(c-1)

def wb2LL(p, x): #log-likelihood
    return sum(log(stats.weibull_min.pdf(x, p[1], 0., p[0])))

def fit_sigmoid(P):

    for animal in P.keys():

        print "Fitting curve to: %s" % animal
        mnames = P[animal]['percent_right'].keys()
        mnums = sorted([int(re.findall("[-+]?\d+?\d*", m)[0]) for m in mnames])

        empties = []
        for i in P[animal]['totals']:
            if P[animal]['totals'][i]==0:
                empties.append(int(re.findall("[-+]?\d+?\d*", i)[0]))

        mnums_to_use = [i for j, i in enumerate(mnums) if j not in empties]

        xvals_to_use = mnums_to_use
        keys_yvals_to_use = [i for j,i in enumerate(sorted(P[animal]['percent_right'], key=natural_keys)) if j not in empties]
        yvals_to_use = [P[animal]['percent_right'][k] for k in sorted(keys_yvals_to_use, key=natural_keys)]

        # xvals = np.flipud(xvals_to_use)
        xdata = xvals_to_use
        ydata = [1-i for i in yvals_to_use] #np.flipud(yvals_to_use)

        try:
            popt, pcov = curve_fit(sigmoid, xdata, ydata, maxfev=1000)
            print popt
        except RuntimeError:
            print("Error - curve_fit failed:  %s" % animal)

        x = np.linspace(-1, 23, 50)
        #y = sigmoid(x, *popt)
        y = weib(x, *popt)

        fig = plt.figure()
        plt.plot(xdata, ydata, 'o', label='data')
        plt.plot(x,y, label='fit')
        plt.ylim(0, 1.05)
        plt.legend(loc='best')

        plt.title('%s: fit sigmoid' % animal)
        plt.ylabel('P(choose right port)')
        plt.xlabel('morph number')

        imname = '%s_fit_sigmoid.png' % animal
        outdir = os.path.join(os.path.split(datadir)[0], 'figures')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        fig.savefig(outdir + '/' + imname)
        print outdir + '/' + imname


        plt.show()

import h5py
import hdf5storage
import scipy.io
def convert(input):
    if isinstance(input, dict):
        return {convert(key): convert(value) for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [convert(element) for element in input]
    elif isinstance(input, str):
        # return input.encode('utf-8')
        return unicode(input)
    else:
        return input



datadir = sys.argv[1]   

if __name__ == "__main__":
    get_trials_by_session(datadir)
    P = get_morph_stats(datadir, show_curve=False)

    # Use SUCCESS as measure: 
    mdata = get_success_matlab(P)
    umdata = convert(mdata)

    fn = 'P_success.mat'
    outpath = os.path.join(os.path.split(datadir)[0], 'matfiles')
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    outfile = os.path.join(outpath, fn)

    scipy.io.savemat(outfile, mdict={'mdata': mdata})


    # Use CHOICE as measure 
    mdata = get_choice_matlab(P)
    umdata = convert(mdata)

    fn = 'P_choice.mat'
    outpath = os.path.join(os.path.split(datadir)[0], 'matfiles')
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    outfile = os.path.join(outpath, fn)

    scipy.io.savemat(outfile, mdict={'mdata': mdata})



    # hdf5storage.savemat(outfile, mdata)
    # hdf5storage.savemat(outfile, mdata)


    # hdf5storage.write(umdata, path=outpath, filename=fn)
    # hdf5storage.savemat(outfile, umdata)
    # hdf5storage.savemat(outfile, umdata, format='7.3', oned_as='column', store_python_metadata=True)

    # s = Series(im_names)
    # vc = s.value_counts()
    # vc = vc.sort_index()
    # print vc
    # mkeys = [i for i in vc.keys() if 'morph' in i]
    # tkeys = [i for i in vc.keys() if not 'morph' in i]
    # morph_total = sum(vc[mkeys])
    # trans_total = sum(vc[tkeys])


    # outcomes = dict()
    # outcomes['success'] = 0
    # outcomes['failure'] = 0
    # outcomes['ignore'] = 0
    # for m in mtrials:
    #     if m['outcome'] == 'success':
    #         outcomes['success'] += 1

    #     elif m['outcome'] == 'failure':
    #         outcomes['failure'] += 1

    #     elif m['outcome'] == 'ignore':
    #         outcomes['ignore'] += 1

    # m_outcomes[t] = outcomes
