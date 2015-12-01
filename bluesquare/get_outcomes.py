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
    if not os.path.exists(outpath):                                                     # initiate dict if data from this expt has never been processed before
        print "First analysis!"
        os.makedirs(outpath)
        for animal in session_dict.keys():
            new_sessions[animal] = session_dict[animal]
            print "Adding %s to animal-session dict." % animal                         
    else:                                                                               # only chopse new stuff to be analyzed (animals and/or sessions)
        for animal in session_dict.keys():
            outfiles = [os.path.join(outpath, i) for i in os.listdir(outpath) if animal in i]                    # get processed files for loading
            print "Existent outfiles: ", outfiles
            if outfiles:                                                                # only use new sessions if animal has been processed
                fn_summ = open([i for i in outfiles if '_summary.pkl' in i][0], 'rb')   # _summary.pkl is main analysis info, so will exist if animal's data previoulsy processed (i.e., if outfiles has anything)            
                summary = pkl.load(fn_summ)
                new_sessions[animal] = [s for s in session_dict[animal] if s not in summary.keys()]
                if new_sessions[animal]:
                    print "Analyzing sessions: %s,\n%s" % (animal, new_sessions[animal])
                else:
                    print "No new sessions found."
                fn_summ.close()
            else:
                print "New animal %s detected..." % animal                              # get all sessions if new animal
                new_sessions[animal] = session_dict[animal]

    return new_sessions


def get_session_summary(dfns):
    '''
    Returns and saves a dict for a given animal with session names as keys, 
        e.g., {'20151123_JR003E.mwk': [[{trial1 dict}, {trial2 dict}, etc.]]}
        nested lists in case stop and start occurred more than once mid-session

    dfns : list
        list of strings containing datafiles (.mwk) to be parsed
    '''
    # summary = dict()

    # 
    trials = parse_trials(dfns)


    # METHOD 2:  summaries + other stats w/ stat funcs:
    # stats = get_outcome_stats()
    # summary['stats'] = process_datafiles(dfns, stats)                           # session: list of outcomes + stats

    return trials


###############################################################################
# TRIAL SPECIFIC FUNCTIONS:
###############################################################################


# METHOD 2a. ?? ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# FROM MORPH ANALYSES:

# def stat(extract_function=get_session_range, combine_function=lambda a, b: a + b):
#     return (extract_function, combine_function)


def get_session_stats():
    stats = {}
    for n in ('session_correct_lick', 'session_bad_lick', \
            'session_bad_ignore', 'session_correct_ignore', \
            'session_target_trialcounter', 'session_distractor_trialcounter', \
            'trials_in_session'): #, 'aborted_counter'):
        stats[n] = stat(functools.partial(get_session_range, name=n))
    for n in ('targetprob', 'curr_contrast', 'curr_phase', \
            'targetprob_lower'):
        stats[n] = stat(functools.partial(get_last, name=n), \
                lambda a, b: b)
    stats['engagement'] = stat(measure_engagement, \
            lambda a, b: (a[0] + b[0], a[1] + b[1]))

    return stats


# METHOD 2b. ?? ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def process_datafiles(dfns, stats):
    data = {}
    try:
        for dfn in dfns:
            logging.debug("Datafile: %s" % dfn)
            df = None
            df = pymworks.open(dfn)
            r = {}
            for s in stats:
                r[s] = stats[s][0](df)
                logging.debug("\t%s: %s" % (s, r[s]))
            df.close()
            data[dfn] = r
        return data
    except KeyError:
        print "NO KEY FOUND: %s", dfn


def get_outcomes(df, name=''):
    evs = df.get_events('#stimDisplayUpdate')
    outcomes = df.get_events(name)
    ts = pymworks.events.display.to_trials(evs, outcomes)
    gts = datautils.grouping.group(ts, 'outcome')
    nOutcomes = datautils.grouping.ops.stat(gts, len)

    respkeys = [k for k in nOutcomes.keys()]

    outcome = []
    for k in respkeys:
        outcome.append(nOutcomes[k])

    return outcome


def stat(extract_function=get_outcomes, \
        combine_function=lambda a, b: a + b):
    return (extract_function, combine_function)


def get_outcome_stats():
    stats = {}
    for n in ('session_correct_lick', 'session_bad_lick', 'session_bad_ignore', \
            'session_correct_ignore'):
        stats[n] = stat(functools.partial(get_outcomes, name=n))
    for n in ('targetprob', 'curr_contrast', 'phases_completed', \
            'targetprob_lower'):
        stats[n] = stat(functools.partial(get_last, name=n), \
                lambda a, b: b)
    stats['engagement'] = stat(measure_engagement, \
            lambda a, b: (a[0] + b[0], a[1] + b[1]))
    return stats



# METHOD 1 ?? ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# 565 -- last count each 393+164 -- 557
# 591

# idles = df.get_events('idle_trial')    # all ENDS | #true '#announceTrial' events
# idles = session_incremented(idles)

# ends = df.get_events('trial_end')
# ends = session_incremented(ends)

# announces = df.get_events('#announceTrial')
# announces = [i for i in announces if i.value]

# completes = df.get_events('trial_completed')
# completes = session_incremented(completes)

# len(ends) - len(idles) == len(completes)

# B = df.get_events('idle_trial')[-1]['value'] - df.get_events('idle_trial')[0]['value']  # IDLES

# C = df.get_events('trial_completed')[-1]['value'] - df.get_events('trial_completed')[0]['value']
# # A - B = C = number of session_counts total

# rewards = [i for i in df.get_events('reward_given') if i.value]
# reached = [i for i in df.get_events('outcome_reached') if i.value]


# def get_trial_outcomes(dfns, remove_orphans=1):
#     trialdata = {}                                                  # initiate output dict
#     for dfn in dfns:
#         df = None
#         df = pymworks.open(dfn)                                     # open the datafile
#         tmp_devs = df.get_events('#stimDisplayUpdate')                  # get all display update events
#         # stims = pymworks.events.display.to_stims(devs)

#         # tmp_devs = [d for d in tmp_devs if None not in d.value]
#         # devs = [e for e in tmp_devs for i in e.value if i['name']=='BlueSquare']

#         dimmed = np.mean([i.value for i in dimmed])

#         # if dimmed:
#         #     devs = [e for e in devs for i in e.value if 'alpha_multiplier' in i.keys() and i['alpha_multiplier']==1.0]


#         # stims = pymworks.events.display.to_stims(df.get_events('#stimDisplayUpdate'))
#         # sstims = [i for i in stims if i['alpha_multiplier']==1.0]

#         # ss = pymworks.events.display.to_stims(devs)
#         # # stims and ss are off by # of "faded" target occurrences...
#         # # WTF is the diff between 'ss' and 'dev' (i.e., besides the fading)??

#         # G = []
#         # Gb = []
#         # for i,x in enumerate(devs): #devs
#         #     s_match = [s for s in ss if s['time']==x['time']]
#         #     if s_match:
#         #         G.append([i,x])
#         #     else:
#         #         Gb.append([i,x])

#         resptypes = ['session_correct_lick','session_correct_ignore',\
#                     'session_bad_lick','session_bad_ignore'] #,\
#                     # 'aborted_counter']                              # list of response variables to track...
#         outevs = df.get_events(resptypes)                           # get all events with these vars
#         responses = []
#         for r in resptypes:                                         # get list of incrementing variables
#             revs = df.get_events(r)
#             incr_resp = session_incremented(revs)
#             responses.append(incr_resp)
#         responses = list(itertools.chain.from_iterable(responses))
#         real_responses = [r for r in responses if r.value != 0]
#         R = sorted(real_responses, key=lambda e: e.time)

#         T = pymworks.events.display.to_trials(tmp_devs, R, remove_unknown=False)
#         if dimmed:
#             trials = [i for i in T if i['alpha_multiplier']==1.0]
#         else:
#             trials = T

#         if remove_orphans:
#             trials = [t for t in trials if t['outcome'] not 'unknown']
#             orphans = [(i,x) for i,x in enumerate(trials) if x['outcome']=='unknown']
#             print "Found and removed %i orphan stimulus events." % len(orphans)


#         # D = sorted(devs, key=lambda e: e.time)

#         # delay = []
#         # for r,d in zip(R, D):
#         #     t_delay = (r['time'] - d['time']) / 1E9
#         #     delay.append(t_delay)

#         #synctrials = syncdicts(stims,real_responses,direction=-1)
#         #trash = [t for t in synctrials if t is None]
#         #trials = [t for t in synctrials if t not in trash]

#         # trials = pymworks.events.display.to_trials(devs, responses,\
#         #             remove_unknown=True)                            # get rid of orphan events

#         trialdata[dfn] = trials

#     return trialdata



def parse_trials(dfns, remove_orphans=True):
    trialdata = {}                                                              # initiate output dict
    
    for dfn in dfns:
        df = None
        df = pymworks.open(dfn)                                                 # open the datafile

        sname = os.path.split(dfn)[1]
        trialdata[sname] = []

        modes = df.get_events('#state_system_mode')
        run_idxs = np.where(np.diff([i['time'] for i in modes])<20)
        bounds = []
        for r in run_idxs[0]:
            stop_ev = next(i for i in modes[r:] if i['value']==0)
            bounds.append([modes[r]['time'], stop_ev['time']])

        print "................................................................"
        # print "----------------------------------------------------------------"
        print "Parsing file\n%s... " % dfn
        print "Found %i start events in session." % len(bounds)
        print "................................................................"


        for bidx,boundary in enumerate(bounds):
            print "----------------------------------------------------------------"
            print "SECTION %i" % bidx
            print "----------------------------------------------------------------"
            tmp_devs = df.get_events('#stimDisplayUpdate')                          # get *all* display update events
            tmp_devs = [i for i in tmp_devs if i['time']<= boundary[1] and i['time']>=boundary[0]]


            dimmed = df.get_events('flag_dim_target')
            dimmed = np.mean([i.value for i in dimmed])                             # set dim flag to separate trials based on initial stim presentation

            resptypes = ['session_correct_lick','session_correct_ignore','session_bad_lick','session_bad_ignore'] #,\ # 'aborted_counter']                                        # list of response variables to track...
            
            outevs = df.get_events(resptypes)                                       # get all events with these vars
            outevs = [i for i in outevs if i['time']<= boundary[1] and i['time']>=boundary[0]]
            # responses = []
            # for r in resptypes:                                                     # get list of incrementing variables
                # revs = df.get_events(r)
                # incr_resp = [r for r in revs if r['time']>=boundary[0] and r['time']<=boundary[1]]
                # incr_resp = session_incremented(revs)
                # responses.append(incr_resp)
            # responses = list(itertools.chain.from_iterable(responses))
            # real_responses = [r for r in responses if r.value != 0]                 # remove initial 0 value set at session start
            # R = sorted(real_responses, key=lambda e: e.time)                        # sort by time
            R = sorted([r for r in outevs if r.value > 0], key=lambda e: e.time)


            T = pymworks.events.display.to_trials(tmp_devs, R, remove_unknown=False)# match stim events to response events
            if dimmed:                                                              # don't double count stimulus for "faded" target stimuli
                trials = [i for i in T if i['alpha_multiplier']==1.0]
            else:
                trials = T

            print "N total response events: ", len(trials)
            if remove_orphans:                                                      # this should always be true, generally...
                orphans = [(i,x) for i,x in enumerate(trials) if x['outcome']=='unknown']
                trials = [t for t in trials if not t['outcome']=='unknown']          # get rid of display events without known outcome within 'duration_multiplier' time
                print "Found and removed %i orphan stimulus events in file %s" % (len(orphans), dfn)
                print "N valid trials: %i" % len(trials)

            trialdata[sname].append(trials)

    return trialdata



def get_trial_outcomes(dfns):
    trialdata = {}                                                  # initiate output dict
    for dfn in dfns:
        df = None
        df = pymworks.open(dfn)                                     # open the datafile
        devs = df.get_events('#stimDisplayUpdate')                  # get all display update events
        # stims = pymworks.events.display.to_stims(devs)
        resptypes = ['session_correct_lick','session_correct_ignore',\
                    'session_bad_lick','session_bad_ignore'] #,\
                    # 'aborted_counter']                              # list of response variables to track...
        outevs = df.get_events(resptypes)                           # get all events with these vars
        responses = []
        for r in resptypes:                                         # get list of incrementing variables
            revs = df.get_events(r)
            incr_resp = session_incremented(revs)
            responses.append(incr_resp)
        responses = list(itertools.chain.from_iterable(responses))
        real_responses = [r for r in responses if r.value != 0]

        trials = pymworks.events.display.to_trials(devs, responses,\
                    remove_unknown=True)                            # get rid of orphan events

        #synctrials = syncdicts(stims,real_responses,direction=-1)
        #trash = [t for t in synctrials if t is None]
        #trials = [t for t in synctrials if t not in trash]

        trialdata[dfn] = trials

    return trialdata


def session_incremented(evs, test=lambda a, b: b.value != (a.value - 1),
                   recurse=True):
    """
    Remove events where evs[i].value != (evs[i-1].value - 1)

    evs[i-1] occurs later in physical time than evs[i]. Re-reverse at end.
    """
    ## remove evs[i] if NEXT trial value (evs[i+1]) is more than CURR trial value ev[i] - 1.

    events = sorted(evs, key=lambda e: e.time, reverse=True)
    bad_indices = []
    bw = events
    for i in xrange(1, len(events)):
        if test(events[i - 1], events[i]):
            bad_indices.append(i)
    for i in bad_indices[::-1]: #[::-1]
        del(events[i])
    if recurse and len(bad_indices):
        bw = session_incremented(session_incremented(events, test=test, recurse=recurse))
    return sorted(bw, key=lambda e: e.time, reverse=False)


def get_max(df, name=''):
    return pymworks.stats.events.valuemax(df.get_events(name)).value


def get_last(df, name=''):
    return sorted(df.get_events(name), key=lambda e: e.time)[-1].value


def measure_engagement(df, name='head_input', threshold=500):
    return pymworks.stats.events.time_in_state(df.get_events(name), \
            test=lambda e: e.value < threshold)




###############################################################################
# do stuff...
###############################################################################

datadir = sys.argv[1]                                                               # data INPUT folder (all others are soft-coded relative to this one)
F = get_session_list(datadir)                                                       # list of all existent datafiles

outpath = os.path.join(os.path.split(datadir)[0], 'info')                           # set up output path for session info
if not os.path.exists(outpath):
    os.makedirs(outpath)
save_dict(F, outpath, 'sessions.pkl')

processed_path = os.path.join(os.path.split(outpath)[0], 'processed')               # path to previously processed data
new_sessions = get_new_sessions(processed_path, F)                                  # get dict of NEW sessions (and animals)


# analyze and parse new datafiles:
for animal in new_sessions.keys():
    tmp_new = new_sessions[animal]
    dfns = [os.path.join(datadir, animal, i) for i in tmp_new]

    # Get summary info for each session:
    # curr_summary = get_session_stats(datadir, animal, dfns)
    curr_summary = get_session_summary(dfns)
    outfiles = [os.path.join(processed_path, i)\
                for i in os.listdir(processed_path) if animal in i]

    if outfiles:
        fn_summ = open([i for i in outfiles if '_summary.pkl' in i][0], 'rb')   # _summary.pkl is main analysis info, so will exist if animal's data previoulsy processed (i.e., if outfiles has anything)            
        summary = pkl.load(fn_summ) # if summary file exists for current animal, just append 
        summary.update(curr_summary)
    else:
        summary = curr_summary                                                      # otherwise, create new summary dict for this animal

    fname = '%s_summary.pkl' % animal 
    save_dict(summary, processed_path, fname)                                       # save/overwrite summary file with new session data


    # # Parse each session into trial info:
    # curr_trials = parse_trials(dfns)

    # # outfiles = [os.path.join(processed_path, i)\
    # #             for i in os.listdir(processed_path) if animal in i]                 # get processed files for loading
    # print outfiles
    # if outfiles:
    #     fn_trials = open([i for i in outfiles if '_trials.pkl' in i][0], 'rb')

    #     trials = pkl.load(fn_trials)
    #     trials.update(curr_trials)
    # else:
    #     trials = curr_trials

    # fname = '%s_trials.pkl' % animal 
    # save_dict(trials, processed_path, fname)


