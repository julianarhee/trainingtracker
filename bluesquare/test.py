#!/usr/bin/env python

import os
import glob
import json
import pymworks
import re


raw_dir = '/Users/julianarhee/Documents/coxlab/projects/old_data/gonogo_bluesquare_thresholdv2/raw'
animalid = 'T6'

dfns = glob.glob(os.path.join(raw_dir, animalid, '*.mwk'))


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


class struct():
    pass

def parse_datafile_name(dfn):
    fn = os.path.splitext(os.path.split(dfn)[-1])[0]
    fparts = fn.split('_')

    assert len(fparts) == 2, "*Warning* Unknown naming fmt: %s" % str(fparts)
    animalid = fparts[0]
    datestr = fparts[1]

    # Make sure no exra letters are in the datestr (for parta, b, etc.)
    if not datestr.isdigit():
        datestr = re.split(r'\D', datestr)[0] # cut off any letter suffix
    if len(datestr) == 6:
        session = '20%s' % datestr 
    elif len(datestr) == 8:
        session = datestr 

    return animalid, session


def get_run_time(df):
    state_modes = df.get_events('#state_system_mode')
    
    running = next(d for d in state_modes if d.value==2)
    start_time = running.time

    strt = state_modes.index(running)
    try:
        stopping = next(d for d in state_modes[strt:] if d.value != 2) 
        end_time = stopping.time
    except StopIteration:
        end_time = df.get_maximum_time()

    run_bounds = (start_time, end_time)

    return run_bounds

def parse_trials(df, remove_orphans=True):
    boundary = get_run_time(df)

    tmp_devs = df.get_events('#stimDisplayUpdate')                      # get *all* display update events
    tmp_devs = [i for i in tmp_devs if boundary[0] <= i['time']<= boundary[1]] #\
                #]                                 # only grab events within run-time bounds (see above)

    dimmed = df.get_events('FlagDimTarget')
    dimmed = np.mean([i.value for i in dimmed])                         # set dim flag to separate trials based on initial stim presentation

    resptypes = ['correct_lick','correct_ignore',\
                    'bad_lick','bad_ignore']
                     #,\ # 'aborted_counter']                           # list of response variables to track...

    outevs = df.get_events(resptypes)                                   # get all events with these vars
    outevs = [i for i in outevs if i['time']<= boundary[1] and\
                i['time']>=boundary[0]]
    R = sorted([r for r in outevs if r.value > 0], key=lambda e: e.time)

    T = pymworks.events.display.to_trials(tmp_devs, R,\
                remove_unknown=False)                                   # match stim events to response events

    if dimmed:                                                          # don't double count stimulus for "faded" target stimuli
        trials = [i for i in T if 'alpha_multiplier' not in i.keys() or i['alpha_multiplier']==1.0]
    else:
        trials = T

    print "N total response events: ", len(trials)
    if remove_orphans:                                                  # this should always be true, generally...
        orphans = [(i,x) for i,x in enumerate(trials) if\
                    x['outcome']=='unknown']
        trials = [t for t in trials if not t['outcome']=='unknown']     # get rid of display events without known outcome within 'duration_multiplier' time
        print "Found and removed %i orphan stimulus events in file %s"\
                    % (len(orphans), dfn)
        print "N valid trials: %i" % len(trials)

    return trials


# def parse_trials(df, outcome_types=[], start_t=None, end_t=None, pixelclock=False):
#     ir_threshold = 500
#     outcome_types = ['correct_lick', 'correct_ignore', 'bad_lick', 'bad_ignore']
#     non_stim = ['BlankScreenGray', 'TargetOverlay']

#     trials = []
#     # get stimulus events:
#     devs = df.get_events('#stimDisplayUpdate')
#     if start_t is not None and end_t is not None:
#         devs = [d for d in devs if start_t <= d.time <= end_t]
#     devs = [d for d in devs if len(d.value) > 1]
#     devs = sorted(devs, key=lambda x: x.time)

#     # get trial outcomes:
#     tevs = df.get_events(outcome_types)
#     tevs = sorted([t for t in tevs if t.value != 0], key=lambda x: x.time)

#     # for each trial outcome, find matching stimdisplay ev in reverse
#     hevs = sorted(df.get_events('HeadInput'), key=lambda x: x.time)
#     if start_t is not None and end_t is not None:
#         tevs = [t for t in tevs if start_t <= t.time <= end_t]
#         hevs = [h for h in hevs if start_t <= h.time <= end_t]

#     trial_starts = sorted(df.get_events('trial_start'), key=lambda x: x.time)
#     tstarts = [t for t in trial_starts if t.value > 0]

#     prev_trial=None
#     outcome_iter = copy.copy(tevs)
#     for dix, dev in enumerate(devs):
#         try:
#             # Find first outcome that occurs after curr stim and before next stim
#             curr_outcome_ev = next(t for t in tevs if t.time > dev.time \
#                                 and t.time < devs[dix+1].time)
#             outcome_ix = tevs.index(curr_outcome_ev)
#         except StopIteration:
#             print "No proper outcome found. Skipping!"
#             continue

#         # Get stimulus info and outcome info for current trial:
#         stim_info = [s for s in dev.value if s['name'] not in non_stim]
#         if pixelclock is False:
#             assert len(stim_info) == 1, "** Warning ** More than 1 stim shown!"
#             stim_info = stim_info[0]

#         # Get trial start time:
#         if prev_trial_end is None:
#             prev_trial_end = 0
#         else:
#             prev_trial_end = tevs[outcome_ix-1].time
#         curr_head_evs = [h for h in hevs if prev_trial_end < h.time < dev.time]
#         head_sensed = sorted([h for h in curr_head_evs if h.value > ir_threshold], key=lambda x: x.time)
#         if len(head_sensed) == 0:
#             # try to find trial-start ev:
#             head_sensed = [t for t in tstarts if prev_trial_end < t.time < dev.time]
#         head_sensed_ev = head_sensed[0]

#         tstart = [t for t in tstarts if prev_trial_end < t.time < dev.time][0]

#         # Create entry for current trial:
#         curr_trial = {'stimulus': stim_info,
#                       'outcome': curr_outcome_ev.name,
#                       't_stimulus_on':  dev.time,
#                       't_choice': curr_outcome_ev.time,
#                       't_head_sensed': head_sensed_ev.time,
#                       't_start': tstart.time}
                     
#         trials.append(curr_trial)
        
#         # increment stim evs and head evs:
#         outcome_iter = tevs[outcome_ix+1:]
#         stim_iter = devs[dix+1:]



# curr_display_evs = devs[::-1]
#     for tev in tevs:
#         outcome_time = tev.time
#         curr_stim_ev = next(d for d in curr_display_evs if d.time < outcome_time)
#         stim_ev_ix = devs.index(curr_stim_ev)
#         stim_info = [s for s in curr_stim_ev.value if s['name'] not in non_stim]
#         if pixelclock is False:
#             assert len(stim_info) == 1, "More than 1 stimulus shown!"
#             stim_info = stim_info[0]

#         # Get time trial started:
#         if stim_ev_ix == 0:
#             prev_trial_t = 0
#         else:
#             prev_trial_t = devs[stim_ev_ix-1].time

#         curr_head_evs = [h for h in hevs if prev_trial_t < h.time < curr_stim_ev.time]
#         head_sensed = sorted([h for h in curr_head_evs if h.value > ir_threshold], key=lambda x: x.time)
#         head_sensed_ev = head_sensed[0]

#         curr_trial = {'stimulus': stim_info,
#                       'outcome': tev.name,
#                       't_stimulus_on':  curr_stim_ev.time,
#                       't_choice': tev.time,
#                       't_start': head_sensed_ev.time}
                     
#         trials.append(curr_trial)
        
#         # increment stim evs and head evs:
#         curr_display_evs = devs[stim_ev_ix:][::-1]

#     return trials

class SessionData():
    def __init__(self, session='', datapath=''):
        # Get session start/end times:
        if isinstance(datapath, list):
            all_bounds = []
            for dpath in sorted(datapath, key=natural_keys):
                df = pymworks.open(dpath)
                all_bounds.append(get_run_time(df))
            start_t = min([b[0] for b in all_bounds])
            end_t = max([b[1] for b in all_bounds])
            run_bounds = (start_t, end_t)
        else: 
            df = pymworks.open(datapath)
            run_bounds = get_run_time(df)

        self.date = session
        self.datafiles = datapath 
        self.server_address = df.get_events('#serialBridgeAddress')[-1].value
        self.server_name = df.get_events('#serverName')[-1].value
        self.start_time = run_bounds[0]
        self.end_time = run_bounds[1]
        df.close()

    def get_stats



class Animal():
    def __init__(self, name, rootdir, experiment=None):
        self.name = name
        self.experiment = experiment
        sessions = self.get_session_list(rootdir)
        self.sessions = dict() 
        for session, datapath in sessions.items():
            self.sessions[session] = SessionData(session=session, datapath=datapath)

    def get_session_list(self, rootdir):
        # Get all found .mwk files
        dfns = glob.glob(os.path.join(rootdir, self.name, '*.mwk'))
        assert len(dfns) > 0, "[ERROR: %s] No datafiles found in rootdir: %s" % (self.name, rootdir)
        print "[%s] Found %i .mwk raw datafiles." % (self.name, len(dfns))
    
        # Get session names from datafile name
        session_list = dict()
        for dfn in dfns:
            aid, sesh = parse_datafile_name(dfn)
            if aid != animalid: #self.name:
                continue
            if sesh in session_list.keys():
                if not isinstance(session_list[sesh], list):
                    session_list[sesh] = [session_list[sesh]]
                session_list[sesh].append(dfn)
            else:
                session_list[sesh] = dfn 

        return session_list


