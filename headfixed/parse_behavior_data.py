#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:18:50 2019

@author: julianarhee
"""
import pymworks 
import os
import glob

import numpy as np
import pylab as pl
import pandas as pd

from pipeline.python.utils import natural_keys, label_figure

#%%

rootdir = '/n/coxfs01/behavior-data/head-fixed'
paradigmdir = os.path.join(rootdir, 'paradigm-files')

animalid = 'JC084'

#% Load MW behavior data:
mw_fpaths = sorted(glob.glob(os.path.join(paradigmdir, '*%s.mwk' % animalid)), key=natural_keys)
print('[%s] Found %i paradigm files.' % (animalid, len(mw_fpaths)) )

mw_fpaths

#%%

mw_fpath = mw_fpaths[-1]

df = pymworks.open(mw_fpath)

#%%
#% Load camera data:
#camera_fpath = glob.glob(os.path.join(data_dir, '%s*.txt' % mw_fname))[0]
#camera_data = pd.read_csv(camera_fpath, sep='\t')
#fr = 25.
#
##%%
#
##% parse camera info:
#acq_evs = df.get_events('camera_acquisition_start')
#acq_start_ev = [e for e in acq_evs if e.value==1][-1]
#
#all_frame_evs = df.get_events('camera_frame_grab')
#frame_evs = [f for f in all_frame_evs if f.time >= acq_start_ev.time and f.value==1]
#
#trigger_receive_intervals = np.diff([f.time for f in frame_evs]) / 1E6
#camera_frame_intervals = np.array([round(float(i)/1E9, 3) for i in np.diff(camera_data['frame_tstamp'])])
#
## How many frames did we miss?
#nframes_mw = len(frame_evs)
#nframes_cam = camera_data.shape[0]
#print("Received %i frame-grab triggers, camera saved %i frames." %(nframes_mw, nframes_cam))

#%

# print "Relative camera stamp time (min/max)", camera_frame_intervals.min(), camera_frame_intervals.max()


#data_save_intervals = np.diff(camera_data['relative_time'])
#
#
#pl.figure()
#pl.hist(data_save_intervals, alpha=0.5, color='blue')
#pl.hist(trigger_receive_intervals, alpha=0.5, color='orange')
#pl.hist(camera_frame_intervals, alpha=0.5, color='green')
#
##%
#expected_int = 1/fr
#
#pl.figure()
#pl.plot(trigger_receive_intervals)
#pl.xlabel('frame intervals')
#pl.ylabel('time betwen frames (s)')
#pl.title("Receiver: expected frame interval: %.2f s (%.1f Hz)" % (expected_int, fr))


#%%

def get_timekey(item):
    return item.time

def get_session_time(df):
    state_evs = df.get_events('#state_system_mode') 
    start_ev = [e for e in state_evs if e.value==2][0]
    start_ix = [e.time for e in state_evs].index(start_ev.time)
    stop_ev = next(e for e in state_evs[start_ix:] if e.value == 1 or e.value==0)
    
    return start_ev, stop_ev

def get_events_in_bounds(df, events=[], value=None, t_start=None, t_stop=None):
    for event_name in events:
        assert event_name in df.get_codec().values(), "Event not in codec! -- %s" % event_name
    if value is None:
        evs = [e for e in df.get_events(events) if t_start <= e.time <= t_stop]    
    else:
        evs = [e for e in df.get_events(events) if e.value == value\
                        and t_start <= e.time <= t_stop]    
    return sorted(evs, key=get_timekey)

def get_trial_starts(df, t_start=None, t_stop=None):
    '''
    Get all trial start/stop announcements. 
    For each trial stop (complete trial), find corresponding trial start, 
    i.e., stop = master, start = slave.

    arg1=slave, arg2=master
    
    Find closest matching slave event for each master event.
    direction : int, -1, 0 or 1
        if -1, slave events occur before master events
        if  1, slave events occur after master events
        if  0, slave and master events occur simultaneously

    '''
    trial_starts = get_events_in_bounds(df, events=['announce_start'], value=1, t_start=t_start, t_stop=t_stop) 
    trial_stops = get_events_in_bounds(df, ['announce_end'], value=1, t_start=t_start, t_stop=t_stop) 
    print("Found %i trial starts, %i trial stops." % (len(trial_starts), len(trial_stops)))
    trial_start_evs = pymworks.events.utils.sync(trial_starts, trial_stops, direction=-1)
    trial_times = pd.DataFrame({'t_start': [e.time for e in trial_start_evs],
                                't_end': [e.time for e in trial_stops]})

    return trial_times #trial_start_evs, trial_stops


#def get_outcome_events(df, t_start=None, t_stop=None, 
#                       outcome_types=['success', 'failure', 'ignore', 'abort']):
#    outcome_names = ['announce_%s' % outcome for outcome in outcome_types]
#    outcome_evs = get_events_in_bounds(df, events=outcome_names, value=1, t_start=t_start, t_stop=t_stop)
#    
#    outcomes = pd.DataFrame({'outcome': [e.name.split('announce_')[-1] for e in outcome_evs],
#                             't_outcome': [e.time for e in outcome_evs],
#                              })
#    outcome_counts = outcomes.groupby('outcome').count().T
#    n_outcomes = outcome_counts.sum(axis=1)
#    print("Found %i outcomes" % n_outcomes)
#
#    return outcomes
    

def get_trial_events(df, t_start=None, t_stop=None, 
                       outcome_types=['success', 'failure', 'ignore', 'abort']):

    # Get start and stop flags for each trial:
    trial_times = get_trial_starts(df, t_start=t_start, t_stop=t_stop) 
    
    # Get all outcome events:
    outcome_names = ['announce_%s' % outcome for outcome in outcome_types]
    outcome_evs = get_events_in_bounds(df, events=outcome_names, value=1, t_start=t_start, t_stop=t_stop)
    
    # For each trial, identify outcome and its time:
    trial_events = trial_times.copy()
    all_outcomes = []
    for ti in range(trial_times.shape[0]):
        found_outcomes = [e for e in outcome_evs if trial_times['t_start'][ti] < e.time < trial_times['t_end'][ti]]
        assert len(found_outcomes) <= 1, "More than 1 outcome found in trial time block! -- Trial %i" % ti
        if len(found_outcomes) == 1:
            curr_outcome = found_outcomes[0]
        elif len(found_outcomes) == 0:
            curr_outcome = None
        all_outcomes.append(curr_outcome)
    
    # Update trial events dataframe:
    trial_events['outcome'] = [e.name.split('_')[-1] if e is not None else e for e in all_outcomes]
    trial_events['t_outcome'] = [e.time if e is not None else None for e in all_outcomes]
    
    return trial_events
    



#    ## Get stimulus ons:
#    announce_stim_on = get_events_in_bounds(df, events=['announce_stimulus_on'], value=1, t_start=t_start, t_stop=t_stop) 
#    announce_stim_off = get_events_in_bounds(df, events=['announce_stimulus_off'], value=1, t_start=t_start, t_stop=t_stop) 
#    print("Found %i stimulus on announcements." % len(announce_stim_on)) 
#    print("Found %i stimulus off announcements." % len(announce_stim_off))
#    assert len(announce_stim_on)==len(announce_stim_off), "Unmatched stim-on and stim-off announcements!"


#%%
def assign_stim_to_trial(df, t_starts=[], t_ends=[], black_list=['action', 'file_hash', 'filename']):
    '''
    For each pair of trial start and trial end times, find all stimulus udpate events.
    Return as dataframe.
    
    Ignore subset of stimDisplayUpdate dictionary items.
    '''
    stim_df_list = []
    for ti, (ts, te) in enumerate(zip(t_starts, t_ends)):
        if ti % 20 == 0:
            print("Processed %i of %i trials" % (int(ti+1), len(t_starts)))
            
        ### Get all stimulus events
        devs = get_events_in_bounds(df, events=['#stimDisplayUpdate'], t_start=ts, t_stop=te)
        update_evs = [d for d in devs if len(d.value) > 1]
        
        stims = []
        for iev in update_evs:
            stimd = iev.value[1]
            stimd.update({'t_update': iev.time})
            for key in black_list:
                if key in stimd.keys():
                    stimd.pop(key)
            if '.png' in stimd['name']:
                stimd['name'] = stimd['name'][0:-4]
            stims.append(stimd)
    
        stim_df = pd.DataFrame(stims)
        stim_df['trial'] = [ti for _ in range(len(stims))] # save trial number
        stim_df_list.append(stim_df)
    
    stimdf = pd.concat(stim_df_list, axis=0).reset_index(drop=True)
    
    return stimdf


def assign_encoder_to_trial(df, t_starts=[], t_ends=[]):
    '''
    For each pair of trial start and trial end times, find all encoder udpate events.
    Return as dataframe.
    '''
    encoder_df_list = []
    for ti, (ts, te) in enumerate(zip(t_starts, t_ends)):
        ### Get all encoder events
        encoder_evs = get_events_in_bounds(df, events=['encoder_ticks'], t_start=ts, t_stop=te)
        
        encoder_df = pd.DataFrame({'value': [e.value for e in encoder_evs],
                                   't_update': [e.time for e in encoder_evs],
                                   'trial': [ti for _ in range(len(encoder_evs))]})
        
        encoder_df_list.append(encoder_df)

    encoder = pd.concat(encoder_df_list, axis=0).reset_index(drop=True)

    return encoder

#%%



## Get session bounds:
start_ev, stop_ev = get_session_time(df)
print("Session dur: %.2f min" % ((stop_ev.time - start_ev.time)/1E6/60.))
t_start = start_ev.time
t_stop = stop_ev.time


### Get trial start/ends:
#trial_times = get_trial_starts(df, t_start=t_start, t_stop=t_stop)

### Get all outcome events:
trials = get_trial_events(df,  t_start=t_start, t_stop=t_stop)
n_trials = trials.shape[0]

fig, ax = pl.subplots()
for ti in range(n_trials):

    ax.plot([0, (trials['t_end'][ti] - trials['t_start'][ti])/1E6], [ti, ti], 'k', alpha=0.5)
    ax.plot((trials['t_outcome'][ti] - trials['t_start'][ti])/1E6, ti, 'r*', markersize=1)
    
#
#first_start =  trials['t_start'][0] 
#fig, ax = pl.subplots()
#for ti in range(n_trials):
#
#    ax.plot([(trials['t_start'][ti] - first_start)/1E6, (trials['t_end'][ti] - first_start)/1E6], [ti, ti], 'k', alpha=0.5)
#    ax.plot((trials['t_outcome'][ti] - first_start)/1E6, ti, 'r*', markersize=1)
#    
    
trial_durs = (trials['t_end'] - trials['t_start'])/1E6
pl.figure()
pl.hist(trial_durs, alpha=0.5)
pl.xlabel("trial duraton (s)")
pl.ylabel('counts')
pl.title("Max: %.2f, Min: %.2f" % (trial_durs.max(), trial_durs.min()))


#%%
# Assign stimulus events to trials
stimdf = assign_stim_to_trial(df, t_starts=trials['t_start'], t_ends=trials['t_end'])

n_trials = len(stimdf['trial'].unique())
print("Got stimulus update events for %i trials." % n_trials)

encoder = assign_encoder_to_trial(df, t_starts=trials['t_start'], t_ends=trials['t_end'])

#%%

pl.figure()

for t in range(n_trials):
    curr_xpos = stimdf[stimdf['trial']==t]['pos_x'].values
    curr_encoder = encoder[encoder['trial']==t]['value'].values
    
    pl.plot(curr_xpos, 'b',)
    pl.plot(curr_encoder, 'orange')
    
#%%
import seaborn as sns
#%%
n_trials_plot = 10
fig, axes = pl.subplots(n_trials_plot, 1, sharex=True, sharey=True, figsize=(4, n_trials_plot))

emin = encoder['value'].min()
emax = encoder['value'].max()
max_e = max([abs(emin), abs(emax)])


for tix in range(n_trials_plot):
    ax = axes[tix]
    
    tdf = encoder[encoder['trial']==tix]
    sdf = stimdf[stimdf['trial']==tix]
    stim_name = sdf['name'].unique()[0]
    
    ts = trials['t_start'][tix]
    te = trials['t_end'][tix]
    t_outcome = (trials['t_outcome'][tix] - ts)/1E6
    
    curr_color = 'b' if trials['outcome'][tix] == 'success' else 'r'
    
    # Plot wheel:
    wheel_t = (tdf['t_update'].values - ts)/1E6
    wheel_v = tdf['value'].values
    outcome_ix = int(np.where(abs(wheel_t - t_outcome) == np.min(abs(wheel_t - t_outcome)))[0])
    ax.plot(wheel_t[0:outcome_ix], wheel_v[0:outcome_ix], label='wheel', color='k', alpha=0.7)
    ax.plot(wheel_t[outcome_ix:], wheel_v[outcome_ix:], label='wheel', color='k', alpha=0.2)
    
    #  Plot stimulus pos:
    stimulus_t = (sdf['t_update'].values - ts)/1E6
    stimulus_v = sdf['pos_x'].values
    outcome_ix_stim = int(np.where(abs(stimulus_t - t_outcome) == np.min(abs(stimulus_t - t_outcome)))[0])
    ax.plot(stimulus_t[0:outcome_ix_stim], stimulus_v[0:outcome_ix_stim], label=stim_name, color=curr_color, alpha=0.7)
    ax.plot(stimulus_t[outcome_ix_stim:], stimulus_v[outcome_ix_stim:], label=stim_name, color=curr_color, alpha=0.2)

    ax.plot(0, 0, 'g', marker='^')
    ax.plot(t_outcome, 0, 'magenta', marker='^')

    ax.set_ylim([-max_e, max_e])
    if tix != n_trials_plot-1:
        ax.xaxis.set_visible(False)
        sns.despine(ax=ax, bottom=True, trim=True, offset=2)
    else:
        sns.despine(ax=ax, top=True, right=True, trim=True, offset=2)

    #ax.legend(fontsize=6)


    
    
