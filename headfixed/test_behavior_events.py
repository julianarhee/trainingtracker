#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:15:03 2019

@author: julianarhee
"""


import pymworks 
import os
import glob

import numpy as np
import pylab as pl
import pandas as pd

#%%

data_dir = '/home/julianarhee/Documents/behavior/headfixed/test_staircase'

#% Load MW behavior data:
mw_fname = 'test_training_phase3'
mw_fpath = os.path.join(data_dir, '%s.mwk' % mw_fname)
df = pymworks.open(mw_fpath)

#% Load camera data:
camera_fpath = glob.glob(os.path.join(data_dir, '%s*.txt' % mw_fname))[0]

camera_data = pd.read_csv(camera_fpath, sep='\t')


fr = 25.

#%%

#% parse camera info:
acq_evs = df.get_events('camera_acquisition_start')
acq_start_ev = [e for e in acq_evs if e.value==1][-1]

all_frame_evs = df.get_events('camera_frame_grab')
frame_evs = [f for f in all_frame_evs if f.time >= acq_start_ev.time and f.value==1]

trigger_receive_intervals = np.diff([f.time for f in frame_evs]) / 1E6
camera_frame_intervals = np.array([round(float(i)/1E9, 3) for i in np.diff(camera_data['frame_tstamp'])])

# How many frames did we miss?
nframes_mw = len(frame_evs)
nframes_cam = camera_data.shape[0]
print("Received %i frame-grab triggers, camera saved %i frames." %(nframes_mw, nframes_cam))

#%%

print "Relative camera stamp time (min/max)", camera_frame_intervals.min(), camera_frame_intervals.max()


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
    return evs

def get_trial_starts(df, t_start=None, t_stop=None):
    '''
    Get all trial start/stop announcements. 
    For each trial stop (complete trial), find corresponding trial start, 
    i.e., stop = master, start = slave.
    '''
    trial_starts = get_events_in_bounds(df, events=['announce_start'], value=1, t_start=t_start, t_stop=t_stop) 
    trial_stops = get_events_in_bounds(df, ['announce_end'], value=1, t_start=t_start, t_stop=t_stop) 
    print("Found %i trial starts, %i trial stops." % (len(trial_starts), len(trial_stops)))
    trial_start_evs = pymworks.events.utils.sync(trial_starts, trial_stops, direction=-1)
    trial_times = pd.DataFrame({'t_start': [e.time for e in trial_start_evs],
                                't_end': [e.time for e in trial_stops]})

    return trial_times #trial_start_evs, trial_stops


def get_outcome_events(df, t_start=None, t_stop=None, 
                       outcome_types=['success', 'failure', 'ignore', 'abort']):
    outcome_names = ['announce_%s' % outcome for outcome in outcome_types]
    outcome_evs = get_events_in_bounds(df, events=outcome_names, value=1, t_start=t_start, t_stop=t_stop)
    outcomes = pd.DataFrame({'outcome': [e.name.split('announce_')[-1] for e in outcome_evs],
                             't_outcome': [e.time for e in outcome_evs],
                              })
    outcome_counts = outcomes.groupby('outcome').count().T
    n_outcomes = outcome_counts.sum(axis=1)
    print("Found %i outcomes" % n_outcomes)

    return outcomes
    


#    ## Get stimulus ons:
#    announce_stim_on = get_events_in_bounds(df, events=['announce_stimulus_on'], value=1, t_start=t_start, t_stop=t_stop) 
#    announce_stim_off = get_events_in_bounds(df, events=['announce_stimulus_off'], value=1, t_start=t_start, t_stop=t_stop) 
#    print("Found %i stimulus on announcements." % len(announce_stim_on)) 
#    print("Found %i stimulus off announcements." % len(announce_stim_off))
#    assert len(announce_stim_on)==len(announce_stim_off), "Unmatched stim-on and stim-off announcements!"


#%%
def assign_stim_to_trial(df, t_starts=[], t_ends=[], black_list=['action', 'file_hash', 'filename']):
    stim_df_list = []
    for ti, (ts, te) in enumerate(zip(t_starts, t_ends)):
        ### Get all stimulus events
        devs = get_events_in_bounds(df, events=['#stimDisplayUpdate'], t_start=ts, t_stop=te)
        update_evs = [d for d in devs if len(d.value) > 1]
        #image_evs = [e for e in update_evs if e.value[1]['type'] != 'blankscreen']
        #print("Found %i image events" % len(image_evs))
        
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
trial_times = get_trial_starts(df, t_start=t_start, t_stop=t_stop)

### Get all outcome events:
outcome_types = ['success', 'failure', 'ignore', 'abort']
outcomes = get_outcome_events(df, t_start=t_start, t_stop=t_stop, outcome_types=outcome_types)

trials = pd.concat([trial_times, outcomes], axis=1)


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


    
    
