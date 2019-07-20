#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:49:59 2019

@author: julianarhee
"""

import pymworks 
import os
import glob

import numpy as np
import pylab as pl
import pandas as pd

#%%

mw_fname = 'test_start'
#mw_fname = 'test_sync_rewired_25Hz'
#mw_fname = 'test_sync'
fn = '/home/julianarhee/Documents/behavior/headfixed/test_data/%s.mwk' % mw_fname
df = pymworks.open(fn)

fname = 'test_start_frame_metadata_20190718_151824580480'
#fname = 'test_sync_rewired_25Hz_frame_metadata_20190718_110532346875'
#fname = 'test_sync_frame_metadata_20190717_153442971074'
# test_triggers_frame_metadata_20190717_132132039828
camera_fpath = '/home/julianarhee/Documents/behavior/headfixed/test_data/%s.txt' % fname

camera_data = pd.read_csv(camera_fpath, sep='\t')


fr = 25.



#%%


acq_evs = df.get_events('camera_acquisition_start')
acq_start_ev = [e for e in acq_evs if e.value==1][-1]

all_frame_evs = df.get_events('camera_frame_grab')
frame_evs = [f for f in all_frame_evs if f.time >= acq_start_ev.time]

print(len(frame_evs))

frame_ons = [f for f in frame_evs if f.value == 1]
frame_offs = [f for f in frame_evs if f.value == 0 and f.time >= frame_ons[0].time]

nframes = min([len(frame_ons), len(frame_offs)])

#diffs = []
#for on, off in zip(frame_ons[0:nframes], frame_offs[0:nframes]):
#    diffs.append(off.time - on.time)

#frame_ints = [float(d)/1E6 for d in diffs]


frame_ints = np.diff([f.time for f in frame_ons]) / 1E6
cam_ints = np.array([float(i)/1E9 for i in np.diff(camera_data['frame_tstamp'])])

# How many frames did we miss?
nframes_mw = len(frame_ons)
nframes_cam = camera_data.shape[0]
print("Received %i frame-grab triggers, camera saved %i frames." %(nframes_mw, nframes_cam))

#%%
comp_relative_time_int = np.diff(camera_data['relative_time'])
cam_relative_time_int = np.diff(camera_data['relative_camera_time'])

pl.figure()
pl.hist(comp_relative_time_int, alpha=0.5, bins=100, color='b')
pl.hist(cam_relative_time_int, alpha=0.5, bins=10, color='orange')

print "Relative computer time (min/max)", comp_relative_time_int.min(), comp_relative_time_int.max()
print "Relative camera stamp time (min/max)", cam_relative_time_int.min(), cam_relative_time_int.max()

#%%


#pl.figure()
#pl.scatter([round(f,2) for f in frame_ints[0:nframes]], [round(f, 3) for f in cam_ints[0:nframes]])

#%%
pl.figure()
pl.hist(frame_ints, bins=100, alpha=0.5)
pl.hist(cam_ints, bins=100, alpha=0.5)

print frame_ints.min(), frame_ints.max()

#%%

expected_int = 1/fr

pl.figure()
pl.plot(frame_ints)
pl.xlabel('frame intervals')
pl.ylabel('time betwen frames (s)')
pl.title("Receiver: expected frame interval: %.2f s (%.1f Hz)" % (expected_int, fr))


#%%
fig = pl.figure()

ax = pl.subplot2grid((2, 2), (0, 0), colspan=2)

ax.plot(cam_ints, label='camera', c='k', alpha=0.5)
ax.plot(frame_ints, label='receiver', c='r', alpha=0.5)
ax.set_xlabel('frame intervals')
ax.set_ylabel('time betwen frames (s)')
ax.set_title("Expected frame interval: %.2f s (%.1f Hz)" % (expected_int, fr))
ax.legend()

#%%


fudge_amount = 0.01
print("Expected interval (s): %.2f" % expected_int)
funky_intervals = np.where(frame_ints >= expected_int+fudge_amount)[0]
print("Found %i frame grabs with interval >= %.2f" % (len(funky_intervals), expected_int+fudge_amount))

window = 200

ax = pl.subplot2grid((2, 2), (1, 0), colspan=1)
ax.set_title("Grab rate << expected")
for i in funky_intervals:
    ax.plot(np.arange(0, window*2), frame_ints[i-window:i+window], label='%.2fs (idx %i)' % (frame_ints[i], i))
ax.legend()
ax.set_ylabel('time between frames (s)')
ax.set_xlabel('centered frame intervals')


#%

fudge_amount = 0.01
print("Expected interval (s): %.2f" % expected_int)
small_intervals = np.where(frame_ints <= expected_int-fudge_amount)[0]
print("Found %i frame grabs with interval <= %.2f" % (len(small_intervals), expected_int-fudge_amount))
small_ixs = [i for i in np.where(np.diff(small_intervals)>1)[0]]
if len(small_ixs) == 0:
    small_ixs = [small_intervals[-1]]
window = min([small_intervals[small_ixs], 50])[0]

ax = pl.subplot2grid((2, 2), (1, 1), colspan=1)
ax.set_title("Grab rate >> expected")
for i in small_ixs:
    center_ix = small_intervals[i]
    if center_ix < 50:
        window_tail = 100
    else:
        window_tail = window
    nplot = window + window_tail
        
    ax.plot(np.arange(0, nplot), frame_ints[center_ix-window:center_ix+window_tail], label='%.2fs (idx %i)' % (frame_ints[center_ix], center_ix))
ax.legend()
ax.set_ylabel('time between frames (s)')
ax.set_xlabel('centered frame intervals')




#%%
cam_ints = np.array([float(i)/1E9 for i in np.diff(camera_data['frame_tstamp'])])


cam_ints = np.diff(camera_data['relative_time'])

