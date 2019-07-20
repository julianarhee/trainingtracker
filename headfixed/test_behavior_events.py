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

data_dir = '/home/julianarhee/Documents/behavior/headfixed/test_training_phase'

#% Load MW behavior data:
mw_fname = 'test_phase1'
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


data_save_intervals = np.diff(camera_data['relative_time'])


pl.figure()
pl.hist(data_save_intervals, alpha=0.5, color='blue')
pl.hist(trigger_receive_intervals, alpha=0.5, color='orange')
pl.hist(camera_frame_intervals, alpha=0.5, color='green')


#%%
expected_int = 1/fr

pl.figure()
pl.plot(trigger_receive_intervals)
pl.xlabel('frame intervals')
pl.ylabel('time betwen frames (s)')
pl.title("Receiver: expected frame interval: %.2f s (%.1f Hz)" % (expected_int, fr))


#%%


