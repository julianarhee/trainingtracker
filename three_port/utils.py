#!/usr/bin/env python2

import matplotlib as mpl
mpl.use('agg')
import os
import glob
import json
import pymworks
import re
import datautils
import copy
import math
import time

import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
import cPickle as pkl
from cPickle import PicklingError

import scipy.stats as spstats

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


class Animal():
    def __init__(self, animalid='RAT', experiment='EXPERIMENT', rootdir='/n/coxfs01/behavior-data'): #, dst_dir='/tmp'):
        self.animalid = animalid
        self.experient = experiment
        #self.processed_dir = dst_dir
        self.sessions = {}
        self.path = None
        
        cohort = str(re.findall('(\D+)', animalid)[0])
        curr_processed_dir = os.path.join(rootdir, experiment, 'cohort_data', cohort, 'processed')
        animal_datafile = os.path.join(curr_processed_dir, 'data', '%s.pkl' % animalid)
        self.path = animal_datafile
        
    def get_datafile_list(self, metadata):
        return metadata[metadata.animalid==self.animalid]['datasource'].values
        
    def get_session_list(self, metadata):
        #self.sessions = sorted(metadata[metadata.animalid==self.animalid]['session'].values)
        return metadata[metadata.animalid==self.animalid]['session'].values
        

class Session():
    def __init__(self, session_meta):  
        self.animalid = session_meta['animalid'].unique()[0]
        self.session = session_meta['session'].unique()[0]
        self.source = sorted(session_meta['datasource'].unique(), key=natural_keys)
        self.experiment = None
        self.experiment_path = None
        self.protocol = None
        self.server = None # session_meta['server']
        self.trials = None
        self.flags = None
        self.stimuli = None
        self.counts = None
        self.summary = None
    
    def print_meta(self):
        print("Name: %s" % self.animalid)
        print("Session: %s" % self.session)
        print("Source: ", self.source)
    
    def get_trials(self, 
                     ignore_flags=None,
                     response_types=['Announce_AcquirePort1', 'Announce_AcquirePort3', 'ignore'],
                     outcome_types=['success', 'failure', 'ignore']):
                    # If there is no "abort" use "Announce_TrialEnd" ? -- these should be "aborted" trials?
        
        #### Parse trials and outcomes
        trials = []
        flags = {}
        if isinstance(self.source, list) and len(self.source) > 1:
            tmp_flags = []
            for dfn in self.source:
                curr_trials, curr_flags, df = parse_trials(dfn, response_types=response_types, 
                                                       outcome_types=outcome_types,
                                                       ignore_flags=ignore_flags)
                if curr_trials is not None:
                    trials.extend(curr_trials)
                    tmp_flags.append(curr_flags)

            # Combine flag values across data files:
            if len(tmp_flags) > 0:
                flags = dict((fkey, []) for fkey in tmp_flags[0].keys())
                for tmp_flag in tmp_flags:
                    for flag_key, flag_value in tmp_flag.iteritems():
                        if flag_key not in flags.keys():
                            flags[flag_key] = []
                        if flag_value not in flags[flag_key]:
                            flags[flag_key].append(flag_value)
        else:
            # Open data file:
            dfn = self.source[0]
            trials, flags, df = parse_trials(dfn, response_types=response_types, 
                                         outcome_types=outcome_types,
                                         ignore_flags=ignore_flags)
        self.trials = trials
        self.flags = flags

        #### Get current session server info while df open:
        server_address = df.get_events('#serialBridgeAddress')[-1].value
        server_name = df.get_events('#serverName')[-1].value
        self.server =  {'address': server_address, 'name': server_name}
        
        #### Save experiment and protocol info:
        # [(payload_type, event_type)]:  [(4013, 1002), (2001, 1001), (4007, 1002)] # 1002:  Datafile creation
        experiment_load = 4013 #:  Looks like which experiment file(s) loaded (.mwk)
        protocol_load = 2001 #:  Which protocols found and which loaded
        sys_evs = df.get_events('#systemEvent')
        
        #### Get experiment name:
        exp_evs = [v for v in sys_evs if v.value['payload_type']==experiment_load]
        exp_path = list(set([v.value['payload']['experiment path'] for v in exp_evs]))
        #assert len(exp_path) == 1, "*ERROR* More than 1 experiment loaded..."
        #exp_path = exp_path[0].split('/Experiment Cache/')[1]
        self.experiment_path = exp_path

        #### Get protocol protocol name:
        prot_evs = [v for v in sys_evs if v.value['payload_type']==protocol_load]
        protocol = list(set([v.value['payload']['current protocol'] for v in prot_evs]))
        #assert len(protocol) == 1, "*ERROR* More than 1 protocol loaded..."
        #protocol = protocol[0]
        self.protocol = protocol

        return trials
    
    def print_session_summary(self):
        if self.experiment_path is None:
            print("No trials loaded. Run:  parse_trials()")
        else:        
            print("Experiment: %s" % (self.experiment_path))
            print("Protocol: %s [server: %s]" % (self.protocol, self.server['name']))
            print("N trials: %i" % len(self.trials))
       
        return

    def get_counts_by_stimulus(self):
        print("... Getting stimulus counts ...")
        counts = None
        if self.trials is not None:
            
            by_stim = datautils.grouping.group(self.trials, 'name')
            ordered_stim = sorted(by_stim.keys(), key=lambda x: x.split('_')[-1][1:])

            counts = dict((stim, {}) for stim in by_stim.keys())
            for stim in by_stim.keys():
                counts[stim]['ntrials'] = len(by_stim[stim])
                counts[stim]['nsuccess'] = sum([1 if trial['outcome']=='success' else 0 for trial in by_stim[stim]])
                counts[stim]['nfailure'] = sum([1 if trial['outcome']=='failure' else 0 for trial in by_stim[stim]])
                counts[stim]['nignore'] = sum([1 if trial['outcome']=='ignore' else 0 for trial in by_stim[stim]])
                counts[stim]['nchoose1'] = sum([1 if trial['response']=='Announce_AcquirePort1' else 0 for trial in by_stim[stim]])
                counts[stim]['nchoose3'] = sum([1 if trial['outcome']=='Announce_AcquirePort3' else 0 for trial in by_stim[stim]])

            # Save stimulus names for easy access:
            stimulus_names = list(set([trial['name'] for trial in self.trials]))
            self.stimuli = stimulus_names
        self.counts = counts 
        
        return counts

    def get_summary(self):
        print("... Getting session summary ...")
        
        # Get counts by stim, if not run:
        if self.counts is None:
            self.get_counts_by_stimulus()
        
        if self.counts is not None:
            summary_keys = ['ntrials', 'nsuccess', 'nfailure', 'nignore']
            summary = dict((k, 0) for k in summary_keys)
            if len(self.counts.keys()) > 0:
                for stat in summary.keys():
                    summary[stat] = [val[stat] for stim, val in self.counts.items()]
            self.summary = summary
        
    def plot_performance_by_transform(self, save_dir='/tmp'):
        print("... Plotting performance by transform ...")
        counts = self.counts
        
        # Check if there are morphs also:
        morph_list = [s for s in counts.keys() if 'morph' in s]
        if len(morph_list) > 0:
            stimulus_list = [s for s in counts.keys() if s not in morph_list]
        else:
            stimulus_list = counts.keys()
            
        # Check stimulus name:
        blob_names = [b for b in self.stimuli if 'Blob_' in b]
        if 'N' in blob_names[0].split('_')[1]: # this is Blob_Nx_CamRot naming scheme:
            blob1_name = 'Blob_N1'
            blob2_name = 'Blob_N2'
        else:
            blob1_name = 'Blob_1'
            blob2_name = 'Blob_2'
            
        if counts is not None:
            values = [('%s_%s' % ('_'.join(stim.split('_')[0:2]), stim.split('_')[3]), \
                       counts[stim]['nsuccess']/float(counts[stim]['ntrials'])) for stim in stimulus_list]
            print values

            pl.figure()
            if 'CamRot' in stimulus_list[0]:
                blob1 = [(int(v[0].split('_')[-1][1:]), v[1]) for v in values if blob1_name in v[0]]
                blob2 = [(int(v[0].split('_')[-1][1:]), v[1]) for v in values if blob2_name in v[0]]
            else:
                blob1 = [(int(v[0].split('_')[-1]), v[1]) for v in values if blob1_name in v[0]]
                blob2 = [(int(v[0].split('_')[-1]), v[1]) for v in values if blob2_name in v[0]]
                
            pl.plot([b[0] for b in sorted(blob1, key=lambda x: x[0])], \
                    [b[1] for b in sorted(blob1, key=lambda x: x[0])], 'ro', label=blob1_name)
            pl.plot([b[0] for b in sorted(blob2, key=lambda x: x[0])], \
                    [b[1] for b in sorted(blob2, key=lambda x: x[0])], 'bo', label=blob2_name)
            pl.legend()
            pl.ylim(0, 1)
            pl.title('%s, %s' % (self.animalid, self.session)) #datestr)
            pl.savefig(os.path.join(save_dir, '%s_%s_bystim.png' % (self.animalid, self.session)))
            pl.close()

    def plot_performance_by_morph(self, save_dir='/tmp'):
        print("... Plotting performance by morph ...")

        counts = self.counts
            
        if counts is not None:
            stimulus_list = [s for s in counts.keys() if 'morph' in s]
            values = [('morph_%s' % (stim.split('morph')[1]), \
                       counts[stim]['nchoose1']/float(counts[stim]['ntrials'])) for stim in stimulus_list]

            pl.figure()
            pchoose1 = [(int(v[0].split('_')[-1]), v[1]) for v in values]
            pl.plot([b[0] for b in sorted(pchoose1, key=lambda x: x[0])], \
                    [b[1] for b in sorted(pchoose1, key=lambda x: x[0])], 'bo')
            pl.ylim(0, 1)
            pl.ylabel("perc. choose port 1)")
            pl.title(self.session) #datestr)
            pl.savefig(os.path.join(save_dir, '%s_%s_morphs.png' % (self.animalid, self.session)))
            pl.close()


def parse_datafile_name(dfn):
    '''
    Generally expects:
        ANIMALID_YYYYMMDD_stuff.mwk
        ...
    '''
    fn = os.path.splitext(os.path.split(dfn)[-1])[0]
    fparts = fn.split('_')

    assert len(fparts) >= 2, "*Warning* Unknown naming fmt: %s" % str(fparts)
    animalid = fparts[0]
    datestr = re.search('(\d+)', fparts[1]).group(0)

    # Make sure no exra letters are in the datestr (for parta, b, etc.)
    if not datestr.isdigit():
        datestr = re.split(r'\D', datestr)[0] # cut off any letter suffix
    if len(datestr) == 6:
        session = '20%s' % datestr 
    elif len(datestr) == 8:
        session = datestr 

    return animalid, session


def get_run_time(df):
    '''
    When was the session running?
    '''
    state_modes = df.get_events('#state_system_mode')
    run_bounds = None
    
    while True:
        try:
            running = next(d for d in state_modes if d.value==2)
            start_time = running.time
            strt = state_modes.index(running)
        except StopIteration:
            return run_bounds

        try:
            stopping = next(d for d in state_modes[strt:] if d.value != 2) 
            end_time = stopping.time
            stp = state_modes.index(stopping)
        except StopIteration:
            end_time = df.get_maximum_time()
            stp = 0

        if run_bounds is None:
            run_bounds = []
        run_bounds.append((start_time, end_time))

        # Check if there are additional run chunks:
        remaining_state_evs = state_modes[stp:]
        additional_starts = [s for s in remaining_state_evs if s.value == 2]
        if len(additional_starts) > 0:
            state_modes = remaining_state_evs
        else:
            break
        
    return run_bounds


def process_sessions_mp(new_sessions, session_meta, dst_dir='/tmp',
                         nprocesses=1, plot_each_session=True,
                         ignore_flags=None,
                         response_types=['Announce_AcquirePort1', 'Announce_AcquirePort3', 'ignore'],
                         outcome_types = ['success', 'ignore', 'failure'], create_new=False):
    
    #print "Saving output to:", dst_dir    
    def parser(curr_sessions, session_meta, ignore_flags, response_types, outcome_types, dst_dir, create_new, plot_each_session, out_q):
        parsed_sessions = {}
        for session in curr_sessions:
            curr_sessionmeta = session_meta[session_meta.session==session] #session_info[datestr]
            print(curr_sessionmeta)
            S = process_session(curr_sessionmeta, 
                                dst_dir=dst_dir,
                                ignore_flags=ignore_flags,
                                response_types=response_types, 
                                outcome_types=outcome_types,
                                create_new=create_new, plot_each_session=plot_each_session)
            parsed_sessions[session] = S
        out_q.put(parsed_sessions)
    
    # Get a chunksize of sessions to process and queue for outputs:
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(new_sessions) / float(nprocesses)))
    procs = []
    for i in range(nprocesses):
        p = mp.Process(target=parser,
                      args=(new_sessions[chunksize * i:chunksize * (i + 1)],
                           session_meta, 
                           ignore_flags,
                           response_types,
                           outcome_types,
                           dst_dir,
                           create_new,
                           plot_each_session,
                           out_q))
        procs.append(p)
        p.start()
        
    # Collect all results into single dict:
    processed_dict = {}
    for i in range(nprocesses):
        processed_dict.update(out_q.get())
    
    # Wait for all worker processes to finish:
#     for p in procs:
#         print "Finished:", p
#         p.join()
    TIMEOUT = 60
    start = time.time()
    while time.time() - start <= TIMEOUT:
        if any([p.is_alive() for p in procs]):
            time.sleep(.1)
        else:
            break # all processes complete, break
    else:
        # kill processes if time out
        print("timed out... killing all processes.")
        for p in procs:
            p.terminate()
            p.join()
            
        
    return processed_dict

def process_session(session_meta, dst_dir='/tmp',
                    response_types=['Announce_AcquirePort1', 'Announce_AcquirePort3', 'ignore'],
                    outcome_types=['success', 'ignore', 'failure'],
                    ignore_flags=None, create_new=False, plot_each_session=True):

    # Create session object from meta (quick)
    S = Session(session_meta) 
   
    # Create output dir for processed data
    src_dir = S.source[0] if isinstance(S.source, list) else S.source

    if dst_dir is None or dst_dir == '/tmp':
        dst_dir = os.path.join(S.source[0].split('/raw')[0], 'processed')
        dst_dir_figures = os.path.join(dst_dir, 'figures')
        if not os.path.exists(dst_dir_figures):
            os.makedirs(dst_dir_figures)
    print("Saving processed output to: %s" % dst_dir)

    # Check if session data exists
    tmp_file_dir = os.path.join(dst_dir_figures, 'tmp_files')
    if not os.path.exists(tmp_file_dir): os.makedirs(tmp_file_dir)
    tmp_processed_file = os.path.join(tmp_file_dir, 'proc_%s_%s.pkl' % (S.animalid, S.session))
    parse_data=False
    if os.path.exists(tmp_processed_file) and (create_new is False):
        print("... loading existing parsed session file")
        try:
            with open(tmp_processed_file, 'rb') as f:
                S = pkl.load(f)
        except ImportError:
            parse_data=True
    else:
        parse_data = True

    if parse_data or create_new:
        S.get_trials(response_types=['Announce_AcquirePort1', 'Announce_AcquirePort3', 'ignore'], \
                     outcome_types = ['success', 'ignore', 'failure'])
        S.get_summary() #S.get_counts_by_stimulus()
            
        # Save tmp file:
        with open(tmp_processed_file, 'wb') as f:
            pkl.dump(S, f, protocol=pkl.HIGHEST_PROTOCOL)
     
    if S.summary is None or S.summary['ntrials'] == 0:
        print("--- no trials ---")
        return None
  
    if plot_each_session:
        print("... plotting some stats...")
        S.plot_performance_by_transform(save_dir=dst_dir_figures)

        if any('morph' in i for i in S.stimuli):
            S.plot_performance_by_morph(save_dir=dst_dir_figures)
           
    return S

def parse_trials(dfn, response_types=['Announce_AcquirePort1', 'Announce_AcquirePort3', 'ignore'], \
                 outcome_types = ['success', 'ignore', 'failure'],\
                 ignore_flags=[], remove_orphans=True):
    
    stim_blacklists = [
        lambda s: (('type' in s.keys()) and (s['type'] == 'blankscreen')),
        ]

    print "***** Parsing trials *****"
    df = pymworks.open(dfn)
    
    # Separate behavior-training flag states from current trial states
    if ignore_flags is None or len(ignore_flags)==0:
        codec = df.get_codec()
        ignore_flags = []
        all_flags = [f for f in codec.values() if 'Flag' in f or 'flag' in f]
        for fl in all_flags:
            evs = df.get_events(fl)
            vals = list(set([v.value for v in evs]))
            if len(vals) > 1 or len(evs) > 5:
                ignore_flags.append(fl)
        
    # Get run bounds:
    bounds = get_run_time(df)
    if bounds is None:
        return None, None, df

    trials = []; flag_list = []; flags = {};
    for bound in bounds:
        
        if (bound[1]-bound[0])/1E6 < 2.0:
            continue
            
        # Identify no feedback conditions
        #  231: 'nofeedback_depth_rotation_min',
        #  232: 'nofeedback_depth_rotation_max',
        #  233: 'nofeedback_size_min',
        #  234: 'nofeedback_size_max',
        if 'nofeedback_depth_rotation_min' not in df.get_codec().values():
            always_feedback = True
        else:
            always_feedback = False
            no_fb_params_tmp = df.get_events(['nofeedback_size_min',
                                           'nofeedback_size_max',
                                           'nofeedback_depth_rotation_min',
                                           'nofeedback_depth_rotation_max'])
            fb_info = list(set([(e.name, e.value) for e in no_fb_params_tmp]))
            no_fb = {}
            for fb in fb_info:
                param = '_'.join(fb[0].split('_')[1:-1])
                if param not in no_fb.keys():
                    no_fb[param] = [fb[1]]
                else:
                    no_fb[param].append(fb[1])
        
        # Get display events:
        tmp_devs = df.get_events('#stimDisplayUpdate')                     
        tmp_devs = [i for i in tmp_devs if bound[0] <= i['time']<= bound[1]] 

        # Get behavior flags:
        codec = df.get_codec()
        all_flags = [f for f in codec.values() if 'Flag' in f or 'flag' in f]
        flag_names = [f for f in all_flags if f not in ignore_flags]
        tmp_flags = dict((flag, None) for flag in flag_names)
        for flag in flag_names:
            if flag == 'FlagNoFeedbackInCurrentTrial': continue
            found_values = [e.value for e in df.get_events(flag) if bound[0] <= e.time <=bound[1]]
            if (len(found_values) > 1) or (len(list(set(found_values)))) > 1:
                print("More than 1 value found for flag: %s" % flag)
                tmp_flags[flag] = int(found_values[-1])
            elif (len(found_values) == 1) or (len(list(set(found_values)))) == 1:
                tmp_flags[flag] = int(found_values[0])
            else:
                tmp_flags.pop(flag)
        
        # Add current flag values to flags list:
        flag_list.append(tmp_flags)
        
        # Add boundary time to flag info:
        tmp_flags.update({'run_bounds': bound})
        
        # Check for valid response types and get all response events:
        response_types = [r for r in response_types if r in codec.values()]
        response_evs = [e for e in df.get_events(response_types) if e.value==1] #if (bound[0] < e['time'] < bound[1]) and e.value==1]    
        outcome_evs = [e for e in df.get_events(outcome_types) if e.value==1] #if (bound[0] < e['time'] < bound[1]) and e.value is not None and e.value != 0] #not in [0, None]]  
        print(len(response_evs), len(outcome_evs))

        # Sync response events to true outcome events:  response occurs after stimulus, stimulus is the master
        # Convert to trials: match stimulus events and response events:
        outcome_key = 'response'
        responses = to_trials(tmp_devs, response_evs, outcome_key=outcome_key,
                                                   duration_multiplier=1.,
                                                   stim_blacklists=stim_blacklists,
                                                   remove_unknown=True)

        # **sync outcome events to response events as master (direction 1=slave after master, -1=slave before master)
        #outcome_evs = [e for e in df.get_events(outcome_types) if (bound[0] < e['time'] < bound[1]) and e.value!=0]
        outcomes = pymworks.events.utils.sync(outcome_evs, responses, direction=1, mkey=lambda x: x['response_time'])

        print "N total response events: ", len(responses)
        print "N total outcome events: ", len(outcomes)

        assert len(responses) == len(outcomes), "**ERROR:  N responses (%i) != N outcomes (%i)" % (len(responses), len(outcomes))
        tmp_trials = copy.copy(responses)
        for trial_ix, (response, outcome) in enumerate(zip(responses, outcomes)):
            if outcome is not None:
                tmp_trials[trial_ix].update({'outcome': outcome.name, 'outcome_time': outcome.time}) #['outcome']})
            else:
                tmp_trials[trial_ix].update({'outcome': 'unknown'})

        # Get rid of display events without known outcome within 'duration_multiplier' time
        if remove_orphans:                                                  
            orphans = [(i,x) for i,x in enumerate(tmp_trials) if\
                        x['outcome']=='unknown' or x['%s' % outcome_key]=='unknown']
            tmp_trials = [t for t in tmp_trials if not t['outcome']=='unknown']
            tmp_trials = [t for t in tmp_trials if not t['%s' % outcome_key]=='unknown']

            print "Found and removed %i orphan stimulus events in file %s" % (len(orphans), df.filename)
            print "N valid trials: %i" % len(tmp_trials)
        
        # Add current trials in chunk to trials list:
        trials.extend(tmp_trials)

    if len(trials) == 0:
        return trials, flags, df
    
    for t in trials:
        stim_aspect = [v.value for v in df.get_events('StimAspectRatio')][-1]
        assert t['response_time'] < t['outcome_time'], "**ERROR: Mismatch in response/outcome alignment"

        # Supplement trial info
        stimname = t['name'].split(' ')[0].split('.png')[0]
        t['name'] = stimname
        # Can be: Blob_1_RotDep_0, Blob_N2_CamRot_y-45
        
        if 'RotDep' in stimname:
            drot_str = stimname.split('_')[-1]
            depthrot_value = int(drot_str)
            t['depth_rotation'] = depthrot_value #Blob_N2_CamRot_y-45
        elif 'CamRot' in stimname and 'LighPos' in stimname:
            depthrot_value = int(stimname.split('CamRot_y')[1].split('_')[0])
            lightpos_value = tuple([int(i) for i in re.findall("[-\d]+", stimname.split('LighPos')[1])])
            t['depth_rotation'] = depthrot_value #Blob_N2_CamRot_y-45
            t['light_position'] = lightpos_value
        elif 'CamRot' in stimname:
            depthrot_value = int(stimname.split('CamRot_y')[1].split('_')[0])
            t['depth_rotation'] = depthrot_value #Blob_N2_CamRot_y-45
        elif 'morph' in stimname:
            t['depth_rotation']=0
            
#         if 'y' in drot_str:
#             depthrot_value = int(drot_str[1:])
#         else:
#             depthrot_value = int(drot_str)
                    
        t['size'] = round(t['size_x']/stim_aspect, 1)

        # Check if no feedback
        if always_feedback:
            t['no_feedback'] = True
        else:
            t['no_feedback'] = all([np.min(lims) < t[k] < np.max(lims) for k, lims in no_fb.items()])

    # Combine all flag states:
    for fi, flag_dict in enumerate(flag_list):
        if fi == 0:
            flags = copy.copy(flag_dict)
        else:
            for flag_name, flag_value in flag_dict.items():
                existing_value = flags[flag_name] #.value()
                if flag_value == existing_value:
                    continue
                if not isinstance(flags[flag_name], list):
                    flags[flag_name] = list(flags[flag_name])
                flags[flag_name].append(flag_value)


    return trials, flags, df

def to_trials(stim_display_events, outcome_events, outcome_key='outcome',
              remove_unknown=True,
              duration_multiplier=2, stim_blacklists=None):
    """
    If remove_unknown, any trials where a corresponding outcome_event cannot
    be found will be removed.
    
    If duration_multiplier is not None, to_trials() will check to see if the
    outcome event occured within duration_multiplier * duration microseconds
    of the trial start. If the outcome event occured later, the trial outcome
    will be marked as unknown.

    pymworks.events.utils.sync(slave, master, direction=-1, skey=None, mkey=None)
    
    Find closest matching slave event for each master event.
    direction : int, -1, 0 or 1
        if -1, slave events occur before master events
        if  1, slave events occur after master events
        if  0, slave and master events occur simultaneously


    """
    if (len(outcome_events) == 0) or (len(stim_display_events) == 0):
        return []
    assert hasattr(outcome_events[0], 'name')

    trials = pymworks.events.display.to_stims(stim_display_events, as_dicts=True,
                      blacklist=stim_blacklists)

    if (len(trials) == 0):
        return []

    outcomes = pymworks.events.utils.sync(outcome_events, trials,
                          direction=1, mkey=lambda x: x['time'], skey=lambda x: x.time)
#     outcomes = pymworks.events.utils.sync(trials, outcome_events,
#                           direction=-1, mkey=lambda x: x.time) #['time'])

    assert len(trials) == len(outcomes), "%i trials for %i outcomes..." % (len(trials), len(outcomes))
    unknowns = []
    if duration_multiplier is None:
        dtest = lambda t, o: True
    else:
        dtest = lambda t, o: \
            o.time < (t['time'] + t['duration'] * duration_multiplier)
    for i in xrange(len(trials)):
        if (outcomes[i] is not None) and dtest(trials[i], outcomes[i]):
            trials[i]['%s' % outcome_key] = outcomes[i].name
            trials[i]['%s_time' % outcome_key] = outcomes[i].time
        else:
            if remove_unknown:
                unknowns.append(i)
            else:
                trials[i]['%s' % outcome_key] = 'unknown'
                trials[i]['%s_time' % outcome_key] = 'unknown'

    # remove trials with 'unknown' outcome, in reverse
    for u in unknowns[::-1]:
        del trials[u]

    return trials

