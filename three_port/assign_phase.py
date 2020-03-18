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

import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
import cPickle as pkl
from cPickle import PicklingError

import scipy.stats as spstats
import utils as util
import process_datafiles as processd

import pprint

pp = pprint.PrettyPrinter(indent=4)

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

    
def get_default_params(cohort):
    # array(['', 'b', 'big', 'stimB', 'stimBlowerbound', 'stimClowerbound',
    #        'stimBupperbound', 'stimCupperbound', 'small', 'stimC', 'c',
    #        'stimCalwaysReward', 'alwaysReward', 'size', 'adeptrotl',
    #        'bdeptrotr', 'deptrotl', 'adeptrotr', 'asize', 'bdeptrotl',
    #        'background', 'deptrotr', 'balwaysReward', 'initial', 'deprotl',
    #        'backgroundalwaysReward', 'bsize', 'morphs', 'aalwaysReward',
    #        'empty', 'aasize', 'a', 'probeNewObjects', 'occluded', 'r', 'd',
    #        'transparency', 'nominrt'], dtype=object)


    phase_lookup = {0: 'manual',
                    1: 'default',
                    2: 'size',
                    3: 'depth_rotation',
                    4: 'cross',
                    5: 'size_and_depth_rotation',
                    6: 'depth_and_planar_rotation',
                    7: 'morph',
                    8: 'newstimuli',
                    9: 'fine_grained_size',
                    10: 'fine_grained_depth_rotation',
                    11: 'fine_grained_size_and_depth_rotation',
                    12: 'transparency',
                    13: 'clutter',
                    -1: 'other'}

    default_depth_rotation = 0.
    default_planar_rotation = 0.

    if cohort in ['AK', 'AL', 'AM']:
        expected_sizes = np.linspace(15, 40, 11.)
        expected_drots = np.linspace(-60, 60, 25.)
        default_size = 40

    elif cohort in ['AG', 'AJ']:
        expected_sizes = np.linspace(15, 40, 6.)
        expected_drots = np.linspace(-60, 60, 9.)
        default_size = 30

    expected_size_interval = np.diff(expected_sizes).mean()
    expected_drot_interval = np.diff(expected_drots).mean()

    defaults = {'size': default_size,
                'depth_rotation': default_depth_rotation,
                'planar_rotation':  default_planar_rotation,
                'expected_sizes': expected_sizes,
                'expected_depth_rotations': expected_drots}
    
    return defaults


def assign_phase_to_datafile(cohort, metadata, paradigm='threeport', rootdir='/n/coxfs01/behavior-data'):

    new_stim_descs = ['stimB', 'stimC', 'NewObjects']

    defaults = get_default_params(cohort)
    default_size = defaults['size']
    default_depth_rotation = defaults['depth_rotation']
    default_planar_rotation = defaults['planar_rotation']

    expected_sizes = defaults['expected_sizes']
    expected_drots = defaults['expected_depth_rotations']
    expected_size_interval = np.diff(expected_sizes).mean()
    expected_drot_interval = np.diff(expected_drots).mean()
    
    exclude_ixs = []
    phasedata = []
    animal_meta = metadata[metadata['cohort']== cohort]

    for (animalid, dfn), mgroup in animal_meta.sort_values(by=['animalid', 'session']).groupby(['animalid', 'datasource']):

        phase = -1
        (animalid, cohort, dsource, session, sfx), = mgroup.values

        curr_trials, curr_flags, metainfo = util.parse_mw_file(dfn, create_new=False)

        if curr_trials is None or len(curr_trials)==0:
            exclude_ixs.extend(mgroup.index.tolist())
            continue

        sizes = sorted(np.unique([t['size'] for t in curr_trials]))
        drots = sorted(np.unique([t['depth_rotation'] for t in curr_trials]))
        prots = sorted(np.unique([t['rotation'] for t in curr_trials]))

        tested_size_interval = np.diff(sizes).mean()
        tested_drot_interval = np.diff(drots).mean()

        protocol = metainfo['protocol']
        experiment_folder = metainfo['experiment']

        alpha_values = []
        if any(['alpha_multiplier' in s.keys() for s in curr_trials]):
            alpha_values = np.unique([s['alpha_multiplier'] for s in curr_trials])

    #     if session == 20180625:
    #         break

        if 'morph' in protocol:
            phase = 7

        elif 'newstim' in metainfo['experiment'] or any([descr in sfx for descr in new_stim_descs]):
            phase = 8

        elif len(alpha_values) > 1 or any([a != 1 for a in alpha_values]):
            phase = 12

        elif any(['background' in desc for desc in [sfx, metainfo['experiment']]]):
            phase = 13

        elif 'occluded' in sfx:
            phase = 13

        # ====== PHASE 1 ====================================
        elif len(sizes) == 1 and len(drots) == 1 and len(prots)==1:
            #if protocol in ['Initiate VIsual Pres Protocol', 'Initiate Visual Pres Protocol', '']:
            if (sizes[0]==default_size and drots[0]==default_depth_rotation and prots[0]==default_planar_rotation):
                phase = 1

        # ====== PHASE 2 ====================================
        elif len(sizes) > 1 and (len(drots)==1 and len(prots)==1):
            #if metainfo['protocol'] in ['Staircase through shape parameters', '']:        
            if ( (expected_sizes==sizes) is False):
                if (tested_size_interval < expected_size_interval):
                    phase = 9
                elif (tested_size_interval==expected_size_interval):
                    if (len(sizes) > len(expected_sizes)-1):
                        # Probably just screwed up indexing
                        phase = 2
                    elif all([d in expected_sizes for d in sizes]) : 
                        # only testing a subset of the expected values
                        phase = 2
            elif all(expected_sizes==sizes):
                phase = 2

        # ====== PHASE 3 ====================================
        elif len(drots) > 1 and (len(sizes)==1 and len(prots)==1):
            #if metainfo['protocol'] in ['Staircase through shape parameters', '']:
            if ( (expected_drots==drots) is False):
                if (tested_drot_interval < expected_drot_interval):
                    # Fine-grained spacing
                    phase = 10
                elif (tested_drot_interval==expected_drot_interval):
                    if (len(drots) > len(expected_drots)-1):
                        # Probably just screwed up indexing
                        phase = 3
                    elif all([d in expected_drots for d in drots]): 
                        # only testing a subset of the expected values
                        phase = 3
            elif all(expected_drots==drots):
                phase = 3

        # ====== PHASE 4/5 ====================================
        elif len(drots) > 1 and len(sizes) > 1 and len(prots) == 1:
            #if metainfo['protocol'] in ['Test all transformations', '']:
            off_cross_transforms = list(set([(t['size'], t['depth_rotation']) for t in curr_trials \
                                             if t['size']!=default_size \
                                             and t['depth_rotation']!=default_depth_rotation]))

            if ( (expected_drots==drots) is False) or ( (expected_sizes==sizes) is False):
                if (tested_size_interval < expected_size_interval) or (tested_drot_interval < expected_drot_interval):
                    # Fine-grained size/rotation transformations
                    phase = 11
    #             elif (len(drots) == len(expected_drots)-1) or (len(sizes) == len(expected_sizes)-1):
    #                 # Probably just screwed up indexing
    #                 if len(off_cross_transforms) == 0:
    #                     phase = 4 if len(off_cross_transforms) == 0 else 5
    #             elif ((expected_drots==drots) is False):
    #                 if all(sizes==expected_sizes):
    #                     # Another screw up, just not changing 1 of the depth-rot sides
    #                     phase = 4 if len(off_cross_transforms)==0 else 5
    #             elif ((expected_sizes==sizes) is False):
    #                 if all(drots==expected_drots):
    #                     # Another screw up, just not changing 1 of the sizes
    #                     phase = 4 if len(off_cross_transforms)==0 else 5       
                elif all([d in expected_drots for d in drots]) and all([s in expected_sizes for s in sizes]):
                    phase = 4 if (len(off_cross_transforms)==0) else 5
            else:
                # STANDARD
                if all(drots==expected_drots) and all(sizes==expected_sizes):
                    phase = 4 if (len(off_cross_transforms)==0) else 5

        # ====== PHASE 6 (in-plane) ====================================
        elif len(prots) > 1 and len(sizes)==1:
            if metainfo['protocol'] == 'Test all transformations':
                phase = 6

        else:
            print("NO CLUE.")
            print("[%s, %s%s] unknown protocol: %s" % (animalid, session, suffix, protocol) )

            print("Size", sizes)
            print("D-rots", drots)
            print("P-rots", prots)

        print(animalid, session, phase)
        mgroup['phase'] = [phase for _ in np.arange(0, len(mgroup))]
        mgroup['protocol'] = [protocol for _ in np.arange(0, len(mgroup))]
        mgroup['experiment'] = [experiment_folder for _ in np.arange(0, len(mgroup))]

        phasedata.append(mgroup)

    phasedata = pd.concat(phasedata, axis=0)


    return phasedata, exclude_ixs

    
def get_phase_data(cohort, paradigm='threeport', create_new=False, rootdir='/n/coxfs01/behavior-data'):


    metadata = util.get_metadata(paradigm, rootdir=rootdir, filtered=False, create_meta=False)

    #### Load phase info for cohort
    processed_dir = os.path.join(rootdir, paradigm, 'processed')
    phase_dfile = os.path.join(processed_dir, 'meta', 'phases_%s.pkl' % cohort)
    if os.path.exists(phase_dfile) and create_new is False:
        with open(phase_dfile, 'rb') as f:
            phasedata = pkl.load(f)
    else:
        phasedata, _ = assign_phase_to_datafile(cohort, metadata, paradigm=paradigm, rootdir=rootdir)
        with open(phase_dfile, 'wb') as f:
            pkl.dump(phasedata, f, protocol=pkl.HIGHEST_PROTOCOL)

        
    phasedata = assign_phase_to_datafile(cohort, metadata, paradigm=paradigm, rootdir=rootdir)
    
    return phasedata

def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/behavior-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/behavior-data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default=None, 
                      help='Animal ID')
    parser.add_option('-c', '--cohort', action='store', dest='cohort', default=None, 
                      help='cohort, e.g, AL)')
    parser.add_option('-p', '--paradigm', action='store', dest='paradigm', default='threeport', 
            help='paradigm used (default:  threeport')

    parser.add_option('-n', '--nproc', action='store', dest='n_processes', default=1, 
            help='N processes to use (default:  1)')
    
    parser.add_option('--new', action='store_true', dest='create_new', default=False, 
            help='flag to assign phase info anew')
    
    (options, args) = parser.parse_args(options)

    return options


def main(options):

    opts = extract_options(options)
    rootdir = opts.rootdir
    paradigm = opts.paradigm
    animalid = opts.animalid
    cohort = opts.cohort
    create_new = opts.create_new
    
    phasedata = get_phase_data(cohort, paradigm=paradigm, create_new=create_new, rootdir=rootdir)

if __name__ == '__main__':
    main(sys.argv[1:])
                  