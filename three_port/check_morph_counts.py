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

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]



# file_dir = '/home/juliana/Documents/mworks/data'
file_dir = '/share/coxlab-behavior/mworks-data/three_port_morphs/pnas/input/AG10'

files = os.listdir(file_dir)
print files

fns = sorted([i for i in files if i.endswith('mwk')], key=natural_keys)
print fns

mfns = sorted([i for i in fns if '16062' in i], key=natural_keys)
# f = fns[0]
# df = pymworks.open(os.path.join(file_dir, f))
#   evs = df.get_events('#stimDisplayUpdate')

# devs = [e for e in evs if e.value]
# im_names = [i['name'] for d in devs for i in d.value if '.png' in i['name']]
# morphs = [i for i in im_names if 'morph' in i]
# print len(morphs)
# from collections import Counter
# from pandas import Series

# fn = [i for i in fns if '_SD' in i] #[0]
# print fn
# # f = fn

for f in fns:
    # f = fn[4]
    print f
    df = pymworks.open(os.path.join(file_dir, f))

    # Check stimDisplayUpdate events vs announceStimulus:
    stim_evs = df.get_events('#stimDisplayUpdate')
    devs = [e for e in stim_evs if not e.value==[None]]
    idevs = [i for i in devs for v in i.value if 'png' in v['name']]

    ann_evs = df.get_events('#announceStimulus')
    aevs = [e for e in ann_evs if type(e.value) is dict]
    sevs = [i for i in aevs if 'png' in i.value['name']]

    if not len(idevs)==len(sevs):
        print "MISMATCH in event counts in DF: %s" % f
        print "-------------------------"
        print "#stimDisplayUpdate %i and #announceStimulus %i." % (len(idevs), len(sevs))

    # devs = [e for e in stim_evs if e.value[0] is not None]
    im_names = sorted([i['name'] for d in idevs for i in d.value if '.png' in i['name']], key=natural_keys)
    morphs = sorted([i for i in im_names if 'morph' in i], key=natural_keys)
    transforms = [i for i in im_names if i not in morphs]
    print "N morphs: ", len(morphs)
    print "N transforms: ", len(transforms)

    counts = Counter(im_names)
    print counts

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
    # for i, label in enumerate(list(df.name)):
    #     score = df.ix[i]['count']
    #     ax.annotate(str(label), (i, score)) #, (i - 0.2, score))



    plt.show()

    # s = sorted(Series(im_names), key=natural_keys)
    # vc = s.value_counts()
    # vc = vc.sort_index()
    # print vc
    # mkeys = [i for i in vc.keys() if 'morph' in i]
    # tkeys = [i for i in vc.keys() if not 'morph' in i]
    # morph_total = sum(vc[mkeys])
    # trans_total = sum(vc[tkeys])


    # plt.figure()
    # plt.subplot(2,1,1)
    # ax = vc.plot(kind='bar',alpha=0.75, rot=90)
    # plt.xlabel("")
    # plt.title('Counts of stimulus type (%.4f%% morphs)' % (float(morph_total)/sum(vc)))
    # plt.tight_layout()

    # plt.subplot(2,1,2)
    # labels=['transforms', 'morphs']
    # totals = [trans_total, morph_total]
    # width=1
    # plt.bar([1, 2], totals)
    # plt.xticks(np.array([1,2])+width*.5, labels)
    # plt.tight_layout()


    # One way to plot:
# labels, values = zip(*Counter(im_names).items())
# indexes = np.arange(len(labels))
# width = 1

# plt.bar(indexes, values, width)
# plt.xticks(indexes + width * 0.5, labels)
# plt.show()

# Another way to plot, using pandas: