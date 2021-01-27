import os
import multiprocessing
import math
import pymworks
import matplotlib.pyplot as plt
import numpy as np
import re

from datautils import grouping
import itertools

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




def get_object_rot(datadir):

    '''
    Object rotations are made by rotating blob in POVray. 
    All stimulus-rotation values will be 0. 
    Get true rotation relative to default using image path name.
    '''

    processed_dir = os.path.join(os.path.split(datadir)[0], 'processed')
    animals = os.listdir(processed_dir)
    animals = [i for i in animals if i.endswith('.pkl')]
    animals = [os.path.splitext(a)[0] for a in animals]

    for animal in animals:

        fn = os.path.join(processed_dir, animal+'.pkl')
        with open(fn, 'rb') as f:
            curr_trials = pkl.load(f)

        session_dates = []
        for t in curr_trials.keys():
            subject, date, epoch = parse_filename(t)
            session_dates.append(date)

        for t in curr_trials.keys():
            print t
            if len(curr_trials[t])==0:
                continue
            else:
                if len(curr_trials[t])==1:
                    ctrials = curr_trials[t][0]
                else:
                    ctrials = curr_trials[t]

            df = pymworks.open(os.path.join(datadir, subject, t))

            # if no morphs, move on...
            # mtrials = [i for i in ctrials if 'morph' in i['name']]
            # if len(mtrials)>0:
            #     continue

            # Get TRUE SIZES AND ROTATIONS:
            AspRat_name = 'StimAspectRatio'
            AspRat = df.get_events(AspRat_name)[-1].value
            sizs = [round(i['size_x']/AspRat) for i in ctrials]
            print set(sizs)

            stims = [i['name'] for i in ctrials]
            rots = []
            for s in stims:
                if 'Blob' in s:
                    rots.append(s.rsplit('y', 1)[1].rsplit('.',1)[0])
                else:
                    rots.append(0)

            for sz,rot,ctrial in zip(sizs, rots, ctrials):
                try:
                    ctrial['actual_rotation'] = rot
                    ctrial['actual_size'] = str(sz)
                except KeyError:
                    print ctrial

            curr_trials[t] = ctrials

        
        save_dict(curr_trials, os.path.split(fn)[0], os.path.split(fn)[1])       
        print "Saved to %s" % fn



import cPickle as pkl

# import plotly.plotly as py
# import plotly.graph_objs as go

def group_by_size_and_rot(datadir, show_plot=True):

    processed_dir = os.path.join(os.path.split(datadir)[0], 'processed')
    animals = os.listdir(processed_dir)
    animals = [i for i in animals if i.endswith('_trials.pkl')]
    animals = [os.path.splitext(a)[0] for a in animals]

    G = dict()
    for animal in animals:

        G[animal] = dict()

        fn = os.path.join(processed_dir, animal+'.pkl')
        with open(fn, 'rb') as f:
            curr_trials = pkl.load(f)

        session_dates = []
        for t in curr_trials.keys():
            subject, date, epoch = parse_filename(t)
            session_dates.append(date)


        for t in sorted(curr_trials.keys(), key=natural_keys):
            #print t
            # if len(curr_trials[t])==0:
            #     continue
            # else:
            #     if len(curr_trials[t])==1:
            #         ctrials = curr_trials[t][0]
            #     else:
            #         ctrials = curr_trials[t]

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


            # onyl calculate for NON-morph trials
            ctrials = [i for i in ctrials if 'morph' not in i['name']]
            # if len(mtrials)>0:
            #     continue

            # Funky size issue:
            AspRat = 1.747
            sizes = np.array(sorted(set([i['actual_size'] for i in ctrials])))
            rots = np.array(sorted(set([i['actual_rotation'] for i in ctrials])))

            gkeys = ('actual_size','actual_rotation','outcome')
            G[animal][t] = grouping.groupn(ctrials, gkeys )

    return G




# G = group_by_size_and_rot(datadir)

def create_datamat(G):

    data = dict()
    for animal in G.keys():
        
        sizes = list(set(list(itertools.chain.from_iterable([G[animal][i].keys() for i in G[animal].keys()]))))
        rots = list(set(list(itertools.chain.from_iterable([G[animal][i][j].keys() for i in G[animal].keys() for j in G[animal][i].keys()]))))

        C = grouping.ops.collapse(G[animal], 1) # collapse SIZE-ROT into one
        
        data[animal] = np.zeros((len(sizes), len(rots)))

        for sesh in C.keys():

            for cond in C[sesh].keys():

                curr_size = int(float(cond[0]))
                curr_rot = int(float(cond[1]))

                sidx = sizes.index(cond[0])
                ridx = rots.index(cond[1])

                data[animal][sidx, ridx] = float(len(C[sesh][cond]['success'])) / sum([len(C[sesh][cond][i]) for i in C[sesh][cond].keys()])

    return sizes, rots, data

# x = np.linspace(rotmin, rotmax+rotstep, rotstep) #['A', 'B', 'C', 'D', 'E']

# sizestep = 5
# y = np.linspace(sizemin, sizemax+sizestep, sizestep)

# #       x0    x1    x2    x3    x4
# z = [[0.00, 0.00, 0.75, 0.75, 0.00],  # y0
#      [0.00, 0.00, 0.75, 0.75, 0.00],  # y1
#      [0.75, 0.75, 0.75, 0.75, 0.75],  # y2
#      [0.00, 0.00, 0.00, 0.75, 0.00]]  # y3

def get_array_for_plotting(datamat, sizes, rots):

    for animal in datamat.keys():
        x = sizes
        y = rots
        z = datamat[animal]

        # annotations = []
        # for n, row in enumerate(z):
        #     for m, val in enumerate(row):
        #         var = z[n][m]
        #         annotations.append(
        #             dict(
        #                 text=str(val),
        #                 x=x[m], y=y[n],
        #                 xref='x1', yref='y1',
        #                 font=dict(color='white' if val > 0.5 else 'black'),
        #                 showarrow=False)
        #             )

        colorscale = [[0, '#3D9970'], [1, '#001f3f']]  # custom colorscale
        trace = go.Heatmap(x=x, y=y, z=z, colorscale=colorscale, showscale=False)

        fig = go.Figure(data=[trace])
        fig['layout'].update(
            title="Annotated Heatmap",
            annotations=annotations,
            xaxis=dict(ticks='', side='top'),
            # ticksuffix is a workaround to add a bit of padding
            yaxis=dict(ticks='', ticksuffix='  '),
            width=700,
            height=700,
            autosize=False
        )
        py.iplot(fig, filename='Annotated Heatmap', height=750)
