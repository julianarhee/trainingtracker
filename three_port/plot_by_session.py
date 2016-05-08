#!/usr/bin/env python2


import os
import multiprocessing
import datetime
import pymworks
import matplotlib.pyplot as plt

import sys
import cPickle as pkl
import re

import plotly.plotly as py
import plotly.tools as tls
# import plotly.graph_objs as go
import numpy as np
from plotly.graph_objs import *


from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.offline import plot


# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]



# processed_path='/share/coxlab-behavior/mworks-data/three_port_morphs/nurbs/processed'
# processed_path='/share/coxlab-behavior/mworks-data/three_port_morphs/gabors/processed'
processed_path = sys.argv[1]

stim_condition = os.path.split(os.path.split(processed_path)[0])[1] # GABS or NURBS

animals = [i for i in os.listdir(processed_path) if os.path.isdir(os.path.join(processed_path, i))]
counts_by_animal = dict()

for animal in animals:
    animal_path = os.path.join(processed_path, animal)
    session_list = sorted([i for i in os.listdir(animal_path) if os.path.splitext(i)[1] == '.pkl'], key=natural_keys)
    sessions = [s.split('_')[1].split('.')[0] for s in session_list]
    counts = np.zeros((len(session_list),3))
    for sidx, session in enumerate(session_list):
        fn = os.path.join(processed_path, animal, session)
        with open(fn, 'rb') as f:
            curr_session = pkl.load(f)

        ntrials = len(curr_session)
        success_idxs = [i for (i,trial) in enumerate(curr_session) if trial['behavior_outcome']=='success']
        ignore_idxs = [i for (i,trial) in enumerate(curr_session) if trial['behavior_outcome']=='ignore']
        failure_idxs = [i for (i,trial) in enumerate(curr_session) if trial['behavior_outcome']=='failure']

        # plt.figure()
        # plt.plot(success_idxs, np.ones(len(success_idxs)), 'ro', label='success')
        # plt.plot(ignore_idxs, np.ones(len(ignore_idxs))*-1, 'yo', label='ignore')
        # plt.plot(failure_idxs, np.zeros(len(failure_idxs)), 'ko', label='failure')
        # plt.legend()

        counts[sidx,:] = [len(success_idxs), len(failure_idxs), len(ignore_idxs)]

    # pidxs = np.arange(len(counts)) # nsessions

    counts_by_animal[animal] = counts

    # mpl_fig = plt.figure()
    # ax = mpl_fig.add_subplot(111)

    # width = 0.35
    # p1 = ax.bar(pidxs, [i[0] for i in counts], width, color=(0.2588,0.4433,1.0))
    # p2 = ax.bar(pidxs, [i[1] for i in counts], width, color=(1.0,0.5,0.62))
    # p3 = ax.bar(pidxs, [i[2] for i in counts], width, color=(0.5,1.0,0.62))

    # ax.set_ylabel('outcomes')
    # ax.set_xlabel('sessions')
    # ax.set_title('%s: outcomes by session' % animal)


    # ax.set_xticks(pidxs + width/2.)
    # # ax.set_yticks(np.arange(0, 81, 10))
    # ax.set_xticklabels(tuple(sessions))

    # plotly_fig = tls.mpl_to_plotly( mpl_fig )

    # # For Legend
    # plotly_fig["layout"]["showlegend"] = True
    # plotly_fig["data"][0]["name"] = "success"
    # plotly_fig["data"][1]["name"] = "failure"
    # plotly_fig["data"][2]["name"] = "ignore"


    # plot_url = py.plot(plotly_fig, filename='stacked-bar-chart')

    # p1 = go.Bar(x=sessions, y=[i[0] for i in counts])
    # p2 = go.Bar(x=sessions, y=[i[1] for i in counts])
    # p3 = go.Bar(x=sessions, y=[i[2] for i in counts])

    # data = [trace1, trace2]


fig = tls.make_subplots(rows=4, cols=1, subplot_titles=(counts_by_animal.keys()))
plot_data = dict()
plot_layout = dict()

subplot_idx = 1
for a in counts_by_animal.keys():

    print("CURRENT: %s") % a

    counts = counts_by_animal[a]


    successes = [i[0] for i in counts]
    failures = [i[1] for i in counts]
    ignores = [i[2] for i in counts]

    # Define a trace-generating function (returns a Bar object)
    def make_trace(y, name, color, subplot):
        
        curr_y = 'y%i' % int(subplot)
        curr_x = 'x%i' % int(subplot)
        if subplot==4:
            legend_flag = True
        else:
            legend_flag = False

        return Bar(
            x=np.arange(len(counts)),       # (!) x-coords are the summer month names (global variable)
            y=y,            # take in the y-coordinates
            name=name,      # label for legend/hover
            marker=Marker(
                color=color,        # set bar colors
                line=Line(
                    color='white',  # set bar border color
                    width= 2.5      # set bar border width
                )
            ),
            xaxis=curr_x,
            yaxis=curr_y,
            showlegend=legend_flag

        )

    # (1) Make Data object using make_trace()
    # data = Data([
    #     make_trace(successes, 'success', '#BD8F22'),
    #     make_trace(failures, 'failure', '#E3BA22'),
    #     make_trace(ignores, 'ignore', '#F2DA57')
    # ])
    data = [
        make_trace(successes, 'success', '#BD8F22', subplot_idx),
        make_trace(failures, 'failure', '#E3BA22', subplot_idx),
        make_trace(ignores, 'ignore', '#F2DA57', subplot_idx)
        ]

    # Define an annotation-generating function
    def make_annotation(x, y):
        return Annotation(
            text=str(y),     # text is the y-coord
            showarrow=False, # annotation w/o arrows, default is True
            x=x,               # set x position
            xref='x',          # position text horizontally with x-coords
            xanchor='center',  # x position corresp. to center of text
            yref='y',            # set y position 
            yanchor='top',       # position text vertically with y-coords
            y=y,                 # y position corresp. to top of text
            font=Font(
                color='#262626',  # set font color
                size=13           #   and size   
            )
        )

    # Make Annotations object (list-like) with make_annotation()
    # annotations = Annotations(
    #     [make_annotation(x, y) for x, y in zip(range(3), successes)] +
    #     [make_annotation(x, y) for x, y in zip(range(3), failures)] +
    #     [make_annotation(x, y) for x, y in zip(range(3), ignores)]
    # )
    annotations = Annotations(
        [make_annotation(x, i) for x, y in zip(range(len(counts)), counts) for i in y]
    )


    # title = '%s outcomes by session' % a # plot's title  

    # # (2) Make Layout object
    # layout = Layout(
    #     barmode='overlay',  # (!) bars are overlaid on this plot
    #     title=title,        # set plot title
    #     yaxis=YAxis(
    #         zeroline=False,          # no thick y=0 line
    #         showgrid=False,          # no horizontal grid lines
    #         showticklabels=False     # no y-axis tick labels
    #     ),
    #     legend=Legend(
    #         x=0,     # set legend x position in norm. plotting area coord.  
    #         y=1,     # set legend y postion in " " " "
    #         yanchor='middle'   # y position corresp. to middle of text
    #     )
    #     #annotations=annotations # link the Annotations object
    # )


    # (3) Make Figure object
    # fig = Figure(data=data, layout=layout)

    plot_data[a] = data

    subplot_idx += 1

fig['data'] = Data(
    plot_data[plot_data.keys()[0]] + \
    plot_data[plot_data.keys()[1]] + \
    plot_data[plot_data.keys()[2]] + \
    plot_data[plot_data.keys()[3]])
# fig['data'] = [plot_data['JR001B'], plot_data['JR002B'], plot_data['JR003B'], plot_data['JR004B']]


# fig = Figure(data=[plot_data['JR001B'], plot_data['JR002B'], plot_data['JR003B'], plot_data['JR004B']])

# 

fig_title= 'Behavior outcomes by animal - %s' % stim_condition

fig['layout'].update(
    barmode='overlay',
    height=1200, 
    width=600, 
    title=fig_title

    )

# fig['layout'].update(

#     barmode='overlay',  # (!) bars are overlaid on this plot
#     title=title,        # set plot title
#     yaxis=YAxis(
#         zeroline=False,          # no thick y=0 line
#         showgrid=False,          # no horizontal grid lines
#         showticklabels=False     # no y-axis tick labels
#     )
#     )


    # (@) Send to Plotly and show in notebook
# py.plot(fig, filename='behav_%s' % stim_condition)
#py.image.save_as({'data':data}, 'scatter_plot', format='png')
plot(fig, filename='behav_%s' % stim_condition)


# from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
# from plotly.offline import plot