#!/usr/bin/env python

# Map trials to stimulus update events -- group by outcome/response
# Does same as 

import pymworks
import datautils
import matplotlib.pyplot as plt

import fnmatch
import functools
import logging
import os

import matplotlib
import numpy as np
import sys

def listdir_nohidden(path):
    dir_list = []
    temp_list = os.listdir(path)
    for f in temp_list:
        if not f.startswith('.'):
            dir_list.append(f)
    return dir_list

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


def get_datafiles(indir, match='*.mwk'):
    dfns = []
    for d, sds, fns in os.walk(indir):
        mfns = fnmatch.filter(fns, match)
        dfns += [os.path.join(d, fn) for fn in mfns]
    return dfns


# STATS FUNCS:

def get_session_stats():
    stats = {}
    for n in ('session_correct_lick', 'session_bad_lick', 'session_bad_ignore', \
            'session_correct_ignore', 'session_target_trialcounter', \
            'session_distractor_trialcounter'):
        stats[n] = stat(functools.partial(get_session_range, name=n))
    for n in ('targetprob', 'curr_contrast', 'phases_completed', \
            'targetprob_lower'):
        stats[n] = stat(functools.partial(get_last, name=n), \
                lambda a, b: b)
    stats['engagement'] = stat(measure_engagement, \
            lambda a, b: (a[0] + b[0], a[1] + b[1]))
    return stats

def session_incremented(evs, test=lambda a, b: b.value != (a.value - 1),
                   recurse=True):
    """
    Remove events where evs[i].value != (evs[i-1].value - 1)

    evs[i-1] occurs later in physical time than evs[i]. Re-reverse at end.
    """
    ## remove evs[i] if NEXT trial value (evs[i+1]) is more than CURR trial value ev[i] - 1.

    events = sorted(evs, key=lambda e: e.time, reverse=True)
    bad_indices = []
    bw = events
    for i in xrange(1, len(events)):
        if test(events[i - 1], events[i]):
            bad_indices.append(i)
    for i in bad_indices[::-1]: #[::-1]
        del(events[i])
    if recurse and len(bad_indices):
        bw = session_incremented(session_incremented(events, test=test, recurse=recurse))

    return sorted(bw, key=lambda e: e.time, reverse=False)


def get_session_range(df, name=''):

    """
    Get range of values (max/min), based on true incrementing, uses session_incremented().

    Modified from function get_incremented_range(), which uses pymworks function \
            remove_non_incrementing(), which removes/grabs the wrong consecutive trials.
    """

    evs = df.get_events(name)
    for ev in evs:
        if type(ev.value) != float:
            ev.value = float(ev.value)
    gevs = session_incremented(evs)

    return pymworks.stats.events.valuerange(gevs)



def get_max(df, name=''):
    return pymworks.stats.events.valuemax(df.get_events(name)).value

def get_last(df, name=''):
    return sorted(df.get_events(name), key=lambda e: e.time)[-1].value

def measure_engagement(df, name='head_input', threshold=500):
    return pymworks.stats.events.time_in_state(df.get_events(name), \
            test=lambda e: e.value < threshold)

def get_incremented_range(df, name=''):
    evs = df.get_events(name)
    gevs = pymworks.stats.events.remove_non_incrementing(evs)
    return pymworks.stats.events.valuerange(gevs)



def get_outcomes(df, name=''):
    evs = df.get_events('#stimDisplayUpdate')
    outcomes = df.get_events(name)
    ts = pymworks.events.display.to_trials(evs, outcomes)
    gts = datautils.grouping.group(ts, 'outcome')
    nOutcomes = datautils.grouping.ops.stat(gts, len)

    respkeys = [k for k in nOutcomes.keys()]

    outcome = []
    for k in respkeys:
        outcome.append(nOutcomes[k])

    return outcome

def stat(extract_function=get_outcomes, \
        combine_function=lambda a, b: a + b):
    return (extract_function, combine_function)

def get_outcome_stats():
    stats = {}
    for n in ('session_correct_lick', 'session_bad_lick', 'session_bad_ignore', \
            'session_correct_ignore'):
        stats[n] = stat(functools.partial(get_outcomes, name=n))
    for n in ('targetprob', 'curr_contrast', 'phases_completed', \
            'targetprob_lower'):
        stats[n] = stat(functools.partial(get_last, name=n), \
                lambda a, b: b)
    stats['engagement'] = stat(measure_engagement, \
            lambda a, b: (a[0] + b[0], a[1] + b[1]))
    return stats

# literally the same thing, but for diff key names:
def get_outcome_stats_old():
    stats = {}
    for n in ('correct_lick', 'bad_lick', 'bad_ignore', \
            'correct_ignore'):
        stats[n] = stat(functools.partial(get_outcomes, name=n))
    for n in ('targetprob', 'curr_contrast', 'phases_completed', \
            'targetprob_lower'):
        stats[n] = stat(functools.partial(get_last, name=n), \
                lambda a, b: b)
    stats['engagement'] = stat(measure_engagement, \
            lambda a, b: (a[0] + b[0], a[1] + b[1]))
    return stats


# def get_outcomes(df, name=''):
#   evs = df.get_events('#stimDisplayUpdate')
#   outcomes = d.get_events(n)
#   ts = pymworks.events.display.to_trials(evs, outcomes)
#   gts = datautils.grouping.groupn(ts, ('name', 'outcome'))
#   nOutcomes = datautils.grouping.ops.stat(gts, len)

#   respkeys = [k for k in nOutcomes.keys()]
#   corrlicks = []
#   badignores = []
#   corrignores = []
#   badlicks = []
#   for k in respkeys:
#       resptypes = nOutcomes[k].keys()
#       for t in resptypes:
#           if 'correct_lick' in t:
#               corrlicks.append(nOutcomes[k][t])
#           if 'bad_ignore' in t:
#               badignores.append(nOutcomes[k][t])
#           if 'correct_ignore' in t:
#               corrignores.append(nOutcomes[k][t])
#           if 'bad_lick' in t:
#               badlicks.append(nOutcomes[k][t])
#   correct_licks = sum(corrlicks)
#   bad_ignores = sum(badignores)
#   correct_ignores = sum(corrignores)
#   bad_licks = sum(badlicks)

#   gevs = [correct_licks, bad_ignores, correct_ignores, bad_licks]
#   return gevs



def process_datafiles(dfns, stats):
    data = {}
    for dfn in dfns:
        logging.debug("Datafile: %s" % dfn)
        df = None
        df = pymworks.open(dfn)
        r = {}
        for s in stats:
            try:
                r[s] = stats[s][0](df)
                logging.debug("\t%s: %s" % (s, r[s]))
            except TypeError:
                print "stat: ", s, "file: ", dfn
        df.close()
        data[dfn] = r

    return data


def process_outcomes(dfns, stats):
    data = {}
    for dfn in dfns:
        df = None
        df = pymworks.open(fn)

        r = {}
        evs = df.get_events('#stimDisplayUpdate')
        outcomes = df.get_events(('session_correct_lick', 'session_correct_ignore',\
                'session_bad_lick', 'session_bad_ignore'))
        ts = pymworks.events.display.to_trials(evs, outcomes)
        gts = datautils.grouping.groupn(ts, ('name', 'outcome'))
        nOutcomes = datautils.grouping.ops.stat(gts, len)

        respkeys = [k for k in nOutcomes.keys()]
        corrlicks = []
        badignores = []
        corrignores = []
        badlicks = []
        for k in respkeys:
            resptypes = nOutcomes[k].keys()
            for t in resptypes:
                if 'correct_lick' in t:
                    corrlicks.append(nOutcomes[k][t])
                if 'bad_ignore' in t:
                    badignores.append(nOutcomes[k][t])
                if 'correct_ignore' in t:
                    corrignores.append(nOutcomes[k][t])
                if 'bad_lick' in t:
                    badlicks.append(nOutcomes[k][t])
        correct_licks = sum(corrlicks)
        bad_ignores = sum(badignores)
        correct_ignores = sum(corrignores)
        bad_licks = sum(badlicks)



        df.close()

        data[dfn] = r

    return data




def datafiles_to_sessions(data, stats):
    sdata = {}
    for k in sorted(data.keys()):
        fn = os.path.splitext(os.path.basename(k))[0]
        animal, session, epoch = parse_filename(fn)
        if animal not in sdata:
            sdata[animal] = {}
        if session not in sdata[animal]:
            sdata[animal][session] = data[k]
        else:  # this is a partial session
            sdata[animal][session] = merge_epochs(sdata[animal][session], \
                    data[k], stats)
    return sdata



# PLOT FUNCS:
def apply_standard_plotstyle(ax, nonpercents=0, default=0):
    if default == 0:
        standard_axes = [1, nSessions, 0, 1.0]
        ax.set_xlabel('sessions')
        
        plt.rc('legend', fontsize=8, loc='upper right', fancybox=True)
        plt.rc('axes', titlesize='medium', labelsize='small')
        plt.rc('xtick', labelsize='small')
        plt.rc('ytick', labelsize='small')
        plt.rc('lines', markersize=3, markeredgewidth=0.3)
        plt.axis([1, nSessions, 0, 1.0])
        if nonpercents==1:
            plt.axis([1, nSessions, 0, maxy])
    elif default == 1:
        plt.xlabel('')
        plt.rcdefaults()

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10)) # outward by 10 pts
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there's no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])



if __name__ == '__main__':

    # parent_input_path = 'input'
    parent_input_path = sys.argv[1]
    input_list = listdir_nohidden(parent_input_path)

    input_paths = []
    for name in input_list:
        input_paths.append(parent_input_path + '/' + name)


    dfns = []
    sdata = {}
    print "Beginning analysis for: %s" % input_paths
    for path in input_paths:
        print "CURR PATH: ", path
        dfns = get_datafiles(path)

        if path[len(path)-2] == 'U':
            stats = get_outcome_stats_old()
        else:
            stats = get_outcome_stats()
        data = process_datafiles(dfns, stats)
        # super-dict, contains dicts with structure:
        # {'input/W1/W1_130x0x': {'stat1': val, 'stat2': val, ..., 'statn': val}}, \
        # {'input/W2/W1_130y0y': {'stat1': val, 'stat2': val, ..., 'statn': val}}...

        file_dir = [] # make file directory for this animal
        for fidx, fn in enumerate(dfns):
            # print "Checking file: %s" % fn
            animal, date, epoch = parse_filename(fn)
            file_dir.append([animal, date])


        curr_sdata = datafiles_to_sessions(data, stats)
        sdata.update(curr_sdata)

    print sdata
    nSubjects = len(sdata)
    for n in range(nSubjects):
        animal = input_list[n]
        sessions = sorted(sdata[animal].keys())

        # GET STATS TO PLOT, BY SESSION:
        performance_vect = []
        adj_performance_vect = []
        target_probs = []
        distractor_contrasts = []
        time_engaged = []
        trial_counts = []
        phases = []
        targprob_lowerbound = []
        try:
            for s in sessions:
                outcomes = {}
                adj_outcomes = []

                resptypes = [r for r in sorted(stats.keys()) if 'session_' in r]
                varnames = [r[8:len(r)] for r in resptypes]

                for var in (varnames):
                    if not sdata[animal][s]['session_'+var]:
                        outcomes[var] = 0.
                    else:
                        outcomes[var] = sdata[animal][s]['session_'+var][0]

                ntargs = outcomes['correct_lick'] + outcomes['bad_ignore']
                ndists = outcomes['correct_ignore'] + outcomes['bad_lick']
                nTrials = ntargs + ndists

                outcomes_num = [outcomes[key] for key in varnames]
                outcomes_percent = map(lambda x: x/float(nTrials), outcomes_num)

                if ntargs == 0:
                    adj_corrlicks = 0.
                    adj_badignores = 0.
                else:
                    adj_corrlicks = float(outcomes['correct_lick'])/ntargs
                    adj_badignores = float(outcomes['bad_ignore'])/ntargs

                if ndists == 0:
                    adj_corrignores = 0.
                    adj_badlicks = 0.
                else:
                    adj_corrignores = float(outcomes['correct_ignore'])/ndists
                    adj_badlicks = float(outcomes['bad_lick'])/ndists                    

                adj_outcomes = [adj_badignores, adj_badlicks, adj_corrignores, \
                        adj_corrlicks]


                # for k, v in sorted(outcomes.iteritems()):
                #     if v==0:
                #         adj_outcomes.append(0.0)
                #     else:
                #         if k=='bad_ignore' or k=='correct_lick':
                #             adj_outcomes.append(float(v)/ntargs)
                #         else:
                #             adj_outcomes.append(float(v)/ndists)


                # if ntargs == 0:
                #     correct_licks = 0.
                #     bad_ignores = 0.

                #     adj_correct_licks = 0.
                #     adj_bad_ignores = 0.
                # else:
                #     correct_licks = correct_lick/nTrials
                #     bad_ignores = bad_ignore/nTrials

                #     adj_correct_licks = correct_lick/ntargs
                #     adj_bad_ignores = bad_ignore/ntargs

                # if ndists == 0:
                #     correct_ignores = 0.
                #     bad_licks = 0.

                #     adj_correct_ignores = 0.
                #     adj_bad_licks = 0.
                # else:
                #     correct_ignores = correct_ignore/nTrials
                #     bad_licks = bad_lick/nTrials

                #     adj_correct_ignores = correct_ignore/ndists
                #     adj_bad_licks = bad_lick/ndists

                # outcomes_num = [correct_lick, correct_ignore, bad_lick, bad_ignore]
                # outcomes_percent = [correct_licks, correct_ignores, bad_licks, bad_ignores]
                # adj_outcomes = [adj_correct_licks, adj_correct_ignores, \
                #                     adj_bad_licks, adj_bad_ignores]
                

                targetprob = sdata[animal][s]['targetprob']
                curr_contrast = sdata[animal][s]['curr_contrast']
                time_headout = (sdata[animal][s]['engagement'][0])/1E6
                time_total = (sdata[animal][s]['engagement'][1])/1E6
                if time_total == 0:
                    print "ZERO DIV ERROR: ", s
                else:
                    percent_engaged = time_headout/time_total
                phases_completed = sdata[animal][s]['phases_completed']
                targetprob_lower = sdata[animal][s]['targetprob_lower']

                # update plot vects:
                performance_vect.append(outcomes_percent)
                adj_performance_vect.append(adj_outcomes)
                target_probs.append(targetprob)
                distractor_contrasts.append(curr_contrast)
                time_engaged.append(percent_engaged)
                trial_counts.append(nTrials)
                phases.append(phases_completed)
                targprob_lowerbound.append(targetprob_lower)

                # p_sdata = sdata[animal]
                # filename = animal+'_stats'
                # # # outpath = 'output/'
                # ofile = open(output_paths[n]+'/'+filename, 'wb')
                # pickle.dump(p_sdata, ofile)
                # ofile.close()

        except IndexError:
            print animal, s


        #------------------------------------------------
        # PLOTTING (by animal):
        #------------------------------------------------
        nSessions = len(sessions)
        nOutcomes = len(outcomes_percent)

        # general specs for main FIG:
        # fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
        plt.close('all')

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
        fig.suptitle(animal, fontsize=12, fontweight='bold')
        fig.facecolor='0.75'


        apply_standard_plotstyle(ax1)
        # ax1:  plot "outcomes"
        xvals = range(1, nSessions + 1)
        yvals = zip(*performance_vect)        
        colors = ['g', 'b', 'y', 'r']
        labels = varnames

        ax1.axis([1, nSessions, 0., 1.])
        ax1.set_title('performance by outcome type')
        ax1.set_ylabel('proportion of trials')

        for n in range(0, nOutcomes, 1):
            ax1.plot(xvals, yvals[n], 'o-', color=colors[n], label=labels[n])
            apply_standard_plotstyle(ax1)
        lgd = ax1.legend()
        lgd.get_frame().set_alpha(0.5)

        ax1.fill_between(xvals, targprob_lowerbound, 1.0, facecolor='gray', \
                edgecolor='None', alpha=0.50, interpolate=True)


        # ax2:  plot adjusted outcome performance:
        xvals = range(1, nSessions + 1)
        yvals = zip(*adj_performance_vect)        
        colors = ['g', 'b', 'y', 'r']
        # labels = ['corrLicks', 'corrIgnores','badLicks', 'badIgnores']

        ax2.axis([1, nSessions, 0., 1.])
        ax2.set_title('adjusted performance')
        ax2.set_ylabel('proportion of trials, by trial type')

        for n in range(0, nOutcomes, 1):
            ax2.plot(xvals, yvals[n], 'o-', color=colors[n], label=labels[n])
            apply_standard_plotstyle(ax2)
        # lgd = ax2.legend()
        # lgd.get_frame().set_alpha(0.5)


        # ax3:  plot staircase performance
        staircases = [target_probs, distractor_contrasts]
        xvals = range(1, nSessions + 1)
        yvals = staircases
        styles = ['o-', 'o--']
        markercolors = ['k','w']
        labels = ['target prob', 'distractor contrast']
        maxy=1.0

        ax3.axis([1, nSessions, 0., 1.])
        ax3.set_title('staircase training performance')
        ax3.set_ylabel('value')

        for n in range(0, len(staircases), 1):
            ax3.plot(xvals, yvals[n], styles[n], color='k', label=labels[n], \
                    markerfacecolor=markercolors[n])
            apply_standard_plotstyle(ax3) 
        lgd = ax3.legend()
        lgd.get_frame().set_alpha(0.5)

        phase_idx = [1, 2, 3, 4, 5]
        phase_colors = ['r','DarkOrange','y', 'g','b']
        for i, p in enumerate(phases):
            ax3.axvline(x=xvals[i], color=phase_colors[int(p)], linewidth=4, \
                alpha=0.4)



        # ax4:  plot engagement vals
        xvals = range(1, nSessions + 1)
        yvals = time_engaged
        
        ax4.axis([1, nSessions, 0., 1.])
        ax4.set_title('engagement')
        ax4.set_ylabel('proportion of time head out')
        
        ax4.plot(xvals, yvals, 'o-')
        apply_standard_plotstyle(ax4)



        # ax5:  plot trials counts:
        xvals = range(1, nSessions + 1)
        yvals = trial_counts
        maxy=max(yvals)
        
        ax5.axis([1, nSessions, 0., maxy])
        ax5.set_title('trials per session')
        ax5.set_ylabel('num trials')

        ax5.plot(xvals, yvals, 'o-')
        apply_standard_plotstyle(ax5, nonpercents=1)



        # ax6:  phase legend
        phase_colors = ['r','DarkOrange','y', 'g','b']
        nPhases = len(phase_colors)
        xvals = [1,2,3,4,5]
        yvals = [(1,1,1,1,1), (1,1,1,1,1), (1,1,1,1,1), \
                    (1,1,1,1,1), (1,1,1,1,1)]
        labels = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5']
        
        ax6.axis([1, 5, 1, 5])
        # ax6.set_title('phase key')
        
        apply_standard_plotstyle(ax6, default=1)
        for n in range(0, nPhases):
            ax6.plot(xvals, yvals[n], '-', lw=4, color=phase_colors[n], \
                            alpha=0.4, label=labels[n])
        adjust_spines(ax6, [])
        # plt.rc('legend', fontsize=12, loc='center', fancybox=True)
        lgd = ax6.legend(loc='center', prop={'size':12}, title='phase key', \
                            frameon=False)


        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

        # plt.savefig('summary.png')
        plt.close('all')


## THIS ALONE WORKS:

# input_path = 'input/W5' # same path as in ab_incrementing.
# fn = input_path+'W5_130726.mwk'
# df = pymworks.open(fn)
# evs = df.get_events('#stimDisplayUpdate')

# outcomes = d.get_events(('session_correct_lick', 'session_correct_ignore',\
#       'session_bad_lick', 'session_bad_ignore', 'aborted_counter'))

# ts = pymworks.events.display.to_trials(evs, outcomes)

# gts = datautils.grouping.groupn(ts, ('name', 'outcome'))

# # gts keys = 'target' 'distractor'
# # gts['target'] = all stim events, can be corrlicks, badignores, aborts
# # gts['distractor'] = all stim events, can be corrignores, badlicks

# nOutcomes = datautils.grouping.ops.stat(gts, len)

# corrlicks = nOutcomes['target']['session_correct_lick']
# badignores = nOutcomes['target']['session_bad_ignore']
# corrignores = nOutcomes['distractor']['session_correct_ignore']
# badlicks = nOutcomes['distractor']['session_bad_lick']


## BLUE SQUARE CRAP:
# corrlicks = nOutcomes['BlueSquare']['session_correct_lick']
# badignores = nOutcomes['BlueSquare']['session_bad_ignore']

# respkeys = [k for k in nOutcomes.keys()]

# corrlicks = []
# badignores = []
# corrignores = []
# badlicks = []
# for k in respkeys:
#   resptypes = nOutcomes[k].keys()
#   for r in resptypes:
#       print k, r
#       if 'correct_lick' in r:
#           corrlicks.append(nOutcomes[k][r])
#       if 'bad_ignore' in r:
#           badignores.append(nOutcomes[k][r])
#       if 'correct_ignore' in r:
#           corrignores.append(nOutcomes[k][r])
#       if 'bad_lick' in r:
#           badlicks.append(nOutcomes[k][r])

# correct_licks = sum(corrlicks)
# bad_ignores = sum(badignores)
# correct_ignores = sum(corrignores)
# bad_licks = sum(badlicks)




# FYI:  for bluesquare, nDistractors = 12, so:
# nOutcomes = 
# {'0': {'session_correct_ignore': 3},
#  '1': {'session_correct_ignore': 2},
#  '2': {'session_correct_ignore': 1},
#  '3': {'session_correct_ignore': 1},
#  '6': {'session_bad_lick': 1, 'session_correct_ignore': 1},
#  '7': {'session_correct_ignore': 1},
#  '8': {'session_bad_lick': 1, 'session_correct_ignore': 1},
#  'BlueSquare': {'session_bad_ignore': 192, 'session_correct_lick': 244}}