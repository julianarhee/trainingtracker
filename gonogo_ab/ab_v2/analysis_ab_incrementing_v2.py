#!/usr/bin/env python
# simple data analysis attempt.
# FIXED:  18 June 2013 (jyr)

# Protocol:  gonogo_ab.xml 
# Animals:  U (2x), V(8x)
# Problem with previous scripts for incremented events, fixed.

import fnmatch
import functools
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pymworks

import cPickle as pickle
import sys


# get_incremented_range -- not okay. removal process chunking wrong.
# get_last -- always grabs value of last event, sorted by time.  OK.
# measure_engagement -- timing order, OK; but, grabbing too much... chunking problem, again.

## NEED FUNCTION THAT COMBS THROUGH ALL EVENTS, CUT OUT ONLY THE ACTUAL SESSION-TRIALS, \
## BASED ON *TIME* AND INCREMENTED VALUES, THEN RETURN THAT CHUNK AS "df"
## Variant of "process_data_files(dfns, stats) -- outputs data" ??


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
    evs = df.get_events(name)
    for ev in evs:
        if type(ev.value) != float:
            ev.value = float(ev.value)
    gevs = session_incremented(evs)
    return pymworks.stats.events.valuerange(gevs)


def measure_engagement(df, name='head_input', threshold=500):
    return pymworks.stats.events.time_in_state(df.get_events(name), \
            test=lambda e: e.value < threshold)


def get_incremented_range(df, name=''):
    evs = df.get_events(name)
    # gevs = get_incrementing(evs)
    gevs = pymworks.stats.events.remove_non_incrementing(evs)
    return pymworks.stats.events.valuerange(gevs)

    # pymworks.stats.events.remove_non_incrementing(evs): Forward removal, then backward removal \
    # (1) FW:  a = evs[i],  b = evs[i+1] \
    # For i=[0, 1, ..., len(evs)-1],  compares evs[i] to evs[i+1]. \
    # Removes evs[i] if evs[i+1].value is GT than evs[i].value+1, i.e., \
    # delete a, if b.value > (a.value + 1), where b is the NEXT event.

    # (2) BW:  a = evs[i-1], b = evs[i] \
    # For i = [1, 2, ..., len(evs)],  compares evs[i] to evs[i-1]. \
    # Removes evs[i] if evs[i].value is LT than evs[i-1].value+1, i.e., \
    # delete b, if b.value < (a.value + 1), where a is the PREV event.

def get_max(df, name=''):
    return pymworks.stats.events.valuemax(df.get_events(name)).value

def get_last(df, name=''):
    return sorted(df.get_events(name), key=lambda e: e.time)[-1].value

def stat(extract_function=get_incremented_range, \
        combine_function=lambda a, b: a + b):
    return (extract_function, combine_function)


def get_session_stats():
    stats = {}
    for n in ('session_correct_lick', 'session_bad_lick', 'session_bad_ignore', \
            'session_correct_ignore', 'session_target_trialcounter', \
            'session_distractor_trialcounter', \
            'trials_in_session', 'aborted_counter'):
        stats[n] = stat(functools.partial(get_session_range, name=n))
    for n in ('targetprob', 'curr_contrast', 'phases_completed', \
            'targetprob_lower'):
        stats[n] = stat(functools.partial(get_last, name=n), \
                lambda a, b: b)
    stats['engagement'] = stat(measure_engagement, \
            lambda a, b: (a[0] + b[0], a[1] + b[1]))
    return stats


def get_stats():
    stats = {}
    for n in ('correct_lick', 'bad_lick', 'bad_ignore', 'correct_ignore',
            'target_trialcounter', 'distractor_trialcounter'):
        stats[n] = stat(functools.partial(get_incremented_range, name=n))
    for n in ('targetprob', 'curr_contrast', 'phases_completed', 'targetprob_lower'):
        stats[n] = stat(functools.partial(get_last, name=n), \
                lambda a, b: b)
    stats['engagement'] = stat(measure_engagement, \
            lambda a, b: (a[0] + b[0], a[1] + b[1]))
    return stats

def get_datafiles(indir, match='*.mwk'):
    dfns = []
    for d, sds, fns in os.walk(indir):
        mfns = fnmatch.filter(fns, match)
        dfns += [os.path.join(d, fn) for fn in mfns]
    return dfns

def process_datafiles(dfns, stats):
    data = {}
    for dfn in dfns:
        logging.debug("Datafile: %s" % dfn)
        df = None
        df = pymworks.open(dfn)
        r = {}
        for s in stats:
            r[s] = stats[s][0](df)
            logging.debug("\t%s: %s" % (s, r[s]))
        df.close()
        data[dfn] = r
    return data

def merge_epochs(epoch0, epoch1, stats):
    # TODO: make sure that epoch 1 is the later one
    r = {}
    for s in stats:
        r[s] = stats[s][1](epoch0[s], epoch1[s])
    return r

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

def listdir_nohidden(path):
    dir_list = []
    temp_list = os.listdir(path)
    for f in temp_list:
        if not f.startswith('.'):
            dir_list.append(f)
    return dir_list

def grab_datafiles_for_animal(animal, input_directory):
    dfs = []
    for f in os.listdir(input_directory):
        try:
            if parse_filename(f)[0].upper() == animal.upper():
                dfs.append('/'.join((input_directory, f)))
        except Exception:
            pass
    return dfs

def create_outfiles_for_animal(animal, output_directory):
    dfs = []
    for f in os.listdir(output_directory):
        try:
            if parse_filename(f)[0].upper() == animal.upper():
                dfs.append('/'.join((output_directory, f)))
        except Exception:
            pass
    return dfs

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
    # setup correct input/output directories
    input_path_name = sys.argv[1]
    output_path_name = os.path.split(input_path_name)[0] + '/output'

    input_name_list = listdir_nohidden(input_path_name)

    input_paths = []
    output_paths = []
    for name in input_name_list:
        input_paths.append(input_path_name + '/' + name)
        output_paths.append(output_path_name + '/' + name)

    for output_path in output_paths:
        try:
            os.makedirs(output_path)
        except OSError:
            pass

    # get data files and session performance summaries for each animal
    dfns = []
    sdata = {}
    print "Running input_paths = %s" % input_paths
    for input_path in input_paths:
        print "CURR PATH: %s" % input_path 
        dfns = get_datafiles(input_path)

        stats = get_session_stats()
        data = process_datafiles(dfns, stats) # super-dict of all animals (input-paths)
        # contains sub-dicts, key='input/animal/animal_date', value=dict of stats
        # data = get_session_trials(dfns, stats)

        countkeys = [s for s in stats.keys() if 'bad' in s or 'correct' in s or 'count' in s]
        # print countkeys
        for k in countkeys:
            if 'bad_lick' in k:
                key_badlick = k
            elif 'bad_ignore' in k:
                key_badign = k
            elif 'correct_lick' in k:
                key_corrlick = k
            elif 'correct_ignore' in k:
                key_corrign = k
            elif 'target' in k and 'count' in k:
                key_targs = k
            elif 'distractor' in k and 'count' in k:
                key_dists = k


        file_dir = [] # make file director for this animal
        for fidx, fn in enumerate(dfns):
            animal, date, epoch = parse_filename(fn)
            file_dir.append([animal, date])

        # stats = get_stats()
        # data = process_datafiles(dfns, stats) # super-dict of all animals (input-paths)
        # contains sub-dicts, key='input/animal/animal_date', value=dict of stats

            tmp_correct_licks = data[fn][key_corrlick]
            tmp_correct_ignores = data[fn][key_corrign]
            tmp_bad_licks = data[fn][key_badlick]
            tmp_bad_ignores = data[fn][key_badign]
            aborted = data[fn]['aborted_counter']

            # sanity checks:
            tmp_nTargets = data[fn][key_targs]
            tmp_nDistractors = data[fn][key_dists]
            targ_outcomes = tmp_correct_licks + tmp_bad_ignores + aborted
            dist_outcomes = tmp_correct_ignores + tmp_bad_licks

            tmp_nTrials = data[fn]['trials_in_session']
            
            if not ((targ_outcomes == tmp_nTargets) and (dist_outcomes ==\
                    tmp_nDistractors)):
                if targ_outcomes != tmp_nTargets:
                    print 'TARGS: off by %d, in %s' % (tmp_nTargets\
                            -targ_outcomes,fn)
                    print 'correct licks %d, bad ignores %d, aborted %d.'\
                            % (tmp_correct_licks, tmp_bad_ignores, aborted)

                if dist_outcomes != tmp_nDistractors:
                    print 'DISTS: off by %d, in %s' % (tmp_nDistractors\
                            -dist_outcomes,fn)
                    print 'correct ignores %d, bad licks %d.' % \
                            (tmp_correct_ignores, tmp_bad_licks)

        print "Completed file checks."

        # merge epochs if nec:
        curr_sdata = datafiles_to_sessions(data, stats)
        sdata.update(curr_sdata) # full-session data for each animal -- num animals should equal len(sdata)

    nSubjects = len(sdata)
    for n in range(nSubjects):
        animal = input_name_list[n]
        sessions = sorted(sdata[animal].keys())

        # get stats to plot, by session:
        performance_vect = []
        adj_performance_vect = []
        target_probs = []
        distractor_contrasts = []
        time_engaged = []
        trial_counts = []
        phases = []
        targprob_lowerbound = []
        for s in sessions:
            correct_lick = float(sdata[animal][s][key_corrlick])
            correct_ignore = float(sdata[animal][s][key_corrign])
            bad_lick = float(sdata[animal][s][key_badlick])
            bad_ignore = float(sdata[animal][s][key_badign])
            
            nTargets = correct_lick+bad_ignore
            nDistractors = correct_ignore+bad_lick
            nTrials = nTargets+nDistractors

            if nTargets == 0:
                correct_licks = 0.
                bad_ignores = 0.

                adj_correct_licks = 0.
                adj_bad_ignores = 0.
            else:
                correct_licks = correct_lick/nTrials
                bad_ignores = bad_ignore/nTrials

                adj_correct_licks = correct_lick/nTargets
                adj_bad_ignores = bad_ignore/nTargets

            if nDistractors == 0:
                correct_ignores = 0.
                bad_licks = 0.

                adj_correct_ignores = 0.
                adj_bad_licks = 0.
            else:
                correct_ignores = correct_ignore/nTrials
                bad_licks = bad_lick/nTrials

                adj_correct_ignores = correct_ignore/nDistractors
                adj_bad_licks = bad_lick/nDistractors

            outcome_trials = [correct_lick, correct_ignore, bad_lick, bad_ignore]
            outcomes = [correct_licks, correct_ignores, bad_licks, bad_ignores]
            adj_outcomes = [adj_correct_licks, adj_correct_ignores, \
                                adj_bad_licks, adj_bad_ignores]
            

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
            performance_vect.append(outcomes)
            adj_performance_vect.append(adj_outcomes)
            target_probs.append(targetprob)
            distractor_contrasts.append(curr_contrast)
            time_engaged.append(percent_engaged)
            trial_counts.append(nTrials)
            phases.append(phases_completed)
            targprob_lowerbound.append(targetprob_lower)

            p_sdata = sdata[animal]
            filename = animal+'_stats'
            # # outpath = 'output/'
            ofile = open(output_paths[n]+'/'+filename, 'wb')
            pickle.dump(p_sdata, ofile)
            ofile.close()

            # p_file = open(ofile, 'rb')
            # f_sdata = pickle.load(p_file)
            # p_file.close()
            # print f_sdata





        #------------------------------------------------
        # PLOTTING (by animal):
        #------------------------------------------------
        nSessions = len(sessions)
        nOutcomes = len(outcomes)

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
        colors = ['r', 'y', 'b', 'g']
        labels = ['corrLicks', 'corrIgnores','badLicks', 'badIgnores']

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
        colors = ['r', 'y', 'b', 'g']
        labels = ['corrLicks', 'corrIgnores','badLicks', 'badIgnores']

        ax2.axis([1, nSessions, 0., 1.])
        ax2.set_title('adjusted performance')
        ax2.set_ylabel('proportion of trials')

        for n in range(0, nOutcomes, 1):
            ax2.plot(xvals, yvals[n], 'o-', color=colors[n], label=labels[n])
            apply_standard_plotstyle(ax2)
        lgd = ax2.legend()
        lgd.get_frame().set_alpha(0.5)


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
