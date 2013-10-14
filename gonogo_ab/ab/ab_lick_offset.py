#!/usr/bin/env python
# simple data analysis attempt for within-session
# April 22, 2013.
# Go/NoGo task, new protocol with optimizers.

# For "logging" mode during debugging/testing new code:
# (These need to be done BEFORE importing pymworks, etc.)

# In [1]: import logging

# In [2]: logging.basicConfig(level=logging.DEBUG)

# In [3]: logging.debug('foo')
# DEBUG:root:foo


import fnmatch
import functools
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sci
import math

import pymworks

def stdev(values, mean):
    size = len(values)
    sum = 0.0
    for n in range(0, size):
        sum += ((values[n] - mean)**2)
    return math.sqrt((1.0/(size-1))*(sum))

def measure_engagement(df, name='head_input', threshold=500):
    return pymworks.stats.events.time_in_state(df.get_events(name), \
            test=lambda e: e.value < threshold)


def get_incremented_range(df, name=''):
    evs = df.get_events(name)
    gevs = pymworks.stats.events.remove_non_incrementing(evs)
    return pymworks.stats.events.valuerange(gevs)


def get_max(df, name=''):
    return pymworks.stats.events.valuemax(df.get_events(name)).value


def get_last(df, name=''):
    return sorted(df.get_events(name), key=lambda e: e.time)[-1].value


def stat(extract_function=get_incremented_range, \
        combine_function=lambda a, b: a + b):
    return (extract_function, combine_function)


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

# JUST COMBINE FILES IF NEC:
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
    # input_path_name = '/Users/Juliana/Documents/lab/coxlab/experiments/GoNogo_BlueSquare/testcode/input'
    # output_path_name = '/Users/Juliana/Documents/lab/coxlab/experiments/GoNogo_BlueSquare/testcode/output'
    input_path_name = 'sessions'
    # output_path_name = 'output'

    input_name_list = listdir_nohidden(input_path_name)

    input_paths = []
    # output_paths = []
    for name in input_name_list:
        input_paths.append(input_path_name + '/' + name)
        # output_paths.append(output_path_name + '/' + name)

    # for output_path in output_paths:
    #     try:
    #         os.makedirs(output_path)
    #     except OSError:
    #         pass

    # get data files and session performance summaries for each animal
    dfns = []
    sdata = {}

    avg_latencies_animal = []
    avg_by_session = []
    stderr_by_session = []

    avg_by_session_durs = []
    stderr_by_session_durs = []

    nsessions = []
    animal_list = []
    # percent_long = []
    # percent_short = []

    print "Running input_paths = %s" % input_paths
    for input_path in input_paths:
        print "Current input_path = %s" % input_path
        dfns = get_datafiles(input_path)

        stats = get_stats()
        data = process_datafiles(dfns, stats) # super-dict{ \
                # {'input_path_name/animal/animal_date1': {'stat1': value, 'stat2': value, etc.}}, \
                # {'input_path_name/animal/animal_date2': {'stat1': value, 'stat2': value, etc.}}, ...}
    
        # merge epochs if nec:
        curr_sdata = datafiles_to_sessions(data, stats)
        sdata.update(curr_sdata)

        file_dir = [] # make file directory for this animal
        tp_sessions = ['' for fn in dfns]
        ntrials_sessions = ['' for fn in dfns]
        fn_sessions = ['' for fn in dfns]

        avg_latencies_session = []
        stderr_session = []
        avg_durs_session = []
        stderr_dur_session = []
        # percent_long = []
        # percent_short = []
        for fidx, fn in enumerate(dfns):
            ff = os.path.splitext(os.path.basename(fn))[0]     
            animal, session, epoch = parse_filename(ff)
            file_dir.append([animal, session, epoch])

            # print fn
            tmp_correct_licks = data[fn]['correct_lick']
            tmp_correct_ignores = data[fn]['correct_ignore']
            tmp_bad_licks = data[fn]['bad_lick']
            tmp_bad_ignores = data[fn]['bad_ignore']

            # sanity checks:
            target_count_check = []
            distractor_count_check = []
            good_counts = []
            tmp_nTargets = data[fn]['target_trialcounter']
            tmp_nDistractors = data[fn]['distractor_trialcounter']
            if (tmp_correct_licks+tmp_bad_ignores) != tmp_nTargets:
                # print 'Target count off by %d, in %s' % (tmp_nTargets-(tmp_correct_licks+tmp_bad_ignores),fn)
                difference = tmp_nTargets-(tmp_correct_licks+tmp_bad_ignores)
                target_count_check.append([fn, difference])
            if (tmp_correct_ignores+tmp_bad_licks) != tmp_nDistractors:
                # print 'Distractor count off by %d, in %s' % (tmp_nDistractors-(tmp_correct_ignores+tmp_bad_licks), fn)
                difference = tmp_nDistractors-(tmp_correct_ignores+tmp_bad_licks)
                distractor_count_check.append([fn, difference])
            if ((tmp_correct_licks+tmp_bad_ignores) == tmp_nTargets) \
                    and ((tmp_correct_ignores+tmp_bad_licks) == tmp_nDistractors):
                good_counts.append([fn])

            if target_count_check != []:
                print "Targets: ", target_count_check
            elif distractor_count_check != []:
                print "Distractors: ", distractor_count_check
            else: 
                print "Session trial counts good."


            # get stimulus display update events, and parse them into trials:
            df = pymworks.open(fn)
            evs = df.get_events('#stimDisplayUpdate')
            # licks = sorted(df.get_events('correct_lick'), key=lambda e: e.time) # just grab those trials where animal correctly licked
            outcome_events = df.get_events(('correct_lick', 'bad_lick', 'correct_ignore', 'bad_ignore'))

            stims = pymworks.events.display.to_stims(evs)
            trials = pymworks.events.display.to_trials(evs, outcome_events) # list of dicts, each dict contains trial info

            if len(stims) != len(trials):
                print "Stims != Trials: %s, %d" % (fn, len(stims)-len(trials))
            else:
                print "Events by stim and trial are matched."

            # get only the correct-lick trials, and make a new list of trial dicts:
            corr_lick_trials = []
            corr_lick_idx = []
            for tidx, trial in enumerate(trials):
                if trial['outcome'] == 'correct_lick':
                    corr_lick_trials.append(trial)
                    corr_lick_idx.append(tidx)
            # print corr_lick_trials

            windows = []
            # last_trial = trials[corr_lick_idx[-1]]['time'] # time stamp of last trial stim onset
            # tstart = trials[corr_lick_idx[0]]['time'] # first trial, first time stamp
            for t in corr_lick_idx:
                tstart = trials[t]['time']
                tend = tstart + trials[t]['duration'] # 1s block
                windows.append([tstart, tend])

            lick_input_evs = df.get_events('lick_input')
            lick_input_evs = sorted(df.get_events('lick_input'), key=lambda e: e.time)
            lick_max = df.get_events('licksensor_max')[1].value
            lick_min = df.get_events('licksensor_min')[1].value
            # print "lick_max: ", lick_max
            # print "lick_min: ", lick_min

            lick_latency_all = []
            lick_dur_all = []
            by_trial = []
            check_late_licks = []
            for widx, w in enumerate(windows):
                onset = w[0]
                time_lick = []

                for lick in lick_input_evs:
                    if (w[0] < lick.time < w[1]) and (lick_min <= lick.value <= lick_max):
                        time_lick.append([lick.time, lick.value])
                    
                by_trial.append(time_lick)

                for lidx, lick_info in enumerate(by_trial):
                    # print lidx, "licks this trial: ", len(lick_info)
                    if len(lick_info) > 1:
                        first_lick = lick_info[0][0]
                        last_lick = lick_info[-1][0]
                        continue
                    elif len(lick_info) == 1:
                        first_lick = lick_info[0][0]
                        last_lick = lick_info[-1][0]
                        continue
                    else:
                        print "no licks found"
                        print "Trying again..."

                time_to_lick = (first_lick - onset) / 1E6

                if (time_to_lick < 0.0):
                    print widx, w, time_to_lick

                time_to_stop = (last_lick - onset) / 1E6
                lick_dur = time_to_stop - time_to_lick

                lick_latency_all.append(time_to_lick)
                lick_dur_all.append(lick_dur)

            lick_latency = [t for t in lick_latency_all if t > 0] # get rid of weird timing trials, no lick detected
            lick_dur = [d for d in lick_dur_all if d > 0]
            # print "n lick latencies, by trial: ", len(lick_latency)
            # print "n correct_lick trials, by trial: ", len(corr_lick_trials)
            # print "n all such licks by trial: ", len(by_trial)

            # print "N lick durations: ", len(lick_dur)

            # CALCULATE MEAN AND STDEV/STDERR FOR LATENCIES AND DURATIONS:
            avg_latency_np = np.mean(lick_latency)
            avg_latency = avg_latency_np.tolist()

            avg_dur_np = np.mean(lick_dur)
            avg_dur = avg_dur_np.tolist()

            stderr = sci.sem(lick_latency)
            stderr_dur = sci.sem(lick_dur)

            # Calculate standard error indendently:
            stdev = np.std(lick_latency, ddof=1)
            # stdev_calc = stdev(lick_latency, avg_latency) # defined func above, but type-error with float64
            # if stdev != stdev_calc:
            #     print "STDEV Numpy function and calculation off: ", stdev, stdev_calc

            stdev_dur = np.std(lick_dur, ddof=1)
            
            size = len(lick_latency)
            # stderr_calc = stdev_calc / math.sqrt(size)
            stderr_calc = stdev / math.sqrt(size)

            size_dur = len(lick_dur)
            stderr_calc_dur = stdev_dur / math.sqrt(size_dur)
            
            # Print out a message of "calculated" vs function for SEM off:
            if stderr != stderr_calc:
                print "STDERR Scipy vs calculation off: ", stderr, stderr_calc

            if stderr_dur != stderr_calc_dur:
                print "STDERR_DUR Scipy vs calculation off: ", stderr_dur, stderr_calc_dur

            # lick_latencies = np.array(lick_latency)
            # long_latency = sum(lick_latencies > 0.500)
            # short_latency = sum(lick_latencies < 0.300)
            # proportion_long = float(long_latency) / len(lick_latency)
            # proportion_short = float(short_latency) / len(lick_latency)
            # print "Proportion longer than 500ms: ", proportion_long


            avg_latencies_session.append(avg_latency) # list of average latency for each session (one animal)
            stderr_session.append(stderr) # standard error (using scipy.stats.sem) for lick-latencies, each session...
            # percent_long.append(proportion_long)
            # percent_short.append(proportion_short)

            avg_durs_session.append(avg_dur)
            stderr_dur_session.append(stderr_dur)

        # print "AVG Latency, by session: ", avg_latencies_session
        if len(avg_latencies_session) != len(stderr_session):
            print "num latencies and num stderr are off: ", len(avg_latencies_session), len(stderr_session)

        if len(avg_durs_session) != len(stderr_dur_session):
            print "num DURs and num STDERR_DURs are off: ", len(avg_durs_session), len(stderr_dur_session)
        
        nsessions.append(len(avg_latencies_session))
        
        avg_by_session.append(avg_latencies_session) #list containing n lists, where n corresponds to diff animal
        stderr_by_session.append(stderr_session)

        avg_by_session_durs.append(avg_durs_session)
        stderr_by_session_durs.append(stderr_dur_session)

        animal_list.append(animal)

        avg_across_sessions = np.mean(avg_latencies_session) # get mean lick-latency for that animal, across all sessions
        avg_latencies_animal.append(avg_across_sessions)

    # print "ANIMALS: ", animal_list
    # print "nsessions, all: ", nsessions
    # print "AVG, by session, all: ", avg_by_session
    # print "AVG all: ", avg_latencies_animal
    # print "AVG DUR, by session, all: ", avg_by_session_durs
    
    # print "Percent long, all: ", percent_long
    # print "Percent short, all: ", percent_short



    ##############################################
    # PLOT ANIMALS TOGETHER, one plot, by session:
    ##############################################

    # LICK DURATIONS ############################:

    plt.close('all')

    FIG = plt.figure(1)
    ax = FIG.add_subplot(111)
    xvals = ['' for n in nsessions]
    for nid, n in enumerate(nsessions):
        xvals[nid] = range(1, n+1)
    yvals = avg_by_session_durs
    stderrs = stderr_by_session_durs

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'gray']
    # colormap = plt.cm.gist_ncar
    # plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])

    for i in range(len(animal_list)):
        ax.axis([0, xvals[i][-1]+2, 1.0, 8.0])
        # plt.axis([1, nsessions[i], 0., 1.0])
        ax.set_ylabel('lick duration (sec)', fontsize=12)
        ax.set_xlabel('session', fontsize=12)
        ax.errorbar(xvals[i], yvals[i], stderrs[i], fmt='o')
        ax.plot(xvals[i], yvals[i], color=colors[i], linestyle='None', linewidth=0, marker='o', label=animal_list[i])
    # plt.axis([1, nsessions, 0., 1.0])
    # plt.xlabel('session')
    # plt.ylabel('latency from stim onset (sec)')
    plt.title('lick duration on correct-lick trials', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles, labels, loc='upper right')
    lgd = ax.legend()
    lgd.get_frame().set_alpha(0.5)
    plt.tight_layout()
    plt.show()

    # LICK LATENCIES ############################:
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    xvals = ['' for n in nsessions]
    for nid, n in enumerate(nsessions):
        xvals[nid] = range(1, n+1)
    yvals = avg_by_session
    stderrs = stderr_by_session

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'gray']

    for i in range(len(animal_list)):
        ax.axis([0, xvals[i][-1]+2, 0.0, 0.75]) # diff axis than lick-dur?
        # plt.axis([1, nsessions[i], 0., 1.0])
        ax.set_ylabel('time from stim onset (sec)', fontsize=12)
        ax.set_xlabel('session', fontsize=12)
        ax.errorbar(xvals[i], yvals[i], stderrs[i], fmt='o')
        ax.plot(xvals[i], yvals[i], color=colors[i], linestyle='None', \
                linewidth=0, marker='o', label=animal_list[i])
    # plt.axis([1, nsessions, 0., 1.0])
    # plt.xlabel('session')
    # plt.ylabel('latency from stim onset (sec)')
    plt.title('lick latency on correct-lick trials', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles, labels, loc='upper right')
    lgd = ax.legend()
    lgd.get_frame().set_alpha(0.5)
    plt.tight_layout()
    plt.show()

    
    # PLOT ANIMAL'S SESSION INFO, subplots:
    # plt.close('all')

    # fig = plt.figure(2, facecolor='None', edgecolor='None')
    # ax = fig.add_subplot(111)
    

    # xvals = ['' for n in nsessions]
    # for nid, n in enumerate(nsessions):
    #     xvals[nid] = range(1, n+1)
    # yvals = avg_by_session
    # stderrs = stderr_by_session

    # num_subplots = len(avg_latencies_animal)
    # for i in range(num_subplots):
    #     f = i+1
    #     ax1 = fig.add_subplot(num_subplots,1,f)
    #     print i, xvals[i][-1]
    #     ax1.axis([1, xvals[i][-1], 0.0, 1.0])
    #     ax1.errorbar(xvals[i], yvals[i], stderrs[i])
    #     ax1.set_ylabel("lick latency, sec", fontsize=8)
    #     ax1.set_title(animal_list[i], fontsize=8)
    #     ax1.plot(xvals[i], yvals[i])

    # ax.set_axis_bgcolor('none')
    # ax.set_xlabel('session', fontsize=8)

    # plt.suptitle('Lick Latencies, by session', fontsize=12)
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.95, hspace=0.50, wspace=0.20, bottom=0.04)
    # plt.show()