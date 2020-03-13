#!/usr/bin/env python2

'''
This script parses each animal's behavioral outcome data by session.

Saves a .pkl in processed_path:

For each session (each .mwk file), creates a time-ordered list of dicts, where each dict is info about a trial.

{} [{"trial_num": 1,
       "behavior_outcome": "failure",
       "stm_pos_x": 7.5,
       },
      {"trial_num": 2,
       "behavior_outcome": "success",
       "stm_pos_x": -7.5
       }]

This dict is saved as a .pkl for each animal in its processed_path (e.g., "path_to_processed/animal").

Also creates a dict of animal_session info as a .pkl in path_to_processed, so that only new files are processed.

Option to plot, but use "plot_by_session" instead...

input : /path/to/input/folders
'''

import os
import multiprocessing
import datetime
import pymworks
import matplotlib.pyplot as plt

import sys
import cPickle as pkl

def get_animals_and_their_session_filenames(path, append_data=True):
    '''
    Returns a dict with animal names as keys (it gets their names from the
        folder names in 'input' folder--each animal should have its own
        folder with .mwk session files) and a list of .mwk filename strings as
        values.
            e.g. {'V1': ['V1_140501.mwk', 'V1_140502.mwk']}

    :param path: a string of the directory name containing animals' folders
    '''
    # Check if experiment analyzed before, and load file list if so:
    if append_data is True:
        fname = os.path.join(out_dir, 'animals_session.pkl')
        if os.path.isfile(fname): # already processed this exp before
            with open(fname, 'rb') as f:
                result = pkl.load(f)
        else:
            result = {}
    else:
        result = {}

    new_data = {}

    dirs_list = [each for each in os.walk(path)]
    for each in dirs_list[1:]:
        files_list = each[2]
        animal_name = each[0].split("/")[len(each[0].split("/")) - 1]

        if animal_name not in result.keys():
            result[animal_name] = []
            # new_data[animal_name] = []
        new_data[animal_name] = []
        # Only grab new files:
        for filename in files_list:
            if not filename.startswith('.') and filename not in result[animal_name]: #dont want hidden files
                result[animal_name].append(filename)
                new_data[animal_name].append(filename)

    # print("Starting analysis for animals:")
    # for each in result.keys():
    #     print(each)
    print("Starting analysis for animals:")
    for each in new_data.keys():
        print(each)

    # save animal-session info so that next time, only new sessions are analyzed:
    # fname = os.path.join(out_dir, 'animals_session.pkl')
    with open(fname, 'wb') as f:
        pkl.dump(result, f, protocol=pkl.HIGHEST_PROTOCOL) #protocol=pkl.HIGHEST_PROTOCOL) 

    return new_data

def analyze_sessions(animals_and_sessions, graph_as_group=False):
    '''
    Starts analysis for each animals' sessions in a new process to use cores.
        We don't want to wait all day for this, y'all.

    :param animals_and_sessions: a dict with animal names as keys and
        a list of their session filenames as values.
    '''
    #use all CPU cores to process data
    pool = multiprocessing.Pool(None)

    results = [] #list of multiprocessing.AsyncResult objects
    for animal, sessions in animals_and_sessions.iteritems():
        result = pool.apply_async(analyze_animal_sessions,
            args=(animal, sessions))
        results.append(result)
    pool.close()
    pool.join() #block until all the data has been processed
    if graph_as_group:
        raise NotImplementedError, "Group graphing coming soon..."
    #
    print("Graphing session data...")
    for each in results:
        data_for_animal = each.get() #returns analyze_animal_sessions result
        #make_a_figure(data_for_animal)

    print("Finished")

def make_a_figure(data):
    '''
    Shows a graph of an animal's performance and trial info.

    :param data: a dict with x and y value lists returned by
        analyze_animal_sessions()
    '''

    f, ax_arr = plt.subplots(2, 2) #make 4 subplots for figure
    f.suptitle(data["animal_name"]) #set figure title to animal's name
    f.subplots_adjust(bottom=0.08, hspace=0.4) #fix overlapping labels

    ax_arr[0, 0].plot(data["x_vals"], data["total_pct_correct_y_vals"], "bo")
    ax_arr[0, 0].set_title("% correct - all trials")
    ax_arr[0, 0].axis([0, len(data["x_vals"]), 0.0, 100.0])
    ax_arr[0, 0].set_xlabel("Session number")

    ax_arr[0, 1].plot(data["x_vals"], data["pct_corr_in_center_y_vals"], "bo")
    ax_arr[0, 1].set_title("% correct - trials with stim in center")
    ax_arr[0, 1].axis([0, len(data["x_vals"]), 0.0, 100.0])
    ax_arr[0, 1].set_xlabel("Session number")

    ax_arr[1, 0].plot(data["x_vals"], data["total_trials_y_vals"], "bo")
    ax_arr[1, 0].set_title("Total trials in session")
    ax_arr[1, 0].axis([0, len(data["x_vals"]), 0, \
        max(data["total_trials_y_vals"])])
        #largest y axis tick is largest number of trials in a session
    ax_arr[1, 0].set_xlabel("Session number")

    ax_arr[1, 1].plot(data["x_vals"], data["num_trials_stim_in_center_y_vals"],
        "bo")
    ax_arr[1, 1].set_title("Total trials with stim in center of the screen")
    ax_arr[1, 1].axis([0, len(data["x_vals"]), 0, \
        max(data["total_trials_y_vals"])])
        #largest y axis tick is largest number of trials in a session
        #so it's easier to compare total trials and total trials with
        #stim in center
    ax_arr[1, 1].set_xlabel("Session number")

    plt.show() #show each figure, user can save if he/she wants

    #make plot of the % of trials in center
    plt.close("all")
    plt.plot(data["x_vals"], data["pct_trials_stim_in_center"], "bo")
    plt.axis([0, len(data["x_vals"]), 0.0, 100.0])
    plt.title("% trials with stim in center " + data["animal_name"])
    plt.xlabel("Session number")
    plt.show()

def analyze_animal_sessions(animal_name, sessions):
    '''
    Analyzes one animals' sessions and outputs dict with x and y value lists
    for different types of graphs, e.g. percent correct, total trials, etc.
    See return dict below.
    This is wrapped by analyze_sessions() so it can run in a process on
    each CPU core.

    :param animal_name: name of the animal (string)
    :param sessions: the animal's session filenames (list of strings)
    '''

    list_of_session_stats = get_stats_for_each_session(animal_name, sessions)

    x_vals = [each["session_number"] for each in list_of_session_stats]
    pct_corr_whole_session_y = [each["pct_correct_whole_session"] for each in \
        list_of_session_stats]
    pct_corr_in_center_y = [each["pct_correct_stim_in_center"] for each in \
        list_of_session_stats]
    total_num_trials_y = [each["total_trials"] for each in \
        list_of_session_stats]
    total_trials_stim_in_center_y = [each["trials_with_stim_in_center"] for \
        each in list_of_session_stats]
    pct_trials_stim_in_center = [each["pct_trials_stim_in_center"] for \
        each in list_of_session_stats]

    return {"x_vals": x_vals, #x axis will be session number for all graphs
            "total_pct_correct_y_vals": pct_corr_whole_session_y,
            "pct_corr_in_center_y_vals": pct_corr_in_center_y,
            "total_trials_y_vals": total_num_trials_y,
            "num_trials_stim_in_center_y_vals": total_trials_stim_in_center_y,
            "pct_trials_stim_in_center": pct_trials_stim_in_center,
            "animal_name": animal_name}

def get_stats_for_each_session(animal_name, sessions):
    '''
    Returns a list of dicts with statistics about each session for an
    animal. e.g.
    result = [{
        'session_number': 1,
        'ignores': 2,
        'successes': 2,
        'failures': 0,
        'pct_correct_whole_session': 50.0,
        'pct_correct_stim_in_center': 50.0,
        'total_trials': 4,
        'trials_with_stim_in_center': 4,
        'pct_trials_stim_in_center': 100.0
    },

    #Note the NoneType values in this session

    { 'session_number': 2,
      'ignores': 0,
      'successes': 0,
      'failures': 0,
      'pct_correct_whole_session': None,
      'pct_correct_stim_in_center': None,
      'total_trials': 0,
      'trials_with_stim_in_center': 0,
      'pct_trials_stim_in_center': None}]

    NOTE: if there are no trials for the denominator of a percentage key
        (e.g. pct_correct_stim_in_center), the key's value is set to None.
        Behavior outcomes (e.g. ignores, successes, etc.) with no occurances
        are left with value = 0.
    '''

    result = []
    session_num = 1
    for session in sessions:
        all_trials = get_session_statistics(animal_name, session)

        #make dict to store session data
        session_result = {"session_number": session_num,
                          "total_trials": len(all_trials),
                          "filename": session}
        #go through each trial to get stats
        all_success = 0
        all_failure = 0
        all_ignore = 0
        success_in_center = 0
        failure_in_center = 0
        ignore_in_center = 0
        for trial in all_trials:
            if trial["behavior_outcome"] == "success":
                if trial["stm_pos_x"] == 0.0:
                    success_in_center += 1
                all_success += 1
            elif trial["behavior_outcome"] == "failure":
                if trial["stm_pos_x"] == 0.0:
                    failure_in_center += 1
                all_failure += 1
            elif trial["behavior_outcome"] == "ignore":
                if trial["stm_pos_x"] == 0.0:
                    ignore_in_center += 1
                all_ignore += 1
            else:
                print "No behavior_outcome for %s %s\
                , trial number %s" % (animal_name, session, trial["trial_num"])

        #add session data to session result dict
        session_result["successes"] = all_success
        session_result["failures"] = all_failure
        session_result["ignores"] = all_ignore
        try:
            # session_result["pct_correct_whole_session"] = (float(all_success)/\
            #     (float(all_success + all_ignore + all_failure))) * 100.0
            session_result["pct_correct_whole_session"] = (float(all_success)/\
                (float(all_success + all_failure))) * 100.0 # + all_ignore

        except ZeroDivisionError:
            session_result["pct_correct_whole_session"] = None
        try:
            session_result["pct_correct_stim_in_center"] = \
                (float(success_in_center)/(float(success_in_center + \
                failure_in_center))) * 100.0 # + ignore_in_center
        except ZeroDivisionError:
            session_result["pct_correct_stim_in_center"] = None

        session_result["trials_with_stim_in_center"] = \
            success_in_center + failure_in_center + ignore_in_center
        try:
            session_result["pct_trials_stim_in_center"] = \
                (float(session_result["trials_with_stim_in_center"])/\
                (float(len(all_trials)))) * 100.0
        except ZeroDivisionError:
            session_result["pct_trials_stim_in_center"] = None

        #add each session's result dict to the list of session result dicts
        result.append(session_result)

        session_num += 1
    return result

def get_session_statistics(animal_name, session_filename):
    '''
    Returns a time-ordered list of dicts, where each dict is info about a trial.
    e.g. [{"trial_num": 1,
           "behavior_outcome": "failure",
           "stm_pos_x": 7.5,
           },
          {"trial_num": 2,
           "behavior_outcome": "success",
           "stm_pos_x": -7.5
           }]
    NOTE: trial_num: 1 corresponds to the FIRST trial in the session,
    and trials occur when Announce_TrialStart and Announce_TrialEnd
    events have success, failure, or ignore events between them with
    value=1.

    :param animal_name: name of the animal string
    :param session_filename: filename for the session (string)
    '''

    #TODO: unfuck this: hard coded paths not ideal for code reuse
    #path = 'input/' + animal_name + '/' + session_filename
    path = os.path.join(data_dir, animal_name, session_filename)

    df = pymworks.open_file(path)
    events = df.get_events(["Announce_TrialStart", "Announce_TrialEnd",
                            "success", "failure", "ignore", "stm_pos_x"])

    result = []
    index = 0
    temp_events = []
    last_announce = None
    trial_num = 0
    while index < len(events):
        if events[index].name == "Announce_TrialStart":
            temp_events = []
            last_announce = "Announce_TrialStart"

        elif events[index].name == "Announce_TrialEnd":
            if last_announce == "Announce_TrialStart":
                trial_result = {}
                for ev in temp_events:
                    if ev.name == "success" and ev.value == 1:
                        trial_result["behavior_outcome"] = "success"
                    elif ev.name == "failure" and ev.value == 1:
                        trial_result["behavior_outcome"] = "failure"
                    elif ev.name == "ignore" and ev.value == 1:
                        trial_result["behavior_outcome"] = "ignore"
                    elif ev.name == "stm_pos_x":
                        trial_result["stm_pos_x"] = ev.value
                    else:
                        pass
                if "behavior_outcome" in trial_result:
                    trial_num += 1
                    trial_result["trial_num"] = trial_num
                    result.append(trial_result)

            last_announce = "Announce_TrialEnd"

        else:
            temp_events.append(events[index])
        index += 1
    #FYI, testing showed some good filtering of weird events here...
    #blah = df.get_events(["success", "failure", "ignore"])
    #print "EVENTS EQUAL? ", len(result) == len(blah) - 6, session_filename
    #subtract 6 because session initialization emits 2 behavior outcomes per
    #outcome type
    #print len(result), len(blah) - 6
    #lines above unequal in 6/77 sessions for AB3&7 because of random behavior
    #outcome events firing in rapid succession. They happens within a couple
    #microseconds of one another so filtering these out is probably good

    # save each animal's analyzed session info:
    out_dir_by_animal = os.path.join(out_dir, '%s' % animal_name)
    if not os.path.exists(out_dir_by_animal):
        os.makedirs(out_dir_by_animal)

    fname = os.path.join(out_dir_by_animal, '%s.pkl' % session_filename)
    with open(fname, 'wb') as f:
        pkl.dump(result, f, protocol=pkl.HIGHEST_PROTOCOL) #protocol=pkl.HIGHEST_PROTOCOL) 

    # make_a_figure(data_for_animal)

    return result

if __name__ == "__main__":
    global data_dir
    data_dir = sys.argv[1]

    global out_dir
    out_dir = os.path.join(os.path.split(data_dir)[0], 'processed')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    animals_and_sessions = get_animals_and_their_session_filenames(data_dir)
    analyze_sessions(animals_and_sessions)