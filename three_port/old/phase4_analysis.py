import os
import multiprocessing
import math
import pymworks
import matplotlib.pyplot as plt
import numpy as np

def get_animals_and_their_session_filenames(path):
    '''
    Returns a dict with animal names as keys (it gets their names from the
        folder names in 'input' folder--each animal should have its own
        folder with .mwk session files) and a list of .mwk filename strings as
        values.
            e.g. {'V1': ['V1_140501.mwk', 'V1_140502.mwk']}

    :param path: a string of the directory name containing animals' folders
    '''
    #TODO maybe make this better, it's slow as hell and ugly
    result = {}
    dirs_list = [each for each in os.walk(path)]
    for each in dirs_list[1:]:
        files_list = each[2]
        animal_name = each[0].split("/")[len(each[0].split("/")) - 1]
        result[animal_name] = [] #list of filenames
        for filename in files_list:
            if not filename.startswith('.'): #dont want hidden files
                result[animal_name].append(filename)
    return result

def analyze_sessions(animals_and_sessions, graph_summary_stats=False):
    pool = multiprocessing.Pool(None)
    results = []
    for animal, sessions in animals_and_sessions.iteritems():
        result = pool.apply_async(get_data_for_figure,
            args=(animal, sessions))
        results.append(result)
    pool.close()
    pool.join()

    all_data = []
    for each in results:
        data_for_animal = each.get()
        all_data.append(data_for_animal)
        make_a_figure(data_for_animal)

    if graph_summary_stats:
        data = get_summary_stats_data(all_data)
        make_summary_stats_figure(data)

def get_data_for_figure(animal_name, sessions):
    all_trials = get_trials_from_all_sessions(animal_name, sessions)
    #make a dict where keys are (size, rotation) tuples and vals are a list of "success", "failure", or "ignore" strings
    trial_outcomes = make_list_of_behavior_outcomes_for_size_rot_grid(all_trials)
    pct_correct_data = get_pct_correct_for_animal(trial_outcomes)

    return {
        "animal_name": animal_name,
        "pct_correct_data": pct_correct_data
    }
def get_summary_stats_data(list_of_animal_data):
    #sum up all the data then divide by the sample size (number of animals) to get average
    result = {}
    for animal in list_of_animal_data:
        data = animal["pct_correct_data"]
        for (size, rotation), pct_correct in data.iteritems():
            try:
                result[(size, rotation)] += pct_correct #add animal data to sum
            except KeyError:
                result[(size, rotation)] = pct_correct
    #divide by sample size to get average
    for k, v in result.iteritems():
        result[k] = v/(len(list_of_animal_data))
    return result

def make_summary_stats_figure(data_for_all_animals):
    x = [-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0] #rotations
    y = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0] #sizes
    X, Y = np.meshgrid(x, y)
    #setup grid coordinates for mapping function

    #TODO dont rely on a hard-coded grid; look into numpy functions for this?
    grid = [
        [(15.0, -60.0), (20.0, -60.0), (25.0, -60.0), (30.0, -60.0), (35.0, -60.0), (40.0, -60.0), (45.0, -60.0)],
        [(15.0, -45.0), (20.0, -45.0), (25.0, -45.0), (30.0, -45.0), (35.0, -45.0), (40.0, -45.0), (45.0, -45.0)],
        [(15.0, -30.0), (20.0, -30.0), (25.0, -30.0), (30.0, -30.0), (35.0, -30.0), (40.0, -30.0), (45.0, -30.0)],
        [(15.0, -15.0), (20.0, -15.0), (25.0, -15.0), (30.0, -15.0), (35.0, -15.0), (40.0, -15.0), (45.0, -15.0)],
        [(15.0, 0.0), (20.0, 0.0), (25.0, 0.0), (30.0, 0.0), (35.0, 0.0), (40.0, 0.0), (45.0, 0.0)],
        [(15.0, 15.0), (20.0, 15.0), (25.0, 15.0), (30.0, 15.0), (35.0, 15.0), (40.0, 15.0), (45.0, 15.0)],
        [(15.0, 30.0), (20.0, 30.0), (25.0, 30.0), (30.0, 30.0), (35.0, 30.0), (40.0, 30.0), (45.0, 30.0)],
        [(15.0, 45.0), (20.0, 45.0), (25.0, 45.0), (30.0, 45.0), (35.0, 45.0), (40.0, 45.0), (45.0, 45.0)],
        [(15.0, 60.0), (20.0, 60.0), (25.0, 60.0), (30.0, 60.0), (35.0, 60.0), (40.0, 60.0), (45.0, 60.0)],
        [(15.0, 75.0), (20.0, 75.0), (25.0, 75.0), (30.0, 75.0), (35.0, 75.0), (40.0, 75.0), (45.0, 75.0)]
    ]
    #will store performance for each block in the grid
    intensity = [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
    ]
    #map grid (size, rotation) tuples to % correct, and add that or NaN to intensity vals
    for i in xrange(len(grid)):
        row = grid[i]
        for e in xrange(len(row)):
            sizerot_coords = row[e]
            try:
                pct_correct = data_for_all_animals[sizerot_coords]
            except KeyError:
                pct_correct = np.nan
            intensity[i].append(pct_correct)


    intensity = np.ma.masked_where(np.isnan(intensity), intensity)
    plt.close('all')
    plt.pcolormesh(X, Y, intensity.T, cmap="hot", vmin=40.0, vmax=100.0)
    bar = plt.colorbar()
    bar.set_label("% correct", rotation=270, labelpad=10)
    plt.axis('tight')
    plt.grid(True, which='minor')
    plt.title("All animals phase 4 performance")
    plt.xlabel("Stimulus rotation in depth (degrees)")
    plt.ylabel("Stimulus size (degrees visual angle)")
    plt.show()

def make_a_figure(data_for_animal):
    x = [-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0] #rotations
    y = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0] #sizes
    X, Y = np.meshgrid(x, y)
    #setup grid coordinates for mapping function

    #TODO dont rely on a hard-coded grid; look into numpy functions for this?
    grid = [
        [(15.0, -60.0), (20.0, -60.0), (25.0, -60.0), (30.0, -60.0), (35.0, -60.0), (40.0, -60.0), (45.0, -60.0)],
        [(15.0, -45.0), (20.0, -45.0), (25.0, -45.0), (30.0, -45.0), (35.0, -45.0), (40.0, -45.0), (45.0, -45.0)],
        [(15.0, -30.0), (20.0, -30.0), (25.0, -30.0), (30.0, -30.0), (35.0, -30.0), (40.0, -30.0), (45.0, -30.0)],
        [(15.0, -15.0), (20.0, -15.0), (25.0, -15.0), (30.0, -15.0), (35.0, -15.0), (40.0, -15.0), (45.0, -15.0)],
        [(15.0, 0.0), (20.0, 0.0), (25.0, 0.0), (30.0, 0.0), (35.0, 0.0), (40.0, 0.0), (45.0, 0.0)],
        [(15.0, 15.0), (20.0, 15.0), (25.0, 15.0), (30.0, 15.0), (35.0, 15.0), (40.0, 15.0), (45.0, 15.0)],
        [(15.0, 30.0), (20.0, 30.0), (25.0, 30.0), (30.0, 30.0), (35.0, 30.0), (40.0, 30.0), (45.0, 30.0)],
        [(15.0, 45.0), (20.0, 45.0), (25.0, 45.0), (30.0, 45.0), (35.0, 45.0), (40.0, 45.0), (45.0, 45.0)],
        [(15.0, 60.0), (20.0, 60.0), (25.0, 60.0), (30.0, 60.0), (35.0, 60.0), (40.0, 60.0), (45.0, 60.0)],
        [(15.0, 75.0), (20.0, 75.0), (25.0, 75.0), (30.0, 75.0), (35.0, 75.0), (40.0, 75.0), (45.0, 75.0)]
    ]
    #will store performance for each block in the grid
    intensity = [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
    ]
    #map grid (size, rotation) tuples to % correct, and add that or NaN to intensity vals
    for i in xrange(len(grid)):
        row = grid[i]
        for e in xrange(len(row)):
            sizerot_coords = row[e]
            try:
                pct_correct = data_for_animal["pct_correct_data"][sizerot_coords]
            except KeyError:
                pct_correct = np.nan
            intensity[i].append(pct_correct)


    intensity = np.ma.masked_where(np.isnan(intensity), intensity)
    plt.close('all')
    plt.pcolormesh(X, Y, intensity.T, cmap="hot", vmin=40.0, vmax=100.0)
    bar = plt.colorbar()
    bar.set_label("% correct", rotation=270, labelpad=10)
    plt.axis('tight')
    plt.grid(True, which='minor')
    plt.title(data_for_animal["animal_name"] + " phase 4 performance")
    plt.xlabel("Stimulus rotation in depth (degrees)")
    plt.ylabel("Stimulus size (degrees visual angle)")
    plt.show()

def get_pct_correct_for_animal(trial_outcomes):
    result = {}
    for (size, rotation), outcome_list in trial_outcomes.iteritems():
        pct_correct = get_pct_correct_from_outcome_list(outcome_list)
        result[(size, rotation)] = pct_correct
    return result

def get_pct_correct_from_outcome_list(outcome_list):
    success = 0
    failure = 0
    ignore = 0
    for outcome in outcome_list:
        if outcome == "success":
            success += 1
        elif outcome == "failure":
            failure += 1
        elif outcome == "ignore":
            ignore += 1
        else:
            pass
    total_trials = success + failure + ignore
    pct_correct = (float(success)/float(total_trials)) * 100
    return pct_correct

def make_list_of_behavior_outcomes_for_size_rot_grid(all_trials):
    sizes_and_rotations = {}
    for trial in all_trials:
        try:
            sizes_and_rotations[(trial["stm_size"], trial["stm_rotation"])].append(trial["behavior_outcome"])
        except KeyError:
            sizes_and_rotations[(trial["stm_size"], trial["stm_rotation"])] = [trial["behavior_outcome"]]
    return sizes_and_rotations

def get_trials_from_all_sessions(animal_name, sessions):
    print "Starting analysis for ", animal_name
    all_trials_all_sessions = []
    for session in sessions:
        trials = get_session_trials(input_dir, animal_name, session)
        all_trials_all_sessions += trials
    return all_trials_all_sessions

def get_session_trials(input_dir, animal_name, session_filename):
    '''
    Returns a time-ordered list of dicts, where each dict is info about a trial.
    e.g. [{"trial_num": 1,
           "behavior_outcome": "failure",
           "stm_size": 40.0,
           },
          {"trial_num": 2,
           "behavior_outcome": "success",
           "stm_size": 35.0
           }]

    :param animal_name: name of the animal string
    :param session_filename: filename for the session (string)
    '''

    #TODO: unfuck this: hard coded paths not ideal for code reuse
    # path = 'input/' + 'phase4/' + animal_name + '/' + session_filename
    path = os.path.join(input_dir, animal_name, session_filename)

    df = pymworks.open_file(path)
    events = df.get_events([
        "Announce_TrialStart",
        "Announce_TrialEnd",
        "success",
        "failure",
        "ignore",
        "stm_size",
        "stm_rotation_in_depth"]
    )

    trials = []
    trial_num = 1
    for index, event in enumerate(events):
        if (event.name == "Announce_TrialStart" and
        event.value == 1):
            trial = {
                "trial_num": trial_num,
                "stm_size": None,
                "behavior_outcome": None,
                "stm_rotation": None
            }

            try:
                if events[index - 1].name == "stm_size":
                    trial["stm_size"] = events[index - 1].value
            except IndexError:
                print "stm_size out of range for session", session_filename, \
                index
            try:
                if events[index - 1].name == "stm_rotation_in_depth":
                    trial["stm_rotation"] = float(events[index - 1].value)
            except IndexError:
                print "stm_rotation_in_depth out of range for session", session_filename, index
            try:
                if events[index - 2].name == "stm_size":
                    trial["stm_size"] = events[index - 2].value
            except IndexError:
                print "stm_size out of range for session", session_filename, index
            try:
                if events[index - 2].name == "stm_rotation_in_depth":
                    trial["stm_rotation"] = float(events[index - 2].value)
            except IndexError:
                print "stm_rotation_in_depth out of range for session", session_filename, index
            try:
                if events[index + 1].name in ["success", "failure", "ignore"]:
                    trial["behavior_outcome"] = events[index + 1].name
            except IndexError:
                print "beh_outcome out of range for session", session_filename,\
                 index
            if (trial["stm_size"] is not None and
            trial["behavior_outcome"] is not None and
            trial["stm_rotation"] is not None):
                trials.append(trial)
                trial_num += 1
    return trials

if __name__ == "__main__":
    import sys
    input_dir = sys.argv[1]
    animals_and_sessions = get_animals_and_their_session_filenames(input_dir)
    analyze_sessions(animals_and_sessions, graph_summary_stats=True)

