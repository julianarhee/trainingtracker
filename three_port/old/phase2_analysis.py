import os
import multiprocessing
import datetime
import random
import math
import pymworks
import matplotlib.pyplot as plt

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
        result = pool.apply_async(get_data_for_figure,
            args=(animal, sessions))
        results.append(result)
    pool.close()
    pool.join() #block until all the data has been processed

    all_data = []
    for each in results:
        data_for_animal = each.get() #returns get_data_for_figure result
        all_data.append(data_for_animal)
        tmp = make_a_figure(data_for_animal)

    if graph_summary_stats:
        data, bins_in_order = get_data_for_summary_statistics_graph(all_data)
        make_summary_stats_figure(data, bins_in_order)

def make_summary_stats_figure(data, bins_in_order, colors=[
        "tomato",
        "turquoise",
        "violet",
        "springgreen",
        "yellow",
        "seagreen",
        "royalblue",
        "indigo",
        "sienna",
        "slategray",
        "yellowgreen",
        "orange",
        "tan",
        "red",
        "darkred",
        "green",
        "orangered",
        "black"
    ]):

    plt.close('all')

    for i in range(len(bins_in_order)):
        bin = bins_in_order[i]
        color = colors[i]
        x = data[bin]["x_vals_sizes"]
        y = data[bin]["y_vals_pct_correct"]
        err = data[bin]["y_vals_error"]
        plt.errorbar(
            x,
            y,
            yerr=err,
            fmt='-o',
            color=color,
            label=bin,
            linewidth=2.0
        )

    plt.xlim(0.0, 45.0)
    plt.ylim(0.0, 100.0)
    plt.grid(axis="y")
    plt.xlabel("Stimulus size (degrees visual angle)")
    plt.ylabel("Percent correct +/- std_dev")
    plt.title("All animals percent correct")
    plt.legend(loc="lower right", title="Sessions")

    plt.show()

def get_data_for_summary_statistics_graph(animal_data_list):
    all_bins = get_bins_in_common_for_all_animals(animal_data_list)
    all_sizes = get_sizes_in_common_for_all_animals(animal_data_list, all_bins)

    #setup summary dict with a list for all sizes in all bins
    summary = {}
    for bin in all_bins:
        summary[bin] = {}
        for size in all_sizes[bin]:
            summary[bin][size] = []

    for bin in all_bins:
        for animal_stats in animal_data_list:
            pct_correct_data = animal_stats["pct_correct_bootstraph_graph_data"]
            sizes = pct_correct_data[bin]["x_vals"]
            pct_corrects = pct_correct_data[bin]["y_vals_observed_pct_correct"]
            for size, pct in zip(sizes, pct_corrects):
                if size in all_sizes[bin]:
                    summary[bin][size].append(pct)

    summary = do_stats_for_summary(summary)
    return summary, all_bins

def do_stats_for_summary(summary_stats):
    result = {}
    for bin, size_data_list in summary_stats.iteritems():
        result[bin] = {
            "x_vals_sizes": [],
            "y_vals_pct_correct": [],
            "y_vals_error": []
        }
        for size, pct_correct_list in size_data_list.iteritems():
            avg, std_dev = calc_summary_stats(pct_correct_list)
            result[bin]["x_vals_sizes"].append(float(size))
            result[bin]["y_vals_pct_correct"].append(avg)
            result[bin]["y_vals_error"].append(std_dev)

    result = sort_summary_stats_into_lists(result)
    return result

def sort_summary_stats_into_lists(result):
    final_result = {}
    for bin, data_lists in result.iteritems():
        x = data_lists["x_vals_sizes"]
        y = data_lists["y_vals_pct_correct"]
        z = data_lists["y_vals_error"]
        xyz = zip(x, y, z)
        xyz.sort()
        x, y, z = zip(*xyz)
        final_result[bin] = {
            "x_vals_sizes": x,
            "y_vals_pct_correct": y,
            "y_vals_error": z
        }
    return final_result

def calc_summary_stats(list_of_floats):
    mean = math.fsum(list_of_floats)/len(list_of_floats)
    variance = (math.fsum([(fl - mean)**2.0 for fl in list_of_floats]))/(len(list_of_floats) - 1)
    std_dev = math.sqrt(variance)
    return mean, std_dev

def get_bins_in_common_for_all_animals(animal_data_list):
    all_bins = []
    for animal in animal_data_list:
        bins_in_order = animal["bootstrap_bins_in_order"]
        all_bins.append(bins_in_order)
    fewest_bins = sorted(all_bins, key=len)[0]
    return fewest_bins

def get_sizes_in_common_for_all_animals(animal_data_list, all_bins):
    result = {}
    for bin in all_bins:
        all_sizes = []
        for animal in animal_data_list:
            sizes = animal["pct_correct_bootstraph_graph_data"][bin]["x_vals"]
            all_sizes.append(sizes)
        fewest_sizes = sorted(all_sizes, key=len)[0]
        result[bin] = fewest_sizes
    return result

def make_a_figure(data, colors=["tomato", "turquoise", "violet", "springgreen",\
    "yellow", "seagreen", "royalblue", "indigo", "sienna", \
    "slategray", "yellowgreen", "orange", "tan", "red", "darkred", \
    "green", "orangered", "black"]):
    '''
    Shows a graph of an animal's performance and trial info.

    :param data: a dict with x and y value lists returned by
        get_data_for_figure()
    '''
    plt.close('all')

    f, ax_arr = plt.subplots(2, 1) #make 2 subplots for figure
    f.suptitle(data["animal_name"]) #set figure title to animal's name

    x = data["x_vals"]
    sizes = data["all_sizes"]
    max_y = 0

    for i in range(len(sizes)):
        #plot each size's d prime across sessions
        y = data["y_vals_d_prime_by_size"][sizes[i]]
        ax_arr[0].plot(x, y, color=colors[i], label=sizes[i], linewidth=3.0)
        #plot total number of trials with stim size across sessions
        y = data["y_vals_total_trials_by_size"][sizes[i]]
        if max(y) > max_y:
            max_y = max(y)
        ax_arr[1].plot(x, y, color=colors[i], label=sizes[i], linewidth=3.0)

    ax_arr[0].legend()
    ax_arr[1].legend()
    ax_arr[0].set_xlim(0, len(x))
    ax_arr[0].set_ylim(-1.0, 1.0)
    ax_arr[1].set_xlim(0, len(x))
    ax_arr[1].set_ylim(0, max_y)
    ax_arr[1].set_xlabel("Session number")
    ax_arr[0].set_ylabel("Discriminability index (d')")
    ax_arr[1].set_ylabel("Number of trials")

    plt.show()
    plt.close('all')

    bs_data = data["bootstrap_graph_data"]
    session_bins_in_order = data["bootstrap_bins_in_order"]
    for i in range(len(session_bins_in_order)):
        bin = session_bins_in_order[i]
        color = colors[i]
        x = bs_data[bin]["x_vals"]
        y = bs_data[bin]["observed_d_primes"]
        error = bs_data[bin]["y_vals_bs_std_dev"]
        plt.errorbar(x, y, yerr=error, fmt="-o", color=color,
            label=bin, linewidth=2.0)
    plt.xlim(0.0, 45.0)
    plt.ylim(-1.0, 1.0)
    plt.xlabel("Stimulus size (degrees visual angle)")
    plt.ylabel("Discriminability index (d') +/- std_dev")
    plt.title(data["animal_name"] + " binned performance\
    (bootstrapped std_dev)")
    plt.legend(loc="lower right", title="Sessions")

    plt.show()
    plt.close('all')

    bs_pct_correct_data = data["pct_correct_bootstraph_graph_data"]
    session_bins_in_order = data["bootstrap_ordered_bins"]
    for i in range(len(session_bins_in_order)):
        bin = session_bins_in_order[i]
        color = colors[i]
        x = bs_pct_correct_data[bin]["x_vals"]
        y = bs_pct_correct_data[bin]["y_vals_observed_pct_correct"]
        error = bs_pct_correct_data[bin]["err_vals_bs_std_devs"]
        plt.errorbar(x, y, yerr=error, fmt='-o', color=color,
            label=bin, linewidth=2.0)
    plt.xlim(0.0, 45.0)
    plt.ylim(0.0, 100.0)
    plt.xlabel("Stimulus size (degrees visual angle)")
    plt.ylabel("Percent correct +/- std_dev")
    plt.title(data["animal_name"] + " binned performance with bootstrapped\
    std_dev")
    plt.legend(loc="lower right", title="Sessions")

    plt.show()
    plt.close('all')

    num_trials_data = data["binned_graph_trial_nums"]
    session_bins_in_order = data["bootstrap_ordered_bins"]
    for i in range(len(session_bins_in_order)):
        bin = session_bins_in_order[i]
        color = colors[i]
        x = num_trials_data[bin]["x_vals_sizes"]
        y = num_trials_data[bin]["y_vals_total_trials"]
        plt.plot(x, y, "-o", color=color, label=bin, linewidth=2.0)
    plt.xlim(0.0, 45.0)
    #plt.ylim(0.0, ) will make ylim same for all animals
    plt.xlabel("Stimulus size (degrees visual angle)")
    plt.ylabel("Sample size (total trials)")
    plt.title(data["animal_name"] + " sample size for all stimuli")
    plt.legend(loc="upper right", title="Sessions")

    plt.show()

def sort_by_size_from_size_strings(list_of_size_strings):
    '''
    Returns a descending-order list of size strings from a list of size
        strings. Use this function so each size always has the same color
        across animals in make_a_figure()

    :param list_of_size_strings: a list of stimulus size strings where each
        string can be converted to a float.
    '''
    tmp = map(float, list_of_size_strings)
    tmp.sort(reverse=True)
    return map(str, tmp)

def get_data_for_figure(animal_name, sessions):
    '''
    Analyzes one animals' sessions and outputs dict with x and y value lists
    for different types of graphs, e.g. percent correct, total trials, etc.
    See return dict below.
    This is wrapped by analyze_sessions() so it can run in a process on
    each CPU core.

    Returns a dict with x_vals list for x axes (session number for all graphs).
        The by_size keys store dicts with the stimulus size as keys and their
        list of y values as values.

        For example,

        {
            "x_vals": [1, 2, 3, 4],
            "all_sizes": ["40.0", "37.5"],
            "animal_name": "AB1",
            "y_vals_total_trials_by_size": {
                "40.0": [2, 2, 4, 8], #total trials size 40.0 in each session
                "37.5": [0, 0, 2, 4] #first 2 sessions had no 37.5
                                     #degree vis. angle stimuli
            }
            "y_vals_d_prime_by_size": {
                "40.0": [1.0, 0.8, 1.0, 1.0], #d prime for trials of size 40.0
                                              #in each session
                "37.5": [None, None, 1.0, 0.8]
            }
        }

        NOTE: if there's no d_prime for a size in a session, the value is None
            (None values don't get graphed). If there are no trials with
            stimulus size for a session, that session's total_trials = 0 in the
            list.

    :param animal_name: name of the animal (string)
    :param sessions: the animal's session filenames (list of strings)
    '''

    list_of_session_stats = get_stats_for_each_session(animal_name, sessions)
    all_sizes_for_all_sessions = get_sizes_in_stats_list(list_of_session_stats)
    bs, bins_in_order = \
        get_bootstrapped_d_prime_and_std_dev(list_of_session_stats)
    bs = make_lists_for_binned_bootstrap_graph(bs, all_sizes_for_all_sessions)

    bs_pct_correct, ordered_bins = \
        get_bootstrapped_pct_correct_and_std_dev(list_of_session_stats)
    bs_pct_correct = \
        make_lists_for_binned_bootstrap_pct_correct_graph(bs_pct_correct,
            all_sizes_for_all_sessions)

    x_vals = [each["session_number"] for each in list_of_session_stats]
    y_vals_d_prime = {}
    y_vals_num_trials = {}
    for stim_size in all_sizes_for_all_sessions:
        y_vals_d_prime[stim_size] = []
        y_vals_num_trials[stim_size] = []
        for session in list_of_session_stats:
            if stim_size in session["d_prime_by_size"]:
                y_vals_d_prime[stim_size].append(session["d_prime_by_size"]\
                    [stim_size])
            else:
                y_vals_d_prime[stim_size].append(None)

            if stim_size in session["total_trials_by_size"]:
                y_vals_num_trials[stim_size].append(session[\
                    "total_trials_by_size"][stim_size])
            else:
                y_vals_num_trials[stim_size].append(0)

    binned_graph_trial_nums = get_trial_nums_for_binned_graph(
        y_vals_num_trials,
        bins_in_order)

    print "Finished analysis for " + animal_name

    return {
        "x_vals": x_vals, #x axis will be session number for all graphs
        "all_sizes": all_sizes_for_all_sessions,
        "y_vals_total_trials_by_size": y_vals_num_trials,
        "y_vals_d_prime_by_size": y_vals_d_prime,
        "animal_name": animal_name,
        "bootstrap_graph_data": bs,
        "bootstrap_bins_in_order": bins_in_order,
        "bootstrap_ordered_bins": ordered_bins, #dont really need this...
        "pct_correct_bootstraph_graph_data": bs_pct_correct,
        "binned_graph_trial_nums": binned_graph_trial_nums
    }

def get_trial_nums_for_binned_graph(
    y_vals_total_trials_by_size,
    bins_in_order,
    sessions_per_bin=8):
    '''
    returns dict like this:
    {
        "1-10": {
            "x_vals_sizes": [40.0, 35.0, 37.0, 15.0, 20.0, etc],
            "y_vals_total_trials": [...]
        }
        "11-20": etc etc
    }
    '''
    result = {}
    lower, upper = 0, sessions_per_bin
    for bin in bins_in_order:
        result[bin] = {
            "x_vals_sizes": [],
            "y_vals_total_trials": []
        }
        for size, list_of_trials in y_vals_total_trials_by_size.iteritems():
            result[bin]["x_vals_sizes"].append(float(size))
            total_trials_for_size = sum(list_of_trials[lower:upper])
            result[bin]["y_vals_total_trials"].append(total_trials_for_size)

        lower, upper = upper, upper + sessions_per_bin
    #data added to result but not in order yet
    #x, y vals have to be sorted for matplotlib
    for bin in bins_in_order:
        result[bin]["x_vals_sizes"], result[bin]["y_vals_total_trials"] = \
            sort_x_y_pairs_by_x_val(
                result[bin]["x_vals_sizes"],
                result[bin]["y_vals_total_trials"]
            )
    return result

def sort_x_y_pairs_by_x_val(x_vals_list, y_vals_list):
    '''
    x_vals_list should be either ints or floats
    '''
    xy = zip(x_vals_list, y_vals_list)
    xy.sort()
    new_x = []
    new_y = []
    for x, y in xy:
        new_x.append(x)
        new_y.append(y)
    return new_x, new_y

def get_bootstrapped_pct_correct_and_std_dev(
    session_stats_list,
    sessions_per_bin=8):
    stats_in_bins = split_list_into_sublists(
        session_stats_list,
        sessions_per_bin)
    bin_stats = {}
    bins_in_order = []
    low, up = 1, sessions_per_bin
    for bin in stats_in_bins:
        bin_str = str(low) + "-" + str(up)
        observed_pct_correct, bootstrapped_std_dev = calc_pct_correct(bin)
        bin_stats[bin_str] = {
            "observed_pct_correct": observed_pct_correct,
            "bootstrapped_pct_correct_std_dev": bootstrapped_std_dev
        }
        bins_in_order.append(bin_str)
        low, up = up + 1, up + sessions_per_bin
    return bin_stats, bins_in_order

def make_lists_for_binned_bootstrap_pct_correct_graph(bin_stats, all_sizes):
    result = {}
    for bin in bin_stats:
        result[bin] = {
            "x_vals": all_sizes,
            "y_vals_observed_pct_correct": [],
            "err_vals_bs_std_devs": []
        }
        for size in all_sizes:
            try:
                result[bin]["y_vals_observed_pct_correct"].append\
                (bin_stats[bin]["observed_pct_correct"][size])
            except KeyError:
                result[bin]["y_vals_observed_pct_correct"].append(None)
            try:
                result[bin]["err_vals_bs_std_devs"].append\
                (bin_stats[bin]["bootstrapped_pct_correct_std_dev"][size])
            except KeyError:
                result[bin]["err_vals_bs_std_devs"].append(None)
    result = removeNoneTypesPctCorrect(result)
    return result

def removeNoneTypesPctCorrect(result):
    final_result = {}
    for bin, stats in result.iteritems():
        x = stats["x_vals"]
        y = stats["y_vals_observed_pct_correct"]
        err = stats["err_vals_bs_std_devs"]

        new_x, new_y, new_err = [], [], []
        for x, y, err in zip(x, y, err):
            if ((x is not None) and (y is not None) and (err is not None)):
                new_x.append(x)
                new_y.append(y)
                new_err.append(err)

        final_result[bin] = {
            "x_vals": new_x,
            "y_vals_observed_pct_correct": new_y,
            "err_vals_bs_std_devs": new_err
        }
    return final_result

def calc_pct_correct(bin):
    bin_data = get_bin_data_for_each_stim_size(bin)
    observed_pct_correct_by_size = calc_real_pct_correct(bin_data)
    std_dev_by_size = run_bootstrap_resample_pct_correct(bin_data)
    return observed_pct_correct_by_size, std_dev_by_size

def calc_real_pct_correct(bin_data):
    real_pct_correct_by_size = {}
    for stim_size in bin_data:
        total = float(bin_data[stim_size]["success"] + \
            bin_data[stim_size]["failure"] + bin_data[stim_size]["ignore"])
        try:
            pct_correct = (float(bin_data[stim_size]["success"])/total) * 100.0
        except ZeroDivisionError:
            pct_correct = None
        real_pct_correct_by_size[stim_size] = pct_correct
    return real_pct_correct_by_size

def run_bootstrap_resample_pct_correct(bin_data, iterations=10000):
    behavior_lists_by_size = make_lists_for_resampling(bin_data)
    bootstrapped_pct_correct_list_by_size = {}
    for stim_size in behavior_lists_by_size.keys():
        bootstrapped_pct_correct_list_by_size[stim_size] = []

    for i in xrange(iterations):
        for size, real_outcomes in behavior_lists_by_size.iteritems():
            successes, failures, ignores = 0, 0, 0
            for e in xrange(len(real_outcomes)):
                chosen = random.choice(real_outcomes)
                if chosen == "success":
                    successes += 1
                elif chosen == "failure":
                    failures += 1
                elif chosen == "ignore":
                    ignores += 1
                else:
                    print "wut"
            total = float(successes + failures + ignores)
            try:
                bs_pct_correct = (float(successes)/total) * 100.0
            except ZeroDivisionError:
                bs_pct_correct = None
            bootstrapped_pct_correct_list_by_size[size].append(bs_pct_correct)
    std_dev_by_size = calc_std_devs(bootstrapped_pct_correct_list_by_size)
    return std_dev_by_size

def get_bootstrapped_d_prime_and_std_dev(
    session_stats_list,
    sessions_per_bin=8):
    stats_in_bins = split_list_into_sublists(
        session_stats_list,
        sessions_per_bin)
    bin_stats = {}
    bins_in_order = []
    low, up = 1, sessions_per_bin
    for bin in stats_in_bins:
        bin_str = str(low) + "-" + str(up)
        observed_d_prime, bootstrapped_std_dev = calculate_d_prime(bin)
        bin_stats[bin_str] = {
            "observed_d_prime": observed_d_prime,
            "bootstrapped_std_dev": bootstrapped_std_dev
        }
        bins_in_order.append(bin_str)
        low, up = up + 1, up + sessions_per_bin
    return bin_stats, bins_in_order

def make_lists_for_binned_bootstrap_graph(bin_stats, all_sizes):
    result = {}
    for bin in bin_stats:
        result[bin] = {
            "x_vals": all_sizes,
            "observed_d_primes": [],
            "y_vals_bs_std_dev": []
        }
        for size in all_sizes:
            try:
                result[bin]["observed_d_primes"].append\
                    (bin_stats[bin]["observed_d_prime"][size])
            except KeyError:
                result[bin]["observed_d_primes"].append(None)
            try:
                result[bin]["y_vals_bs_std_dev"].append\
                    (bin_stats[bin]["bootstrapped_std_dev"][size])
            except KeyError:
                result[bin]["y_vals_bs_std_dev"].append(None)

    result = removeNoneTypes(result)
    return result

def removeNoneTypes(result):
    final_result = {}
    for bin, stats in result.iteritems():
        x = stats["x_vals"]
        y = stats["observed_d_primes"]
        err = stats["y_vals_bs_std_dev"]

        new_x, new_y, new_err = [], [], []
        for x, y, err in zip(x, y, err):
            if ((x is not None) and (y is not None) and (err is not None)):
                new_x.append(x)
                new_y.append(y)
                new_err.append(err)

        final_result[bin] = {
            "x_vals": new_x,
            "observed_d_primes": new_y,
            "y_vals_bs_std_dev": new_err
        }
    return final_result

def split_list_into_sublists(session_stats_list, sessions_per_bin):
    new_list = []
    while len(session_stats_list) >= sessions_per_bin:
        new_list.append(session_stats_list[:sessions_per_bin])
        session_stats_list = session_stats_list[sessions_per_bin:]
    return new_list

def calculate_d_prime(bin):
    bin_data = get_bin_data_for_each_stim_size(bin)
    observed_d_prime_by_size = calc_real_d_prime(bin_data)
    std_dev_by_size = run_bootstrap_resample(bin_data)
    return observed_d_prime_by_size, std_dev_by_size

def run_bootstrap_resample(bin_data, iterations=10000):
    behavior_lists_by_size = make_lists_for_resampling(bin_data)
    bootstrapped_d_prime_list_by_size = {}
    for stim_size in behavior_lists_by_size.keys():
        bootstrapped_d_prime_list_by_size[stim_size] = []

    for i in xrange(iterations):
        for size, real_outcomes in behavior_lists_by_size.iteritems():
            successes = 0
            failures = 0
            for e in xrange(len(real_outcomes)):
                chosen = random.choice(real_outcomes)
                if chosen == "success":
                    successes += 1
                elif chosen == "failure":
                    failures += 1
                else:
                    pass
            total = float(successes + failures)
            try:
                bs_d_prime = float(successes)/total - \
                    float(failures)/total
            except ZeroDivisionError:
                bs_d_prime = None

            bootstrapped_d_prime_list_by_size[size].append(bs_d_prime)
    std_dev_by_size = calc_std_devs(bootstrapped_d_prime_list_by_size)
    return std_dev_by_size

def calc_std_devs(bootstrapped_d_prime_list_by_size):
    std_dev_by_size = {}
    for size, bs_d_primes in bootstrapped_d_prime_list_by_size.iteritems():
        try:
            mean = math.fsum(bs_d_primes)/len(bs_d_primes)
            variance = (math.fsum([(prime - mean)**2.0 \
                for prime in bs_d_primes]))/(len(bs_d_primes) - 1)
            std_dev = math.sqrt(variance)
            std_dev_by_size[size] = std_dev
        except TypeError: #random samples with no success or failure
        #this except clause happens when sample size is small and/or
        #iterations kwarg is very very large
            std_dev_by_size[size] = None
    return std_dev_by_size

def make_lists_for_resampling(bin_data):
    behavior_lists = {}
    for stim_size in bin_data:
        behavior_lists[stim_size] = []
        for each in ["success", "ignore", "failure"]:
            for i in xrange(bin_data[stim_size][each]):
                behavior_lists[stim_size].append(each)
    return behavior_lists

def get_bin_data_for_each_stim_size(bin):
    bin_data = {}
    #start success, failure, and ignore totals at 0
    for session in bin:
        for stim_size in session["total_trials_by_size"].keys():
            bin_data[stim_size] = {
                "success": 0,
                "failure": 0,
                "ignore": 0
            }
    #now add up behavior events from all sessions in the bin
    for session in bin:
        successes_by_size = \
            session["num_behavior_outcomes_by_size"]["success"]
        failures_by_size = \
            session["num_behavior_outcomes_by_size"]["failure"]
        ignores_by_size = \
            session["num_behavior_outcomes_by_size"]["ignore"]

        for stim_size in session["total_trials_by_size"].keys():
            bin_data[stim_size]["success"] += \
                successes_by_size[stim_size]

            bin_data[stim_size]["failure"] += \
                failures_by_size[stim_size]

            bin_data[stim_size]["ignore"] += \
                ignores_by_size[stim_size]
    return bin_data

def calc_real_d_prime(bin_data):
    real_d_prime_by_size = {}
    for stim_size in bin_data:
        total = float(bin_data[stim_size]["success"] + \
            bin_data[stim_size]["failure"])
        try:
            d_prime = float(bin_data[stim_size]["success"])/total - \
                float(bin_data[stim_size]["failure"])/total
        except ZeroDivisionError:
            d_prime = None
        real_d_prime_by_size[stim_size] = d_prime
    return real_d_prime_by_size

def get_sizes_in_stats_list(list_of_session_stats):
    '''
    Returns a list of stimulus size strings. This list contains sizes present
    in ANY session in the list_of_session_stats, not necessarily ALL sessions.

    :param list_of_session_stats: the list returned by
        get_stats_for_each_session()
    '''
    sizes = []
    for each in list_of_session_stats:
        session_sizes = each["total_trials_by_size"].keys()
        for size in session_sizes:
            if not size in sizes:
                sizes.append(size)
    sizes = sort_by_size_from_size_strings(sizes)
    return sizes

def get_stats_for_each_session(animal_name, sessions):
    '''
    Returns a list of dicts with statistics about each session for an
    animal. e.g.
    all_session_results = [{
        'filename': 'AB1_140617.mwk',
        'session_number': 1,
        'ignores': 2,
        'successes': 2,
        'failures': 0,
        'total_trials': 4,
        'd_prime_overall': 1.0,
        'pct_correct_overall': 50.0,
        'pct_failure_overall': 0.0,
        'pct_ignore_overall': 50.0,
        'd_prime_by_size': {
            '40.0': 1.0,
            '35.0': 0.8,
            etc
        },
        'pct_correct_by_size': {
            '40.0': 50.0,
            etc
        },
        'pct_failure_by_size': {
            '40.0': 0.0,
            etc
        }
        ...
        other keys in result:
        'pct_ignore_by_size',
        'total_trials_by_size'

    },

    #Note the NoneType values in this session

    {
        'filename': 'AB1_140618.mwk',
        'session_number': 2,
        'ignores': 0,
        'successes': 0,
        'failures': 0,
        'total_trials': 0,
        'd_prime_overall': None,
        'pct_correct_overall': None,
        'pct_failure_overall': None,
        'pct_ignore_overall': None
        'd_prime_by_size': key: size value: None,
        'pct_correct_by_size': key: size value: None,
        'pct_failure_by_size': key: size value: None
        ...
        other keys in result:
        'pct_ignore_by_size',
        'total_trials_by_size',
    }]

    NOTE: if there are no trials for the denominator of a key
        (e.g. pct_correct or d_prime), the key's value is set to None.
        Behavior outcomes (e.g. ignores, successes, etc.) with no occurances
        are left with value = 0.
    '''
    #TODO break this down into more functions...it's a bit difficult to read
    print "Starting analysis for " + animal_name
    all_session_results = []
    session_num = 1
    for session in sessions:
        all_trials = get_session_trials(animal_name, session)

        #make dict to store session data
        session_result = {"session_number": session_num,
                          "total_trials": len(all_trials),
                          "filename": session}


        total_trials_by_size = {}
        successes = 0
        failures = 0
        ignores = 0
        #keep track of total successes and failures for each size
        num_failure_by_size = {}
        num_success_by_size = {}
        num_ignores_by_size = {}

        for trial in all_trials:
            #add trial to total trials for each size
            try:
                total_trials_by_size[str(trial["stm_size"])] += 1
            except KeyError:
                total_trials_by_size[str(trial["stm_size"])] = 1

            #track successes and failures for each size, will use for d'
            if trial["behavior_outcome"] == "success":
                successes += 1
                try:
                    num_success_by_size[str(trial["stm_size"])] += 1
                except KeyError:
                    num_success_by_size[str(trial["stm_size"])] = 1

            elif trial["behavior_outcome"] == "failure":
                failures += 1
                try:
                    num_failure_by_size[str(trial["stm_size"])] += 1
                except KeyError:
                    num_failure_by_size[str(trial["stm_size"])] = 1
            elif trial["behavior_outcome"] == "ignore":
                ignores += 1
                try:
                    num_ignores_by_size[str(trial["stm_size"])] += 1
                except KeyError:
                    num_ignores_by_size[str(trial["stm_size"])] = 1
            else:
                #this really shouldnt happen, but just in case...
                print "No behavior_outcome in trial ", trial["trial_num"], \
                    "for animal ", animal_name, " session ", session
                #dont include this trial in total trials
                total_trials_by_size[str(trial["stm_size"])] -= 1

        #done with for loop, now populate data for session_result
        #first add data we already have...
        session_result["successes"] = successes
        session_result["failures"] = failures
        session_result["ignores"] = ignores
        try:
            session_result["d_prime_overall"] = (float(successes)/float(\
                successes + failures)) - (float(failures)/float(\
                successes + failures))
        except ZeroDivisionError:
            session_result["d_prime_overall"] = None
        session_result["total_trials_by_size"] = total_trials_by_size

        #now get ready to add data from by_size dicts...
        d_prime_by_size = {}
        pct_correct_by_size = {}
        pct_failure_by_size = {}
        pct_ignore_by_size = {}

        for stim_size in total_trials_by_size:
            #add any missing size keys with 0 value to make life easier
            num_success_by_size = addMissingKey(num_success_by_size, stim_size)
            num_failure_by_size = addMissingKey(num_failure_by_size, stim_size)
            num_ignores_by_size = addMissingKey(num_ignores_by_size, stim_size)

            try:
                d_prime_by_size[stim_size] = (float(num_success_by_size[\
                    stim_size])/float(num_success_by_size[stim_size] + \
                    num_failure_by_size[stim_size])) - (float(\
                    num_failure_by_size[stim_size])/float(num_success_by_size[\
                    stim_size] + num_failure_by_size[stim_size]))
            except ZeroDivisionError:
                d_prime_by_size[stim_size] = None

            total_trials_for_size = float(num_success_by_size[stim_size] + \
                num_ignores_by_size[stim_size] + num_failure_by_size[stim_size])

            try:
                pct_correct_by_size[stim_size] = (float(num_success_by_size[\
                    stim_size]))/total_trials_for_size
            except ZeroDivisionError:
                pct_correct_by_size[stim_size] = None

            try:
                pct_failure_by_size[stim_size] = (float(num_failure_by_size[\
                    stim_size]))/total_trials_for_size
            except ZeroDivisionError:
                pct_failure_by_size[stim_size] = None

            try:
                pct_ignore_by_size[stim_size] = (float(num_ignores_by_size[\
                    stim_size]))/total_trials_for_size
            except ZeroDivisionError:
                pct_ignore_by_size[stim_size] = None

        #finally, add results to the session's results dict
        session_result["d_prime_by_size"] = d_prime_by_size
        session_result["pct_correct_by_size"] = pct_correct_by_size
        session_result["pct_failure_by_size"] = pct_failure_by_size
        session_result["pct_ignore_by_size"] = pct_ignore_by_size
        session_result["num_behavior_outcomes_by_size"] = {
            "success": num_success_by_size,
            "failure": num_failure_by_size,
            "ignore": num_ignores_by_size
        }

        all_session_results.append(session_result)
        session_num += 1
    return all_session_results

def addMissingKey(size_dict, key):
    '''
    Helper func so you don't have to check whether a key exists before doing
    math with its values. If a key doesn't exist in one of the by_size dicts,
    addMissingKey will add the key with value 0 and return the dict.
    '''
    if not key in size_dict:
        size_dict[key] = 0
        return size_dict
    return size_dict

def get_session_trials(animal_name, session_filename):
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
    # path = 'input/' + animal_name + '/' + session_filename
    path = os.path.join(input_dir, animal_name, session_filename)

    df = pymworks.open_file(path)
    events = df.get_events([
        "Announce_TrialStart",
        "Announce_TrialEnd",
        "success",
        "failure",
        "ignore",
        "stm_size"]
    )

    trials = []
    trial_num = 1
    for index, event in enumerate(events):
        if (event.name == "Announce_TrialStart" and
        event.value == 1):
            trial = {
                "trial_num": trial_num,
                "stm_size": None,
                "behavior_outcome": None
            }

            try:
                if events[index - 1].name == "stm_size":
                    trial["stm_size"] = events[index - 1].value
            except IndexError:
                print "stm_size out of range for session", session_filename, \
                index
            try:
                if events[index + 1].name in ["success", "failure", "ignore"]:
                    trial["behavior_outcome"] = events[index + 1].name
            except IndexError:
                print "beh_outcome out of range for session", session_filename,\
                 index
            if (trial["stm_size"] is not None and
            trial["behavior_outcome"] is not None):
                trials.append(trial)
                trial_num += 1
    return trials

if __name__ == "__main__":
    import sys
    input_dir = sys.argv[1]
    animals_and_sessions = get_animals_and_their_session_filenames(input_dir)
    analyze_sessions(animals_and_sessions, graph_summary_stats=True)