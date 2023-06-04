import time
import json
import torch

from experiment_runner import run_experiment_with_reduced_dim, INPUT_DATA_DIMENSION

from figure_utils import create_boxplot_figure_save_to_file

import numpy as np

"""
 In the style of:

<{8536c82dnohash0} --> [prop0]>. %0.69%
<{8536c82dnohash0} --> [prop1]>. %0.69%
<{8536c82dnohash0} --> [NOTprop2]>. %0.60%
...
<{8536c82dnohash0} --> one>. %0.90%


Do we state

<{8536c82dnohash0} <-> {another_instance}>. %0.90%  ????


"""


# def get_time():
#     REDUCED_DIMENSIONS = 4
#     AIKR_Limit = 10
#     results = []
#     word_population_list = ['bed', 'cat', 'down', 'five', 'forward', 'go', 'house', 'left', 'marvin', 'no', 'on',
#                             'seven', 'six', 'tree', 'up', 'visual', 'yes', 'backward', 'bird', 'dog', 'eight', 'follow',
#                             'four', 'happy', 'learn', 'nine', 'off', 'one', 'right', 'sheila', 'stop', 'three', 'two',
#                             'wow', 'zero']
#     random_dim_reducing_matrix = torch.rand(INPUT_DATA_DIMENSION, REDUCED_DIMENSIONS)
#     count = 0
#     count_ok = 0
#     t0 = time.time()
#     for unknown_word in word_population_list:
#         ok = run_experiment_with_reduced_dim(AIKR_Limit=AIKR_Limit,
#                                              REDUCED_DIMENSIONS=REDUCED_DIMENSIONS,
#                                              word_population_list=word_population_list,
#                                              unknown_word=unknown_word,
#                                              quiet=True,
#                                              random_dim_reducing_matrix=random_dim_reducing_matrix,
#                                              out_filename='experiment_c8_get_time.nal')
#         results.append({"ok": ok, "AIKR_Limit": AIKR_Limit, "REDUCED_DIMENSIONS": REDUCED_DIMENSIONS})
#         count += 1
#         if ok:
#             count_ok += 1
#     perf = count_ok / count
#     print("overall perf:", count_ok, '/', count, perf)
#
#     t1 = time.time()
#     print("diff", (t1 - t0))
#     print("average", (t1 - t0) / len(word_population_list), "(includes training time as well)")


def loop_through_words(
        # examples_per_class=3,  # includes the unlabeled / unknown example
        # reduced_dimensions=4,
        # out_nal_filename='find_perf_from_random_matrix.nal',
        # AIKR_Limit=10
        **kwargs
):
    """
    1) Generate a random matrix
    2) run an experiment
    3) cont proportion that are successful

    :param examples_per_class:
    :param REDUCED_DIMENSIONS:
    :param out_nal_filename:
    :return:
    """

    results = []
    perf_result_list = []
    word_population_list = ['bed', 'cat', 'down', 'five', 'forward', 'go', 'house', 'left', 'marvin', 'no', 'on',
                            'seven', 'six', 'tree', 'up', 'visual', 'yes', 'backward', 'bird', 'dog', 'eight', 'follow',
                            'four', 'happy', 'learn', 'nine', 'off', 'one', 'right', 'sheila', 'stop', 'three', 'two',
                            'wow', 'zero']

    reduced_dimensions = kwargs['reduced_dimensions']
    AIKR_Limit = kwargs["AIKR_Limit"]
    examples_per_class = kwargs['examples_per_class']

    random_dim_reducing_matrix = torch.rand(INPUT_DATA_DIMENSION, reduced_dimensions)
    count = 0
    count_ok = 0
    for unknown_word in word_population_list:
        ok = run_experiment_with_reduced_dim(
            word_population_list=word_population_list,
            unknown_word=unknown_word,
            # print_nars=True,
            random_dim_reducing_matrix=random_dim_reducing_matrix,
            # out_filename=out_nal_filename,
            # perform_is_a_specific_label_assert=True
            **kwargs
        )
        results.append({"ok": ok, "AIKR_Limit": AIKR_Limit, "reduced_dimensions": reduced_dimensions,
                        "examples_per_class": examples_per_class, "unlabeled_word": unknown_word})
        count += 1
        if ok:
            count_ok += 1
        perf = count_ok / count
        print("interim perf:", count_ok, '/', count, perf)

    perf_result_list.append({"perf": perf,
                             "examples_per_class": examples_per_class,
                             "AIKR_Limit": AIKR_Limit,
                             "reduced_dimensions": reduced_dimensions,
                             "unlabeled_word": unknown_word,
                             "matrix": random_dim_reducing_matrix.numpy().tolist()

                             })
    with open("old_partial_run/tmp_matrix.json", "w") as f:
        json.dump(perf_result_list, f)

    return perf, random_dim_reducing_matrix, results


def loop_experiment(**kwargs):
    """

    Repeat the experiment a number of times

    :param kwargs:
    :return:
    """
    best_perf = 0
    perf_list = []
    global_raw_results = []
    repeats = kwargs['repeats']
    examples_per_class = kwargs['examples_per_class']
    reduced_dimensions = kwargs['reduced_dimensions']
    filename_prefix = kwargs['filename_prefix']
    AIKR_Limit = kwargs['AIKR_Limit']

    del kwargs['filename_prefix']
    del kwargs['repeats']
    # AIKR_Limit = kwargs['AIKR_Limit']

    for i in range(repeats):
        perf, matrix, raw_results = loop_through_words(
            **kwargs)
        global_raw_results.extend(raw_results)
        if perf > best_perf:
            print("Best Perf", best_perf)
            best_perf = perf
            with open(filename_prefix + "_best_matrix_for_num_examples_" + str(
                    examples_per_class) + "_reduced_dimensions_" + str(reduced_dimensions) + "_AIKR_Limit_"+str(AIKR_Limit) + ".json", "w") as f:
                json.dump({"matrix":matrix.numpy().tolist(),"perf":best_perf}, f)
        perf_list.append(perf)
    print("--------------------")
    print(filename_prefix, " Results:", perf_list)
    print(filename_prefix, " Average:", sum(perf_list) / len(perf_list))
    return perf_list, global_raw_results









