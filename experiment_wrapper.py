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



def get_time():
    REDUCED_DIMENSIONS = 4
    AIKR_Limit = 10
    results = []
    word_population_list = ['bed', 'cat', 'down', 'five', 'forward', 'go', 'house', 'left', 'marvin', 'no', 'on', 'seven', 'six', 'tree', 'up', 'visual', 'yes', 'backward', 'bird', 'dog', 'eight', 'follow', 'four', 'happy', 'learn', 'nine', 'off', 'one', 'right', 'sheila', 'stop', 'three', 'two', 'wow', 'zero' ]
    random_dim_reducing_matrix = torch.rand(INPUT_DATA_DIMENSION, REDUCED_DIMENSIONS)
    count = 0
    count_ok = 0
    t0 = time.time()
    for unknown_word in word_population_list:
        ok = run_experiment_with_reduced_dim(AIKR_Limit=AIKR_Limit,
                                             REDUCED_DIMENSIONS=REDUCED_DIMENSIONS,
                                             word_population_list = word_population_list,
                                             unknown_word=unknown_word,
                                             quiet=True,
                                             random_dim_reducing_matrix=random_dim_reducing_matrix,
                                             out_filename = 'experiment_c8_get_time.nal')
        results.append( {"ok": ok, "AIKR_Limit": AIKR_Limit, "REDUCED_DIMENSIONS": REDUCED_DIMENSIONS})
        count += 1
        if ok:
            count_ok += 1
    perf = count_ok/count
    print( "overall perf:", count_ok, '/', count, perf )

    t1 = time.time()
    print("diff", (t1-t0) )
    print("average", (t1-t0)/len(word_population_list) , "(includes training time as well)")




def find_perf_from_random_matrix(
        examples_per_class = 3, # includes the unlabeled / unknown example
        reduced_dimensions = 4,
        out_nal_filename='find_perf_from_random_matrix.nal'
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
    AIKR_Limit = 10
    results = []
    perf_result_list = []
    word_population_list = ['bed', 'cat', 'down', 'five', 'forward', 'go', 'house', 'left', 'marvin', 'no', 'on', 'seven', 'six', 'tree', 'up', 'visual', 'yes', 'backward', 'bird', 'dog', 'eight', 'follow', 'four', 'happy', 'learn', 'nine', 'off', 'one', 'right', 'sheila', 'stop', 'three', 'two', 'wow', 'zero' ]
    random_dim_reducing_matrix = torch.rand(INPUT_DATA_DIMENSION, reduced_dimensions)
    count = 0
    count_ok = 0
    for unknown_word in word_population_list:
        ok = run_experiment_with_reduced_dim(
            AIKR_Limit=AIKR_Limit,
            reduced_dimensions=reduced_dimensions,
            word_population_list = word_population_list,
            unknown_word=unknown_word,
            print_nars=True,
            random_dim_reducing_matrix=random_dim_reducing_matrix,
            out_filename = out_nal_filename,
            perform_is_a_specific_label_assert=True
        )
        results.append( {"ok": ok, "AIKR_Limit": AIKR_Limit, "reduced_dimensions": reduced_dimensions})
        count += 1
        if ok:
            count_ok += 1
        perf = count_ok/count
        print( "interim perf:", count_ok, '/', count, perf )

    perf_result_list.append({"perf":perf,
                             "examples_per_class": examples_per_class,
                             "AIKR_Limit": AIKR_Limit,
                             "reduced_dimensions":reduced_dimensions,
                             "unlabeled_word": unknown_word,
                             "matrix": random_dim_reducing_matrix.numpy().tolist()

                             })
    with open("tmp_matrix.json", "w") as f:
        json.dump(perf_result_list, f)

    return perf, random_dim_reducing_matrix


def get_results(examples_per_class:int, reduced_dimensions:int, filename_prefix:str = "raw_", repeats:int=50):
    best_perf = 0
    perf_list = []
    for i in range(repeats):
        perf, matrix = find_perf_from_random_matrix(examples_per_class=examples_per_class,
                                                    reduced_dimensions=reduced_dimensions,
                                                    out_nal_filename=filename_prefix + '.nal')
        if perf > best_perf:
            print("Best Perf", best_perf)
            best_perf = perf
            with open(filename_prefix + "_best_matrix_for_num_examples_" + str(
                    examples_per_class) + "_reduced_dimensions_" + str(reduced_dimensions) + ".json", "w") as f:
                json.dump(matrix.numpy().tolist(), f)
        perf_list.append(perf)
    print("--------------------")
    print(filename_prefix, " Results:", perf_list)
    print(filename_prefix, " Average:", sum(perf_list) / len(perf_list))
    return perf_list



def get_results_vary_examples_per_class(reduced_dimensions = 4, filename_prefix = 'examples_per_class'):
    results = []
    labels = []
    array_of_arrays = []
    for examples_per_class in [2,3,4,5,10,20,40,80,160,320,640,1280]:
        perf_list = get_results(examples_per_class, reduced_dimensions, filename_prefix = filename_prefix, repeats=50)
        average = sum(perf_list) / len(perf_list)
        print("Vary Examples per class running average, ", average)
        results.append({"examples_per_class":examples_per_class, "reduced_dimensions":reduced_dimensions, "average":average, "stdev":np.std(perf_list),"max":np.max(perf_list), "min":np.max(perf_list),"n":len(perf_list), "perf_list":perf_list  })
        print(perf_list)
        with open(filename_prefix+"_results_for_reduced_dimensions_"+str(reduced_dimensions)+".json","w") as f:
            json.dump(results,f)

        array_of_arrays.append(perf_list)
        labels.append("Examples="+str(examples_per_class))

        create_boxplot_figure_save_to_file(
            data_as_array_of_arrays = array_of_arrays,
            label_array=labels,
            output_filename=filename_prefix +'_dimensions_'+str(reduced_dimensions)+"_examples_"+str(examples_per_class) + ".png"
        )

    return results, array_of_arrays, labels





