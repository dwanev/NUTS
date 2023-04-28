import torch
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from experiment_runner import run_experiment_with_reduced_dim, INPUT_DATA_DIMENSION

def build_data():
    results = []
    examples_per_class = 20
    reduced_dimensions = 5
    AIKR_Limit = 100
    num_repeats = 100
    filename_prefix = "confusion_matrix_"
    perf_result_list = []
    word_population_list = ['bed', 'cat', 'down', 'five', 'forward', 'go', 'house', 'left', 'marvin', 'no', 'on', 'seven', 'six', 'tree', 'up', 'visual', 'yes', 'backward', 'bird', 'dog', 'eight', 'follow', 'four', 'happy', 'learn', 'nine', 'off', 'one', 'right', 'sheila', 'stop', 'three', 'two', 'wow', 'zero' ]
    random_dim_reducing_matrix = torch.rand(INPUT_DATA_DIMENSION, reduced_dimensions)
    count = 0
    count_ok = 0
    confusion_matrix = {}
    for i in range(num_repeats):
        for unknown_word in word_population_list:
            ok = run_experiment_with_reduced_dim(
                AIKR_Limit=AIKR_Limit,
                reduced_dimensions=reduced_dimensions,
                word_population_list = word_population_list,
                unknown_word=unknown_word,
                print_nars=True,
                random_dim_reducing_matrix=random_dim_reducing_matrix,
                out_nal_filename = filename_prefix+"_tmp.nal",
                perform_is_a_specific_label_assert=False,
                check_all_unlabelled_instances_islike_a_labled_instance_of_the_same_class_with_asserts=False,
                check_is_a_target_and_not_is_all_neg_classes=True,
                confusion_matrix=confusion_matrix
            )
            results.append( {"ok": ok, "AIKR_Limit": AIKR_Limit, "reduced_dimensions": reduced_dimensions, "examples_per_class":examples_per_class, "unlabeled_word":unknown_word})

        print("Parial Perf:", len([x['ok'] for x in results if x['ok'] == True])/len(results))

    with open(filename_prefix + "_raw_results.json", "w") as f:
        json.dump(results, f)

    with open(filename_prefix + "_counts.json", "w") as f:
        json.dump(confusion_matrix, f)


    pd.DataFrame.from_dict(results)


def graph_results():
    filename_prefix = "confusion_matrix_"
    with open(filename_prefix + "_counts.json", "r") as f:
        results = json.load(f)

    word_list = []
    for w in results.keys():
        word_list.extend(results[w].keys())
    word_list = list(set(word_list))



    word_list.sort()

    confusion_matrix = np.zeros( (len(word_list), len(word_list)) )

    for r, ground_truth in enumerate(word_list):
        total = 0.0
        if ground_truth in results:
            total = sum(results[ground_truth].values())
        for c, prediction in enumerate(word_list):
            v = 0
            if ground_truth in results:
                if prediction in results[ground_truth]:
                    v = results[ground_truth][prediction]
            if total > 0.0:
                p = v/total
            else:
                p = 0.0

            confusion_matrix[r,c] = p

    from matplotlib.colors import LinearSegmentedColormap
    c = ["white", "palegreen", "green", "darkgreen"]
    v = [0, .01, .4,   1.]
    l = list(zip(v, c))
    cmap = LinearSegmentedColormap.from_list('rg', l, N=256)


    svm = sns.heatmap(confusion_matrix, annot=False, cmap=cmap, linecolor='white', linewidths=1)
    figure = svm.get_figure()
    figure.savefig(filename_prefix+'.png', dpi=300)

    plt.clf()

    sns.heatmap(data=confusion_matrix, xticklabels=word_list, yticklabels=word_list, cmap=cmap) # cmap=sns.color_palette("Blues",12)
    sns.set(rc={'figure.figsize': (11.7, 8.27), "font.size": 2, "axes.titlesize": 2, "axes.labelsize": 2},
            style="white")
    # sns.set_context("paper", rc={"font.size":2, "axes.titlesize": 2, "axes.labelsize": 2})
    # svm.set_xlabel('REDUCED DIMENSIONS', fontsize=10)
    # svm.set_ylabel('AIKR Limit', fontsize=10)

    plt.savefig(filename_prefix+'_b.png', dpi=1200)



if __name__ == '__main__':
    # build_data()
    graph_results()
