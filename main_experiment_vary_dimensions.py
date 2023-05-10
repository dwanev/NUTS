import numpy as np
import json

from experiment_wrapper import loop_experiment
from figure_utils import create_boxplot_figure_save_to_file




def graph_results():
    filename_prefix = "vary_reduced_dimensions"
    with open(filename_prefix + "_results_for_varying_reduced_dimensions.json", "r") as f:
        results = json.load(f)

    data_as_array_of_arrays = []
    label_array = []

    new_data = {}

    for r in results:
        new_data[r["reduced_dimensions"]] = r["perf_list"]
        examples_per_class = r["examples_per_class"]

    sorted_keys = list(new_data.keys())
    sorted_keys.sort()

    for d in sorted_keys:
        data_as_array_of_arrays.append(new_data[d])
        label_array.append(str(d))

    create_boxplot_figure_save_to_file(
        data_as_array_of_arrays = data_as_array_of_arrays,
        label_array=label_array,
        output_filename=filename_prefix +'_dimensions_various'+"_examples_"+str(examples_per_class)+"_" + ".png"
    )


def vary_dimensions():
    results = []
    examples_per_class = 3
    AIKR_Limit = 10
    filename_prefix = "vary_reduced_dimensions"
    for reduced_dimensions in [#2, 3, 4, 5,
                               6, 7, 8, 9, 10]:
        perf_list, rr = loop_experiment(
            examples_per_class=examples_per_class,
            reduced_dimensions=reduced_dimensions,
            filename_prefix=filename_prefix,
            repeats=50,
            AIKR_Limit = AIKR_Limit,
            check_is_a_target_and_not_is_all_neg_classes=True
        )
        average = sum(perf_list) / len(perf_list)
        print("Dimensions", reduced_dimensions, " performance", average)
        results.append(
            {"examples_per_class": examples_per_class, "reduced_dimensions": reduced_dimensions, "average": average,
             "AIKR_Limit":AIKR_Limit,
             "stdev": np.std(perf_list), "max": np.max(perf_list), "min": np.max(perf_list), "n": len(perf_list),
             "perf_list": perf_list})
        print(perf_list)

        create_boxplot_figure_save_to_file(
            data_as_array_of_arrays = [perf_list],
            label_array=["Dim:"+str(reduced_dimensions)],
            output_filename=filename_prefix +'_dimensions_various'+"_examples_"+str(examples_per_class) + ".png"
        )

        with open(filename_prefix + "_results_for_varying_reduced_dimensions.json", "w") as f:
            json.dump(results, f)



def join_results_files():
    filename_prefix = "vary_reduced_dimensions"
    with open(filename_prefix + "_results_for_varying_reduced_dimensions.json", "r") as f:
        results_2 = json.load(f)

    with open(filename_prefix + "_results_for_varying_reduced_dimensions_1_to_5.json", "r") as f:
        results_1 = json.load(f)

    results_all_1_to_7 = []
    results_all_1_to_7.extend(results_1)
    results_all_1_to_7.extend(results_2)

    with open(filename_prefix + "_results_for_varying_reduced_dimensions_1_to_7.json", "w") as f:
        json.dump(results_all_1_to_7, f)
    # manually rename 1_to_7 to ....




if __name__ == '__main__':
    # join_results_files()
    # vary_dimensions()
    graph_results()
