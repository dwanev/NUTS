import numpy as np
import json

from experiment_wrapper import get_results
from figure_utils import create_boxplot_figure_save_to_file

if __name__ == '__main__':
    results = []
    examples_per_class = 3
    filename_prefix = "vary_reduced_dimensions"
    for reduced_dimensions in [2, 3, 4, 5, 10, 20, 40]:
        perf_list, rr = get_results(
            examples_per_class=examples_per_class,
            reduced_dimensions=reduced_dimensions,
            filename_prefix=filename_prefix,
            repeats=50,
            AIKR_Limit = 10,
            check_is_a_target_and_not_is_all_neg_classes=True
        )
        average = sum(perf_list) / len(perf_list)
        print("Dimensions", reduced_dimensions, " performance", average)
        results.append(
            {"examples_per_class": examples_per_class, "reduced_dimensions": reduced_dimensions, "average": average,
             "stdev": np.std(perf_list), "max": np.max(perf_list), "min": np.max(perf_list), "n": len(perf_list),
             "perf_list": perf_list})
        print(perf_list)

        create_boxplot_figure_save_to_file(
            data_as_array_of_arrays = [perf_list],
            label_array=["Dim:"+str(reduced_dimensions)],
            output_filename=filename_prefix +'_dimensions_'+str(reduced_dimensions)+"_examples_"+str(examples_per_class) + ".png"
        )

        with open(filename_prefix + "_results_for_varying_reduced_dimensions.json", "w") as f:
            json.dump(results, f)



