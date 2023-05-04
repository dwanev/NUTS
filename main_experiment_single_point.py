import numpy as np
import json

from experiment_wrapper import loop_experiment
from figure_utils import create_boxplot_figure_save_to_file


def single_point_repeat_experiment():
    results = []
    examples_per_class = 20
    filename_prefix = "single_point_"
    reduced_dimensions = 5
    for AIKR_Limit in [100]:
        perf_list, rr = loop_experiment(
            examples_per_class=examples_per_class,
            reduced_dimensions=reduced_dimensions,
            filename_prefix=filename_prefix,
            repeats=50,
            AIKR_Limit = AIKR_Limit,
            check_is_a_target_and_not_is_all_neg_classes=True
        )
        average = sum(perf_list) / len(perf_list)
        print("AIKR_Limit", AIKR_Limit, " performance", average)
        results.append(
            {"examples_per_class": examples_per_class, "reduced_dimensions": reduced_dimensions, "average": average,
             "stdev": np.std(perf_list), "max": np.max(perf_list), "min": np.max(perf_list), "n": len(perf_list),
             "perf_list": perf_list,
             "AIKR_Limit":AIKR_Limit})
        print(perf_list)

        create_boxplot_figure_save_to_file(
            data_as_array_of_arrays = [perf_list],
            label_array=["AIKR_Limit:"+str(AIKR_Limit)],
            output_filename=filename_prefix +'_dimensions_'+str(reduced_dimensions)+"_examples_"+str(examples_per_class)+"_AIKR_Limit_"+str(AIKR_Limit) + ".png"
        )

        with open(filename_prefix +'_dimensions_'+str(reduced_dimensions)+"_examples_"+str(examples_per_class)+"_AIKR_Limit_"+str(AIKR_Limit)+ ".json", "w") as f:
            json.dump(results, f)




if __name__ == '__main__':
    single_point_repeat_experiment()


