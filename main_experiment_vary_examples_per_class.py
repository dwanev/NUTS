import json
import numpy as np

from experiment_wrapper import loop_experiment, create_boxplot_figure_save_to_file


def vary_examples_per_class(reduced_dimensions = 4, filename_prefix = 'examples_per_class'):
    results = []
    labels = []
    array_of_arrays = []
    AIKR_Limit = 10
    for examples_per_class in [1,2,3,4,5,10,20,40,80,160,320,640,1280]:
        perf_list, rr = loop_experiment(
            examples_per_class = examples_per_class,
            reduced_dimensions = reduced_dimensions,
            filename_prefix = filename_prefix,
            repeats=50,
            AIKR_Limit = AIKR_Limit,
            check_is_a_target_and_not_is_all_neg_classes=True)
        average = sum(perf_list) / len(perf_list)
        print("Vary Examples per class running average, ", average)
        results.append({"examples_per_class":examples_per_class, "reduced_dimensions":reduced_dimensions, "average":average, "stdev":np.std(perf_list),"max":np.max(perf_list), "min":np.max(perf_list),"n":len(perf_list), "perf_list":perf_list  })
        print(perf_list)
        with open(filename_prefix+"_dimensions_"+str(reduced_dimensions)+"_AIKR_"+".json","w") as f:
            json.dump(results,f)

        array_of_arrays.append(perf_list)
        labels.append("Examples="+str(examples_per_class))

        create_boxplot_figure_save_to_file(
            data_as_array_of_arrays = array_of_arrays,
            label_array=labels,
            output_filename=filename_prefix +'_dimensions_'+str(reduced_dimensions)+"_AIKR_Limit_"+str(AIKR_Limit) + ".png"
        )

    return results, array_of_arrays, labels



def graph_results():
    filename_prefix = "vary_examples_per_class"
    with open(filename_prefix + "_dimensions_4_AIKR_Limit_10.json", "r") as f:
        results = json.load(f)

    data_as_array_of_arrays = []
    label_array = []

    for r in results:
        data_as_array_of_arrays.append(r["perf_list"])
        # AIKR_Limit = r["AIKR_Limit"]
        reduced_dimensions = r["reduced_dimensions"]
        examples_per_class = r["examples_per_class"]
        label_array.append(str(examples_per_class))
    create_boxplot_figure_save_to_file(
        data_as_array_of_arrays = data_as_array_of_arrays,
        label_array=label_array,
        output_filename=filename_prefix +'_dimensions_4_AIKR_Limit_10'+ ".png"
    )

if __name__ == '__main__':
    # vary_examples_per_class(reduced_dimensions=4, filename_prefix='vary_examples_per_class_')
    graph_results()


