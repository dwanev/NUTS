import numpy as np
import json

from experiment_wrapper import loop_experiment
from figure_utils import create_boxplot_figure_save_to_file


def single_point_repeat_experiment():
    examples_per_class = 20
    reduced_dimensions = 5
    perform_is_a_specific_label_assert = False  # we check if the unlabbeled word is a "specific word". Only useful in testing where ground truth is known.
    perform_general_what_is_assert = False  # this is the form of the question we would need in a real word setting where ground truth is not knowm
    check_all_classes_in_loop_with_isa_question_asserts = False  #
    check_all_unlabelled_instances_islike_a_labled_instance_of_the_same_class_with_asserts = False  #
    check_is_a_target_and_not_is_all_neg_classes = True  # 0.64 at 2 examples, 0.89 at 20 examples  # check the unlabeled instance with one query per class, is it bird?, is it one? is it two? etc

    results = []
    filename_prefix = "single_point_"

    if perform_is_a_specific_label_assert:  # we check if the unlabbeled word is a "specific word". Only useful in testing where ground truth is known.
        filename_prefix =  filename_prefix + "perform_is_a_specific_label_assert"+'_'
    if perform_general_what_is_assert:  # this is the form of the question we would need in a real word setting where ground truth is not knowm
        filename_prefix =  filename_prefix + "perform_general_what_is_assert"+'_'
    if check_all_classes_in_loop_with_isa_question_asserts:  # check the unlabeled instance with one query per class, is it bird?, is it one? is it two? etc
        filename_prefix =  filename_prefix + "check_all_classes_in_loop_with_isa_question_asserts"+'_'
    if check_all_unlabelled_instances_islike_a_labled_instance_of_the_same_class_with_asserts:  #
        filename_prefix =  filename_prefix + "check_all_unlabelled_instances_islike_a_labled_instance_of_the_same_class_with_asserts"+'_'
    if check_is_a_target_and_not_is_all_neg_classes:  # 0.64 second version of paper, submitted to editors, no response yet.
        filename_prefix =  filename_prefix + "check_is_a_target_and_not_is_all_neg_classes"+'_'

    for AIKR_Limit in [100]:
        perf_list, rr = loop_experiment(
            examples_per_class=examples_per_class,
            reduced_dimensions=reduced_dimensions,
            filename_prefix=filename_prefix,
            repeats=50,
            AIKR_Limit = AIKR_Limit,
            check_is_a_target_and_not_is_all_neg_classes=check_is_a_target_and_not_is_all_neg_classes,
            perform_is_a_specific_label_assert=perform_is_a_specific_label_assert,
            perform_general_what_is_assert=perform_general_what_is_assert,
            check_all_classes_in_loop_with_isa_question_asserts=check_all_classes_in_loop_with_isa_question_asserts,
            check_all_unlabelled_instances_islike_a_labled_instance_of_the_same_class_with_asserts=check_all_unlabelled_instances_islike_a_labled_instance_of_the_same_class_with_asserts,
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


