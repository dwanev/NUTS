import time
import glob
import json
import torch
import random


from nar_utils import run_experiment_with_nalifier, ASSERT_PREFIX, ASSERT_TRUE_PREFIX, ASSERT_FALSE_PREFIX
from Nalifier import Nalifier


from generate_nar_file_from_wav import _delete_file, load_file_return_torch_tensor, \
    log_mel_spectrogram, N_MELS, convert_tensor_to_statements, normalise_torch, filename_to_instance_name, \
    create_narsese_isa_question, create_narsese_islike_statement, \
    create_narsese_islike_question, create_narsese_islike_question2


INPUT_DATA_DIMENSION = 8000

unlabeled_time_total = 0
unlabeled_count = 0


def load_files_convert_to_mels(word_population_list, examples_per_class = 3):
    mel_by_instance_name = {}
    label2instance_name = {}
    for target_word in word_population_list:
        fl = glob.glob("/home/dwane/Downloads/speech_commands_v0.02/"+target_word+"/*.wav")
        random.shuffle(fl)
        for i, fn in enumerate(fl):
            instance_name = filename_to_instance_name(fn) # expect this to be unique
            torch_data = load_file_return_torch_tensor(fn)
            if torch_data.shape[0] != 16000:
                p1d = (0, 16000 - torch_data.shape[0])  # pad last dim all on the end
                torch_data = torch.nn.functional.pad(torch_data, p1d, "constant", 0)  # effectively zero padding

            mel = log_mel_spectrogram(torch_data, n_mels=N_MELS)
            if mel.shape[1] * mel.shape[0] == INPUT_DATA_DIMENSION:
                mel_by_instance_name[instance_name] = {"base_data": mel,
                                                       "label": target_word}
                if target_word not in label2instance_name:
                    label2instance_name[target_word] = []
                label2instance_name[target_word].append(instance_name)
                if len(label2instance_name[target_word]) >= examples_per_class:
                    break
            else:
                pass
    return mel_by_instance_name, label2instance_name


def run_experiment_with_reduced_dim(AIKR_Limit = 10,
                                    reduced_dimensions = 100,
                                    examples_per_class = 3,
                                    word_population_list = [ 'one', 'two' ],
                                    unknown_word = 'one', # unlabbeled class
                                    print_nars=False, #  logging level nars
                                    random_dim_reducing_matrix=None, # uses the matrix passed in, if one is passed in.
                                    out_nal_filename = 'experiment_c6.nal', # store narsese generated in this file for debugging
                                    debug_print=True, #  logging level ours
                                    perform_is_a_specific_label_assert=False, # we check if the unlabbeled word is a "specific word". Only useful in testing where ground truth is known.
                                    perform_general_what_is_assert=False, # this is the form of the question we would need in a real word setting where ground truth is not knowm
                                    check_all_classes_in_loop_with_isa_question_asserts = False, # check the unlabeled instance with one query per class, is it bird?, is it one? is it two? etc
                                    check_all_unlabelled_instances_islike_a_labled_instance_of_the_same_class_with_asserts = False, # TODO get the performance of this
                                    check_is_a_target_and_not_is_all_neg_classes = False, # 0.64 second version of paper, submitted to editors, no response yet.
                                    confusion_matrix = {}
                                    ):
    """
        1) using the same matrix passed in (otherwise generate one)
        2) , run through each word as the unknown word,
        3) load examples_per_class WAV files per class, convert into MELs
        4) Generate 'training/initialization' Narsese statements
        5) Generate 'inference' Narsese statements
        5) Pass statements to run_experiment_with_nalifier, which passes them to nalifier, and pas output to ONA
        6) Check asserts

    :param AIKR_Limit:
    :param reduced_dimensions:
    :param examples_per_class:
    :param word_population_list:
    :param unknown_word:
    :param print_nars:
    :param random_dim_reducing_matrix:
    :param out_filename:
    :param debug_print:
    :param perform_is_a_specific_label_assert:
    :param perform_general_what_is_assert:
    :param check_all_classes_in_loop_with_isa_question_asserts:
    :param check_all_unlabelled_instances_islike_a_labled_instance_of_the_same_class_with_asserts:
    :param check_is_a_target_and_not_is_all_neg_classes:
    :return:
    """

    assert perform_is_a_specific_label_assert\
           +perform_general_what_is_assert+check_all_classes_in_loop_with_isa_question_asserts\
           +check_all_unlabelled_instances_islike_a_labled_instance_of_the_same_class_with_asserts\
           +check_is_a_target_and_not_is_all_neg_classes == 1, "This routine only supports one of the assert types to be true."
    # generate a random projection
    if random_dim_reducing_matrix is None:
        random_dim_reducing_matrix = torch.rand(INPUT_DATA_DIMENSION, reduced_dimensions)

    _delete_file(out_nal_filename)
    all_instance_names = []

    word_population_list_minus_unlabbeled = word_population_list.copy()
    word_population_list_minus_unlabbeled.remove(unknown_word)

    t0 = time.time()
    mel_by_instance_name, label2instance_name = load_files_convert_to_mels(word_population_list_minus_unlabbeled, examples_per_class=examples_per_class)
    t1 = time.time()
    average_mel_load_time = (t1-t0)/ len(mel_by_instance_name)

    # add the target word (we have an extra instance for it)
    mel_by_instance_name_target_word, label2instance_name_target_word = load_files_convert_to_mels([unknown_word], examples_per_class=examples_per_class+1)
    for k in mel_by_instance_name_target_word.keys():
        mel_by_instance_name[k] = mel_by_instance_name_target_word[k]

    for k in label2instance_name_target_word.keys():
        label2instance_name[k] = label2instance_name_target_word[k]


    # describe each instance in narsese
    statement_list = []
    unlabbeled_instance_list = []
    unlabbeled_instance = random.choice(label2instance_name[unknown_word])
    unlabbeled_instance_list.append(unlabbeled_instance)
    all_instance_names_which_are_same_as_unlabbeled = []

    global unlabeled_time_total
    global unlabeled_count

    # statement_list.append("*volume=0")
    for target_word in label2instance_name.keys():
        for instance_name in label2instance_name[target_word]:
            t0 = time.time()
            add_isa_statement = instance_name != unlabbeled_instance # note: this is a specific instance not a class.
            mel = mel_by_instance_name[instance_name]["base_data"]
            reduced_dim = torch.matmul(mel.reshape([-1]), random_dim_reducing_matrix)
            reduced_dim_norm = normalise_torch(reduced_dim)  # between 0 and 1.0
            _, sub_statement_list = convert_tensor_to_statements(instance_name, reduced_dim_norm, truth_threshold=0.5, add_isa_statement=add_isa_statement,
                                         label=mel_by_instance_name[instance_name]["label"])
            all_instance_names.append(instance_name)
            if target_word == unknown_word and unlabbeled_instance != instance_name:
                all_instance_names_which_are_same_as_unlabbeled.append(instance_name)

            statement_list.extend(sub_statement_list)
            t1 = time.time()
            if not add_isa_statement:
                unlabeled_time_total += (t1-t0) # add the time to generate the Narsese for the unknown class
                unlabeled_count += 1

    # Generate IS_A queries for the unlabeled examples
    inference_statement_list = []

    # generate additional asserts for after inference
    inference_addendum_statement_list = []

    # if we can know the unlabelled instance word in advance, we could use this approach. Used in paper.
    if perform_is_a_specific_label_assert:
        for unlabbeled_instance in unlabbeled_instance_list:
            expected_label = mel_by_instance_name[unlabbeled_instance]['label']
            what_is_question = create_narsese_isa_question(unlabbeled_instance, expected_label)
            inference_statement_list.append(what_is_question)
            # create an asserts
            inference_statement_list.append(ASSERT_TRUE_PREFIX)

    # if we DO NOT know the unlabelled instance word in advance, this is the most realistic real word scenario.
    if perform_general_what_is_assert: # for this we get terrible results (0%) hmmmm
        for unlabbeled_instance in unlabbeled_instance_list:
            expected_label = mel_by_instance_name[unlabbeled_instance]['label']
            is_like_what_question = create_narsese_islike_question(unlabbeled_instance, tense="")
            inference_statement_list.append(is_like_what_question)
            # create an asserts
            inference_statement_list.append(ASSERT_PREFIX + json.dumps([expected_label]))

    if check_all_classes_in_loop_with_isa_question_asserts:
        for unlabbeled_instance in unlabbeled_instance_list:
                mel = mel_by_instance_name[unlabbeled_instance]["base_data"]
                if mel.shape[1] * mel.shape[0] == INPUT_DATA_DIMENSION:
                    reduced_dim = torch.matmul(mel.reshape([-1]), random_dim_reducing_matrix)
                    reduced_dim_norm = normalise_torch(reduced_dim)  # between 0 and 1.0
                    _, sub_statement_list = convert_tensor_to_statements(unlabbeled_instance, reduced_dim_norm, truth_threshold=0.5,
                                                                      add_isa_statement=False,
                                                                      label="ERROR this value should not be used.")
                    inference_statement_list.extend(sub_statement_list)
                    # Query what instance we have
                    # statement = create_narsese_isa_question(unlabbeled_instance, is_a_what=mel_by_instance_name[unlabbeled_instance]["label"])
                    # statement_list.append(statement)

                    for word in word_population_list:
                        unlabbeled_isa_question = create_narsese_isa_question(unlabbeled_instance, is_a_what=word)
                        inference_statement_list.append(unlabbeled_isa_question)
                        # create an assert
                        if word == unknown_word:
                            inference_statement_list.append(ASSERT_TRUE_PREFIX)
                        else:
                            inference_statement_list.append(ASSERT_FALSE_PREFIX)
                else:
                    print("Instance", instance_name, " has the wrong input shape. skipping. size:", mel.shape[1] * mel.shape[0] )
                    print("In valid unseen instance")
                    return None

    if check_all_unlabelled_instances_islike_a_labled_instance_of_the_same_class_with_asserts:
        for unlabbeled_instance in unlabbeled_instance_list:
                mel = mel_by_instance_name[unlabbeled_instance]["base_data"]
                if mel.shape[1] * mel.shape[0] == INPUT_DATA_DIMENSION:
                    reduced_dim = torch.matmul(mel.reshape([-1]), random_dim_reducing_matrix)
                    reduced_dim_norm = normalise_torch(reduced_dim)  # between 0 and 1.0
                    _, sub_statement_list = convert_tensor_to_statements(unlabbeled_instance, reduced_dim_norm, truth_threshold=0.5,
                                                                      add_isa_statement=False,
                                                                      label="ERROR this label value should not be used, so is intensionally set incorrectly.")
                    inference_statement_list.extend(sub_statement_list)
                    # Query what instance we have
                    # statement = create_narsese_isa_question(unlabbeled_instance, is_a_what=mel_by_instance_name[unlabbeled_instance]["label"])
                    # statement_list.append(statement)

                    what_islike_question = create_narsese_islike_question(unlabbeled_instance)
                    inference_statement_list.append(what_islike_question)

                    # create an asserts
                    valid_responses = []
                    for i1 in all_instance_names_which_are_same_as_unlabbeled: # these are valid answers in "what is the unlabeld instance like"
                        valid_responses.append(create_narsese_islike_statement(unlabbeled_instance, i1, tv=None)[:-1])
                    inference_statement_list.append(ASSERT_PREFIX + json.dumps(valid_responses))  # strip the '.' from the statement

                else:
                    print("Instance", instance_name, " has the wrong input shape. skipping. size:", mel.shape[1] * mel.shape[0] )
                    print("In valid unseen instance")
                    return None

    if check_is_a_target_and_not_is_all_neg_classes:
        for unlabbeled_instance in unlabbeled_instance_list:
            mel = mel_by_instance_name[unlabbeled_instance]["base_data"]
            if mel.shape[1] * mel.shape[0] == INPUT_DATA_DIMENSION:
                reduced_dim = torch.matmul(mel.reshape([-1]), random_dim_reducing_matrix)
                reduced_dim_norm = normalise_torch(reduced_dim)  # between 0 and 1.0
                _, sub_statement_list = convert_tensor_to_statements(unlabbeled_instance, reduced_dim_norm,
                                                                     truth_threshold=0.5,
                                                                     add_isa_statement=False,
                                                                     label="ERROR this label value should not be used, so is intentionally set incorrectly.")
                inference_statement_list.extend(sub_statement_list)
                isa_question = create_narsese_isa_question(unlabbeled_instance, unknown_word)
                inference_statement_list.append(isa_question)
                # create an asserts
                inference_statement_list.append(ASSERT_TRUE_PREFIX)

                for c in word_population_list:
                    if c != unknown_word:
                        isa_question = create_narsese_isa_question(unlabbeled_instance, c)
                        inference_addendum_statement_list.append(isa_question)
                        # create an asserts
                        inference_addendum_statement_list.append(ASSERT_FALSE_PREFIX)

            else:
                print("Instance", instance_name, " has the wrong input shape. skipping. size:",
                      mel.shape[1] * mel.shape[0])
                print("In valid unseen instance")
                return None





    if out_nal_filename is not None:
        with open(out_nal_filename, 'a') as f:
            if debug_print:
                print("Saving statement_list to ", out_nal_filename)
            for statement in statement_list:
                f.write(statement + '\n')
            for statement in inference_statement_list:
                f.write(statement + '\n')

    if debug_print:
        print("____________________________________")
        print(" Initialize NARS with ", len(statement_list), "statements")
        print("____________________________________")
    # 'Train' the model: show it labeled data
    nalifier = Nalifier(AIKR_Limit = AIKR_Limit)
    run_experiment_with_nalifier(nalifier, statement_list, reduced_dimensions, print_nars=print_nars, debug_print=debug_print)

    if debug_print:
        print("____________________________________")
        print(" Now perform Inference with ", len(inference_statement_list), "statements")
        print("____________________________________")
    t0 = time.time()
    # 'Infer' with the model: show it unlabeled data
    result, assert_ok_count, assert_bad_count, max_tv_correct_label, label_1 = run_experiment_with_nalifier(nalifier, inference_statement_list, reduced_dimensions, print_nars=print_nars, reset_nars=False, debug_print=debug_print)
    t1 = time.time()
    unlabeled_time_total += (t1 - t0) # add the time to perform inference for the unknown class
    inference_average_time = average_mel_load_time + (unlabeled_time_total/unlabeled_count)
    if debug_print:
        print("Unknown Word", unknown_word, "Inference Time", "Average per instance", inference_average_time, "over ", unlabeled_count, "inferences")


    if len(inference_addendum_statement_list) > 0:
        if debug_print:
            print("____________________________________")
            print(" Now perform Inference Addendum with ", len(inference_addendum_statement_list), "statements")
            print("____________________________________")
        t0 = time.time()
        # 'Extra Infer' with the model: show it unlabeled data
        _extra_result, assert_ok_count, assert_bad_count, max_tv_addendum_label, label_2 = run_experiment_with_nalifier(nalifier, inference_addendum_statement_list, reduced_dimensions, print_nars=print_nars, reset_nars=False, debug_print=debug_print)
        t1 = time.time()
        print("____________________________________")

        if max_tv_correct_label < max_tv_addendum_label:
            result = False
            max_prob_label = label_2
            print("Incorrect label has higher truth value, marking failed tv:",max_tv_correct_label," addendum tv:",max_tv_addendum_label )
        else:
            max_prob_label = label_1

        print("Result", result, "Addendum Result",  _extra_result, assert_ok_count, assert_bad_count)

        if result:
            assert unknown_word == max_prob_label

            if unknown_word not in confusion_matrix:
                confusion_matrix[unknown_word] = {}
            if max_prob_label not in confusion_matrix[unknown_word]:
                confusion_matrix[unknown_word][max_prob_label] = 0
            confusion_matrix[unknown_word][max_prob_label] += 1

    return result
