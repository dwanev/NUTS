"""

Set of utils to parse narsese output, or format into narsese,
or run an experiment end to end.


"""

import json
import nar_wrapper
import time


ASSERT_PREFIX = "# Assert Last Output Was:"  # append a list of valid results in the form of a list, and encode with json to produce str
ASSERT_TRUE_PREFIX = "# Assert Last Output Was True"
ASSERT_FALSE_PREFIX = "# Assert Last Output Was False"


def cascade_join(list_of_elements, op="&", property=True):
    """
    Takes a list of items, and combins them into an recursive and statement
    i.e.
        ['walks', 'talks', 'eats']
    into
       str: '(& [walks] (& [talks] [eats]))'
    """
    statement = ""
    if property:
        for i in range(len(list_of_elements)-2, -1, -1):
            if i == len(list_of_elements)-2:
                statement = "("+op+" ["+list_of_elements[-2]+"] ["+list_of_elements[-1]+"])"
            else:
                statement = "("+op+" ["+list_of_elements[i]+"] "+statement+")"
    else:
        for i in range(len(list_of_elements)-2, -1, -1):
            if i == len(list_of_elements)-2:
                statement = "("+op+" "+list_of_elements[-2]+" "+list_of_elements[-1]+")"
            else:
                statement = "("+op+" "+list_of_elements[i]+" "+statement+")"

    return statement


def tests():
    x = cascade_join(list_of_elements=["a","b","c"], op="&")
    print(x)
    assert x == "(& [a] (& [b] [c]))"

    x = cascade_join(list_of_elements=["a","b"], op="&")
    print(x)
    assert x == "(& [a] [b])"


def generate_simple_duck_nal_statements(truth_value, num_properties, item_list, tense=":|:"):
    """
    Duck typing, if it has all the same properties, it is the same type of object.
    """

    """

    <A <-> ?1>?
    <B <-> ?1>?
    <C <-> ?1>?


    <C <-> ?1>?
    # Expected Answer: <C <-> A>. creationTime=14 Truth: frequency=0.847072, confidence=0.094830
    <?1 <-> C>?
    # expected Answer: <A <-> C>. creationTime=14 Truth: frequency=0.847072, confidence=0.094830
    """

    statements = ["*volume=0"]

    if tense != "":
        tense = " " + tense

    for item in item_list:
        for pn in range(num_properties):
            statement = "<{item} --> [p{property_number}]>.{tense} %{truth_value}%".replace("{item}", item). \
                replace("{property_number}", str(pn)).replace("{truth_value}", str(truth_value)).replace("{tense}",
                                                                                                         str(tense))
            statements.append(statement)

    statements.append("// Now we declare the first two items are similar.")
    statements.append("<" + item_list[0] + " <-> " + item_list[1] + ">." + tense)
    statements.append("// Repeat the question, what is similar to each item")

    item = 'C'
    statements.append("// What is " + item + " most similar to? (No time information) Ideal answer any of A or B")
    statements.append("<" + item + " <-> A>?")
    statements.append(ASSERT_PREFIX + json.dumps(["<C <-> A>", "<C <-> B>", "<C <-> (A & B)>"]))

    return statements


def run_experiment(statements, num_properties):
    # print("________________________")
    # print("Input statements are:")
    # for statement in statements:
    #     print(statement)
    # print("________________________")
    print("Now sending these into ONA")

    nar_wrapper.Reset()

    t0 = time.time()
    for statement in statements:
        if len(statement) > len(ASSERT_PREFIX) and statement[:len(ASSERT_PREFIX)] == ASSERT_PREFIX:
            expected_json_str = statement[len(ASSERT_PREFIX):]
            list_of_valid_results = json.loads(expected_json_str)
            print("Expected one of: ", list_of_valid_results)
            print("Got: ", all_output)
            if "truth" in actual_output[0]:
                tv = float(actual_output[0]['truth']['frequency'][:-1])
            else:
                tv = 0.0
            if str(actual_output[0]['term']) in list_of_valid_results and tv > 0.5:
                print("Assert is OK\n\n\n\n")
                print("Time taken for",num_properties, "properties", time.time() - t0)
                return True
            else:
                print("Unexpected result.Marking as failed.")
                print("Time taken for",num_properties, "properties", time.time() - t0)
                # assert False
                return False
        elif len(statement) >= len(ASSERT_TRUE_PREFIX) and statement[:len(ASSERT_TRUE_PREFIX)] == ASSERT_TRUE_PREFIX:
            if "truth" in actual_output[0]:
                tv = float(actual_output[0]['truth']['frequency'][:-1])
            else:
                tv = 0.0
            if tv > 0.5:
                print("Truth Assert is OK\n\n\n\n")
                return True
            else:
                print("Unexpected result.Marking as failed.")
                return False
        elif len(statement) >= len(ASSERT_FALSE_PREFIX) and statement[:len(ASSERT_FALSE_PREFIX)] == ASSERT_FALSE_PREFIX:
            if "truth" in actual_output[0]:
                tv = float(actual_output[0]['truth']['frequency'][:-1])
            else:
                tv = 0.0
            if tv <= 0.5:
                print("Truth is False, Assert is OK\n\n\n\n")
                return True
            else:
                print("Unexpected result.Marking as failed.")
                return False
        elif len(statement) > 2 and statement[0:2] == '//':
            print(statement)
        elif len(statement) >= 7 and statement[0:7] == '//stats':
            print(nar_wrapper.GetStats())
        else:
            print("Sending Input", statement)
            all_output = nar_wrapper.AddInput(statement)
            actual_output = all_output['answers']

    print("Time taken for",num_properties, "properties", time.time() - t0)

    return False


def run_experiment_with_nalifier(nalifier, statements, num_properties, print_nars=False, reset_nars=True, debug_print=True):
    """

    :param nalifier:
    :param statements:
    :param num_properties:       equals the number of reduced dimensions (if dimensions are reduced)
    :param print_nars:           print NARs states/ output
    :param reset_nars:           needed between runs, but not between 'train/setup', and 'inference'.
    :param debug_print:          print our debugging  info
    :return:
    """
    # print("________________________")
    # print("Input statements are:")
    # for statement in statements:
    #     print(statement)
    # print("________________________")
    if debug_print:
        print("run_experiment_with_nalifier()  Now sending these into ONA")
    nar_result = {}
    if reset_nars:
        nar_wrapper.Reset()
    assert_ok_count = 0
    assert_bad_count = 0
    max_tv = 0.0
    t0 = time.time()
    for statement in statements:
        if len(statement) > len(ASSERT_PREFIX) and statement[:len(ASSERT_PREFIX)] == ASSERT_PREFIX:
            expected_json_str = statement[len(ASSERT_PREFIX):]
            list_of_valid_results = json.loads(expected_json_str)
            if debug_print:
                print("run_experiment_with_nalifier()  Got: ", nar_result)
                print("run_experiment_with_nalifier()  Expected one of: ", list_of_valid_results)
            if "truth" in nar_result['answers'][0]:
                tv = float(nar_result['answers'][0]['truth']['frequency'][:-1])
                max_tv = max(tv, max_tv)
            else:
                tv = 0.0
            if str(nar_result['answers'][0]['term']) in list_of_valid_results and tv > 0.5:
                if debug_print:
                    print("run_experiment_with_nalifier()  Assert is OK tv:",tv, "\n\n\n\n")
                    print("run_experiment_with_nalifier()  Time taken for",num_properties, "properties", time.time() - t0)
                assert_ok_count += 1
            else:
                if debug_print:
                    print("run_experiment_with_nalifier()  Unexpected result.Marking as failed.")
                    print("run_experiment_with_nalifier()  Time taken for",num_properties, "properties", time.time() - t0)
                assert_bad_count += 1
                # return False
        elif len(statement) >= len(ASSERT_TRUE_PREFIX) and statement[:len(ASSERT_TRUE_PREFIX)] == ASSERT_TRUE_PREFIX:
            if "truth" in nar_result['answers'][0]:
                tv = float(nar_result['answers'][0]['truth']['frequency'][:-1])
                max_tv = max(tv, max_tv)
            else:
                tv = 0.0
            if tv > 0.5:
                if debug_print:
                    print("run_experiment_with_nalifier()  Truth Assert is OK tv:",tv, "\n\n\n\n")
                assert_ok_count += 1
                # return True
            else:
                if debug_print:
                    print("run_experiment_with_nalifier()  Unexpected result.Marking as failed.")
                assert_bad_count += 1
        elif len(statement) >= len(ASSERT_FALSE_PREFIX) and statement[:len(ASSERT_FALSE_PREFIX)] == ASSERT_FALSE_PREFIX:
            if "truth" in nar_result['answers'][0]:
                tv = float(nar_result['answers'][0]['truth']['frequency'][:-1])
                max_tv = max(tv, max_tv)
            else:
                tv = 0.0
            if tv <= 0.5:
                print("Truth is False, Assert is OK tv:",tv, "\n\n\n\n")
                assert_ok_count += 1
            else:
                print("Unexpected result. Marking as failed. should be False was True", nar_result['answers'][0]['term'], "tv:", tv)
                assert_bad_count += 1
        elif len(statement) > 2 and statement[0:2] == '//':
            print(statement)
        elif len(statement) >= 7 and statement[0:7] == '//stats':
            print(nar_wrapper.GetStats())
        else:
            if not print_nars:
                print("run_experiment_with_nalifier()  Sending Input", statement)
            # all_output = NAR.AddInput(statement)
            # actual_output = all_output['answers']

            if nalifier is None:
                nar_result = nar_wrapper.AddInput(statement, Print=not print_nars)
            else: # pass to nalifier first
                result = nalifier.ShellInput(statement)
                if result is not None:
                    nar_result = nar_wrapper.AddInput(result, Print=not print_nars)

    overall_ok = assert_ok_count > 0 and assert_bad_count == 0
    if debug_print:
        print("run_experiment_with_nalifier()  Time taken for",num_properties, "properties", time.time() - t0, 'overall_ok', overall_ok)

    return overall_ok, assert_ok_count, assert_bad_count, max_tv




if __name__ == '__main__':
    tests()

    ref_list = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p",]

    for j in range(5, 15, 1):
        print("length", j, "_________________")
        l = ref_list[:j]
        l = [str(e) for e in l]
        print(cascade_join(l, property=False))


    for x in ref_list:
        print("<"+x+">.")