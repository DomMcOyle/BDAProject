"""
Python file containing function to execute a version of seq_scout which can exploit a distributed representation of the dataset.

NOTE: there is a main assumption when running seq_scout inside a worker node that is the portion of sequences of a given class
    for which the algorithm is being executed is small enough to be loaded fully in the main memory of the worker. This can be achieved
    by further splitting the filtered data in different workers or different executors.
"""
# Imports
from pyspark.sql.functions import col, udf
from pyspark.sql.functions import sum as fsum
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql import SparkSession
import math
import random
from utils import PriorityQueue, to_imm_pattern, mutable_seq_copy, save_patterns
import pickle
import sys
from time import time
import gc
import subprocess


def seq_scout(data, data_plus,target_class, numerics_max, top_k, iterations, theta, alpha):
    """
    Function implementing the seq scout algorithm on a distributed dataset.
   
    params:
        data: a pyspark dataframe containing the whole dataset of sequence examples
        data_plus: a pyspark dataframe which is a subsed of data. It contains all or a fraction of sequences
                   of the target class we are interested in (see NOTE in this module)
        target_class: label indicating the class for which pattern are being mined
        numerics_max: a dictionary containing as key all the numeric values names of a possible state and as a value an integer
                      representing the maximum number of unique values inside their domains
        top_k: number of best patterns to be returned at the end of the algorithm
        iterations: number of iterations of the algorithm to run
        theta: parameter in [0,1] used for filtering similar patterns (if a pattern is mined with similarity higher than
                theta it is discarded) theta = 1 disables filtering.
        alpha: parameter in (0,1) indicating the probability of removing a constraint on a numeric value of a state in a pattern.
    return:
        a list of top_k discriminative patterns (immutable objects) and their scores for the target_class class.
            each pattern is in the form [S1,...,Sn] where Si = (frozenset(inputs), (num_value:(interval),...))
    """
    # count the numeber of examples in the dataset and the number of examples with the target class in the dataset
    data_support = data.count()
    class_support = data.filter(col("class")==target_class).count()
    
    # create priority queue for patterns to be stored
    pi = PriorityQueue(k=top_k, theta=theta, cap_length=True) 
    
    # create priority queue for storing each class sequence and its UCB score
    # IMPORTANT: here data_plus is completely dumped on the memory. Be careful.
    scores = PriorityQueue(data_plus)
    tick = time()
    for N in range(1,iterations+1):
        # pop the sequence to be generalized
        _, Ni, mean_quality, sequence = scores.pop_first() 

        # generalize the sequence and add it to the patterns
        gen_seq, new_qual = play_arm(sequence, data, target_class, numerics_max, alpha, data_support, class_support)
        # NOTE: the priority queue extract always the smallest value, so all the scores are added as negatives
        pi.add((-new_qual, to_imm_pattern(gen_seq)))
        
        # update the quality and put back the sequence in the priority queue
        updated_quality = (Ni * mean_quality + new_qual) / (Ni + 1)
        ucb_score = compute_ucb(updated_quality, Ni + 1, N)
        scores.add((-ucb_score, Ni + 1, updated_quality, sequence))
        if N%100 == 0:
            tock = time()
            print(f"reached iteration {N} for class " + target_class )
            print(f"elapsed time:{tock-tick}")
            tick = tock
        
    
    return pi.get_top_k() # priority queue filters automatically if theta <1

def play_arm(sequence, data, target_class, numerics_max, alpha, data_support, class_support): 
    """
    Function for sequence generalization. Each sequence is generalized by randomly removing some inputs 
    (if the all the inputs of a state are removed, the state itself is removed) and randomly creating or removing
    constraint for numerics values.
    
    params:
        sequence: sequence to be generalized. Supposed to be in immutable form: [S1,...,Sn] where Si = [frozenset(inputs), 
                    [num_value:value,...]]
        data: reference to the pyspark dataframe of the whole dataset
        target_class: label indicating the class for which pattern are being mined
        numerics_max: a dictionary containing as key all the numeric values names of a possible state and as a value an integer
                      representing the maximum number of unique values inside their domains
        alpha: parameter in (0,1) indicating the probability of removing a constraint on a numeric value of a state in a pattern.
        data_support: number of examples in the dataset
        class_support: number of examples of target_class class in the dataset
        
    return:
        sequence_m: a generalized sequence in mutable form: [S1,...,Sn] where Si = [frozenset(inputs), [num_value:[interval],...]]
        quality: the computed WRAcc quality of sequence_m
    
    """
    sequence_m = mutable_seq_copy(sequence)
    # get the number of button pressed in the sequence
    tot_num_inputs = sum([len(state[0]) for state in sequence])
    # get a random number of input to be removed
    input_to_remove = random.randint(0, tot_num_inputs-1)

    for i in range(input_to_remove):
        selected_state_idx = random.randint(0, len(sequence_m)-1)
        selected_state = sequence_m[selected_state_idx][0] # we take the input itemset
        
        selected_state.remove(random.choice(list(selected_state))) # remove an element
        
        if len(selected_state) == 0: # if the state looses all the inputs, then it is removed
            sequence_m.pop(selected_state_idx)
            
    for _, numerics in sequence_m:
        for kind, value in numerics.items():
            # first we decide whether to remove the constraint or not
            if random.random() < alpha:
                numerics[kind] = [-float('inf'), float('inf')]
            else:         
                # we assume the dataset has indexes of unique values as numerics values
                # this speeds up this sampling process
                left_value = random.randint(0, value)
                right_value = random.randint(value, numerics_max[kind]-1)

                
                numerics[kind] = [left_value, right_value]

    # now we compute the quality measure
    quality = compute_WRAcc(data, sequence_m, target_class, data_support, class_support)

    return sequence_m, quality

def compute_ucb(score, Ni, N):
    """
    Function computing the UCB score, taken from the original repository.
    
    params:
        score: mean WRAcc score
        Ni: number of times the sequence has been subjected to generalization
        N: current iteration number
    
    return:
        the ucb score of the sequence, given the said parameters
    """
    return (score + 0.25) * 2 + 0.5 * math.sqrt(2 * math.log(N) / Ni)


def compute_WRAcc(data, subsequence, target_class, data_support, class_support): 
    """
    Function computing WRAcc on a distributed dataframe for a given subsequence
    
    params:
        data: reference to the pyspark dataframe of the whole dataset
        subsequence: pattern to be evaluated
        target_class: label indicating the class for which pattern are being mined
        data_support: number of examples in the dataset
        class_support: number of examples of target_class class in the dataset
    
    return:
        the WRAcc score of subsequence
    """
    # data support and class support were passed as it is useless to compute them everytime
    # the schema for the dataframe that will contain the results of the is_subsequence function is built
    schema = StructType([
        StructField("sub_support", IntegerType(),False),
        StructField("sub_sup_c", IntegerType(), False)
    ])
    # the udf is defined...
    udf_subsequence = udf(lambda x,y,z: is_subsequence(subsequence,target_class, x, y, z), schema)
    # ...and applied
    support_data = data.select(udf_subsequence(data.input_sequence,
                                               data.enc_num_sequence,
                                              col("class")).alias("tmp")).select(fsum("tmp.sub_support").alias("sub_support"),
                                                                                 fsum("tmp.sub_sup_c").alias("sub_sup_c"))
    # support_data contains for each example the information whether the sequence is generalized by 
    # subsequence ("sub_support") and if yes, if it also belongs to the same target_class ("sub_sup_c")
    sums = support_data.head()
    support = sums["sub_support"]
    class_pattern_count = sums["sub_sup_c"]

    del sums 
    del support_data
    
    try:
        class_pattern_ratio = class_pattern_count / support
    except ZeroDivisionError:
        return -0.25

    class_data_ratio = class_support / data_support
    wracc = support / data_support * (class_pattern_ratio - class_data_ratio)
    
    return wracc

def is_subsequence(subsequence,classsub, sequence_input, sequence_num, classsuper):
    """
    Function used to check whether a pattern generalizes a sequence and eventually if they belong to the same class
    
    params:
        subsequence: pattern to use for the check
        classsub: class of the pattern subsequence. If None, The function will only return if
                  subsequence generalizes the input sequence (1) or not (0)
        sequence_input: input part of the sequence to be checked
        sequence_num: numeric part of the sequence to be checked
        classsuper: class of the sequence considered
    returns:
        if classub is None:
            an integer. 1 if subsequence generalizes [sequence_input, sequence_num], 0 otherwise
        else:
            a tuple where:
            tuple[0] is an integer. 1 if subsequence generalizes [sequence_input, sequence_num], 0 otherwise
            tuple[1] is an integer. 1 tuple[0]==1 and subsequence and the input sequence have the same class, 0 otherwise
    """
    # sequence input is a list of lists of strings
    # sequence num is a list of rows
    i_sub = 0
    i_seq = 0
    while i_sub<len(subsequence) and i_seq<len(sequence_input):
        if subsequence[i_sub][0].issubset(sequence_input[i_seq]):
            if all([value >= subsequence[i_sub][1][numeric][0] and value <= subsequence[i_sub][1][numeric][1] for numeric, value in
                    sequence_num[i_seq].asDict().items()]):
                i_sub += 1
        i_seq += 1
        
    if i_sub == len(subsequence):
        is_sub = 1
    else:
        is_sub = 0
    
    if classsub is not None:
        if is_sub == 1 and classsub == classsuper:
            return (is_sub,1)
        else:
            return (is_sub,0)
    else:
        return is_sub


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("ERROR: Missing arguments. requiring top_k, iterations, theta and alpha")
    
    spark = SparkSession.builder.appName("RocketLeagueFEseq").getOrCreate()
    df = spark.read.format("json").load("hdfs://hdmaster:9000/user/ubuntu/dataset/processed_df")
    classes = df.select(col("class")).distinct().collect()
    class_list = [c["class"] for c in classes]
    # remove the noise class
    class_list.remove("-1")
    
    # load numerics_max
    with open("numerics_max.pickle", "rb") as file:
        numerics_max = pickle.load(file)
        
    print("classes to be processed:")
    print(class_list)
    for target in class_list:
        # isolate all the sequences of a given classes.
        # NOTE: we are still under the assumption that the class has a few 
        # examples. With bigger dataset the following data_plus
        # should be splitted and the seq_scout should be run on each of its split
        data_plus = df.filter(col("class")==target)
        print("Starting seq_scout for class " + target)
        patterns = seq_scout(df, data_plus,target, numerics_max, int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
        patt_filename = "pattern_for_class_" + target + ".pickle"
        save_patterns(patterns, patt_filename)
        
        # saving the patterns on the hdfs
        command = "/usr/local/hadoop-3.3.4/bin/hdfs dfs -moveFromLocal -f ./" +  patt_filename + " /user/ubuntu/dataset"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE) 
        output, error = process.communicate()
        print(output)
        print(error)
        
        gc.collect()
    spark.stop()