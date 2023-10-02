"""
Python file containig utility functions for the seq scout algorithm
and for dataframe loading and saving

"""
import pickle
import heapq
import json
import subprocess
from pyspark.sql.types import StructType, IntegerType, StructField
from pyspark.sql.functions import col, udf
from pyspark.sql.functions import sum as fsum

def save_df(df, path, df_name, save_locally=False):
  """
  Function to serialize a pyspark dataframe to json while also saving its schema.

  Params:
    df: dataframe to serialize
    path: root directory where the files will be saved. The function will generate one json file 
          for the schema and another one for data.
    df_name: string containing the name of the dataframe to save (added to both json files)
    save_locally: boolean indicating whether to save the dataframe locally or on the hdfs.
  """
  schema_to_save = df.schema.json()  
  spath = './' if not save_locally else path
  with open(spath + "%s_schema.json"%df_name, "w") as json_file:
    json_file.write(json.dumps(schema_to_save, indent = 4))
  if not save_locally:
  # saving the patterns on the hdfs
    command = "/usr/local/hadoop-3.3.4/bin/hdfs dfs -moveFromLocal -f ./" + "%s_schema.json"%df_name + " " + path
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE) 
    output, error = process.communicate()
    print(output)
    print(error)
    # removing temp schema
    command = "rm ./" + "%s_schema.json"%df_name 
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE) 
    output, error = process.communicate()
    print(output)
    print(error)
  df.write.json(path + "/%s.json"%df_name, mode="overwrite")


def load_df(path, df_name, spark, load_locally=False):
  """
  Function to load a pyspark dataframe with the relative schema.

  Params: 
    path: root directory of the files of the 
          serialized dataset and its schema
    df_name: string containing the name of the saved dataframe
    spark: reference to the current spark session
    load_locally: boolean indicating whether to load the dataset from the local system (True) or from the
                  distributed one.
  Returns:
    loaded_df: the loaded dataset with the loaded schema
  """
  if not load_locally:
    command = "/usr/local/hadoop-3.3.4/bin/hdfs dfs -get " + path + "%s_schema.json"%df_name + " ./"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE) 
    output, error = process.communicate()
    print(output)
    print(error)
    spath = './'
  else:
    spath = path
    
  with open(spath + "%s_schema.json"%df_name, "r") as json_file:
    json_obj = json.load(json_file)
    loaded_schema = StructType.fromJson(json.loads(json_obj))
  loaded_df = spark.read.format("json") \
                        .option("header", "true") \
                        .schema(loaded_schema) \
                        .load(path + "/%s.json"%df_name)
  if not load_locally:
    # removing temp schema
    command = "rm ./" + "%s_schema.json"%df_name 
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE) 
    output, error = process.communicate()
    print(output)
    print(error)
  return loaded_df



def import_imm_sequence(seq):
    """
    Function importing a dataframe row as a immutable sequence.
    params:
        seq: tuple containing the list of input sets as first element and the list of rows of numerics values in the second
    return:
        an immutable sequence in the form (S1,...Sn) where Si = (frozenset(inputs), ((num_value, value), ...))
    """
    return tuple([tuple([frozenset(seq[0][i]), tuple(sorted(seq[1][i].asDict().items()))]) for i in range(len(seq[0]))])

def mutable_seq_copy(seq):
    """
    Function creating a mutable copy of an immutable sequence.
    params:
        seq: immutable sequence in the form (S1,...Sn) where Si = (frozenset(inputs), ((num_value, value), ...))
    return:
        copy: copy of seq in the form [S1,...Sn] where Si = [set(inputs), {num_value: value...}]
    """
    copy = []
    for i in seq:
        input_set = set(i[0])
        num_dict = {j[0] : j[1] for j in i[1]}
        copy.append([input_set, num_dict])
    return copy
        
def to_imm_pattern(pattern):
    """
    function converting a mutable pattern into an immutable one:
    params:
        pattern: immutable pattern in the form [S1,...Sn] where Si = [set(inputs), {num_value: [interval]...)
    return:
        a copy of pattern in the form (S1,...Sn) where Si = (frozenset(inputs), ((num_value, (interval))...))
    
    """
    return tuple([tuple([frozenset(i[0]), tuple(sorted([(key, tuple(value)) for key, value in i[1].items()]))]) for i in
                  pattern])

def save_patterns(patterns, filename):
    """
    function used to dump patterns in a file
    param: 
        patterns: a list of immutable patterns
        filename: string containing the target filename
  
    """
    with open(filename, "wb") as file:
        pickle.dump(patterns, file)

def load_patterns(filename):
    """
    function load to dumped patterns from a file
    param: 
        filename: string containing the target filename
    return:
        to_return: patterns: a list of immutable patterns
    """
    with open(filename, "rb") as file:
        to_return = pickle.load(filename)
    return to_return
    
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

class PriorityQueue(object):
    """
    class implementing the priority queue used in seq_scout.
    always returns the element or the list from it with the lowest associated priority value.
    attributes:
        k: number of top elements to return when calling get_top_k
        theta: parameter in (0,1) to be used when filtering. if theta = 1, filtering is automatically discarded
        cap_lenght: boolean indicating whether to keep the maximum lenght of the queue up to k (True) or not (False)
        heap: list respecting the heap property. we assume each element in heap has the reference priority value as first
            and any sequence or pattern as last element
        seq_set: set containing all the sequences already in the queue. Avoids re-inserting pattern or sequences.
    """
    def __init__(self, data=None, k=1,theta=1, cap_length=False):
        """
        NOTE: if data is different from None and is, in fact a pyspark dataset, it will 
        load the full dataframe on the main memory
        """
        self.k = k
        self.theta=theta
        self.cap_length=cap_length if k is not None else False
        if data is not None:  
            self.heap = [tuple([-float('inf'), 0, 0, import_imm_sequence((x["input_sequence"], x["enc_num_sequence"]))]) for x in data.collect()]
            heapq.heapify(self.heap)
            if cap_length and len(self.heap)>self.k:
                self.heap = heapq.nlargest(self.k, self.heap)
            self.seq_set = set([i[-1] for i in self.heap])
        else:
            self.heap = []
            self.seq_set = set()

    def add(self, elem):
        if elem[-1] not in self.seq_set:
            heapq.heappush(self.heap, elem)
            self.seq_set.add(elem[-1])
            if self.cap_length and len(self.heap)>self.k:
                self.heap = heapq.nsmallest(self.k, self.heap)
                self.seq_set = set([i[-1] for i in self.heap])
                
    def pop_first(self):
        head = heapq.heappop(self.heap)
        self.seq_set.remove(head[-1])
        return head
    
    def get_top_k(self, data=None):
        if self.theta == 1:
            return heapq.nsmallest(self.k, self.heap)
        else:
            return self.filter_patterns(data)
            
            
    def filter_patterns(self, data):
        """
        Method to filter patterns given the data, according to the jaccard index
        """ 
        assert data is not None
        elem_list = list(self.heap)
        elem_list.sort()
        
        output = []
        for i in range(len(elem_list)):
            similar = False
            for o in output:
                if self.similarity(mutable_seq_copy(elem_list[i][-1]),mutable_seq_copy(o[-1]), data)>self.theta:
                    similar=True
                    break
            if not similar:
                output.append(elem_list[i])
            
            if self.k is not None and len(output) == self.k:
                break

        return output
        
    def similarity(self, seq1, seq2, data):
        """
        Function computing the similarity between seq1 and seq2 considering data
            Params:
                 seq1: sequence to compare
                 seq2: sequence to compare
                 data: pyspark dataframe containing the sequences to use for computing the jaccard score
             returns:
                 the jaccard score of |sequenceces generalized by seq1 AND seq2|/|sequences generalized by
                     seq1 or seq2 or both|

        """
        def helper(x,y):
            """
            Helper function to compute the values of intersection and union for the score
            """
            a = is_subsequence(seq1, None, x,y, None)
            b = is_subsequence(seq2, None, x,y, None)
            return a*b, (1 if a==1 or b==1 else 0)

        outschema = StructType([
                    StructField("intersection", IntegerType(),False),
                    StructField("union", IntegerType(), False)])
        udf_jaccard = udf(lambda x,y: helper(x,y), outschema)
        ext_data = data.select(udf_jaccard(data.input_sequence,
                                           data.enc_num_sequence).alias("tmp"))\
                                           .select(fsum("tmp.intersection").alias("intersection"),
                                                   fsum("tmp.union").alias("union"))
        sums = ext_data.head()
        intersection = sums["intersection"]
        union = sums["union"]
        
        return intersection/union
        
       