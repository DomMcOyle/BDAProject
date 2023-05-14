"""
Python file containig utility functions for the seq_scout algorithm


"""
import pickle
import heapq
import json
from pyspark.sql.types import StructType

def save_df(df, path, df_name):
  """
  Function to serialize a pyspark dataframe to json while also saving its schema.

  Params:
    df: dataframe to serialize
    path: root directory where the files will be saved. The function will generate one json file 
          for the schema and another one for data.
    df_name: string containing the name of the dataframe to save (added to both json files)
  """
  schema_to_save = df.schema.json()  
  with open(path + "%s_schema.json"%df_name, "w") as json_file:
    json_file.write(json.dumps(schema_to_save, indent = 4))
  df.write.json(path + "/%s.json"%df_name, mode="overwrite")


def load_df(path, df_name, spark):
  """
  Function to load a pyspark dataframe with the relative schema.

  Params: 
    path: root directory of the files of the 
          serialized dataset and its schema
    df_name: string containing the name of the saved dataframe
    spark: reference to the current spark session
  Returns:
    loaded_df: the loaded dataset with the loaded schema
  """
  with open(path + "%s_schema.json"%df_name, "r") as json_file:
    json_obj = json.load(json_file)
    loaded_schema = StructType.fromJson(json.loads(json_obj))
  loaded_df = spark.read.format("json") \
                        .option("header", "true") \
                        .schema(loaded_schema) \
                        .load(path + "/%s.json"%df_name)
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
                #TODO add filtering if necessary
    def pop_first(self):
        head = heapq.heappop(self.heap)
        self.seq_set.remove(head[-1])
        return head
    
    def get_top_k(self):
        if self.theta == 1:
            return heapq.nsmallest(self.k, self.heap)
        else:
            return 0
            #TODO add filtering
    
    
    
#            def add(self, elem):
#        if elem[-1] not in self.seq_set:
#            if len(self.heap)<self.k or not cap_length:
#                heapq.heappush(self.heap, elem)
#                self.seq_set.add(elem[-1])
#            else:
#                last_queue = max(self.heap, key=lambda x: x[0])
#                if elem[0]<last_queue[0]:
#                    
#            if self.cap_length and len(self.heap)>self.k:
#                self.heap = heapq.nsmallest(self.k, self.heap)
#                self.seq_set = set([i[-1] for i in self.heap])
#                #TODO add filtering if necessary