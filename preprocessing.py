"""
Python file containing functions to preprocess the dataset and pyspark dataframes

"""
import os
import json
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, explode, monotonically_increasing_id, posexplode, struct, collect_list
import pickle
from hdfs import InsecureClient

def read_dataset(path):
    """
    Function to read the dataset from the text file. Adapted from the one on the original repository
    params:
        path: sting containing the path to the dataset
    return:
        data: a list containing the read sequences. Each sequence is in the form [s1,s2...sn] where
              s1 is a dictionary containing:
              "input_sequence": a list of sets containing for each state the player inputs.
              "num_sequence": a list of dictionaries containing for each state the numerics values. The name
                              of the numeric values is the key, the values its values
              "class": a string containing the class value for the sequence
    """
    DISCRETE_INPUTS = {'up', 'accelerate', 'slow', 'goal', 'left', 'boost', 'camera', 'down', 'right', 'slide', 'jump'}
    data = []
    with open(path, "r") as file:
        dict_headers = next(file).split()
        new_line = dict()
        for line in file:
            if len(line.split()) <= 1:
                if new_line:
                    data.append(new_line)
                new_line = {"input_sequence": [] ,"num_sequence":[],"class": line.strip()}
            else:
                if len(dict_headers) != len(line.split()):
                    raise ValueError('Number of data and variables do not match')

                numerics = {}
                buttons = []

                for i, value in enumerate(line.split()):
                    if dict_headers[i] in DISCRETE_INPUTS:
                        if value == '1':
                            buttons.append(dict_headers[i])
                    else:
                        numerics[dict_headers[i]] = float(value)

                #state = [buttons, numerics]
                new_line["input_sequence"].append(buttons)
                new_line["num_sequence"].append(numerics)
        data.append(new_line)
    return data

def get_numerics(df):
    """
    Function building the dictionary where for each numeric values a ordered dataframe of unique its unique values is returned.
    This function builds also a dictionary where for each numeric value the maximum number of its unique values is recorded.
    
    params:
        df: pyspark dataframe containing the data of the entire dataset
    return:
        numerics_domains: dictionary containing as key the name of the numeric value and as value a dataframe with
                          two columns: "idx" and the numeric value name. "idx" contains an index from 0 to
                          the number of maximum unique values for a given numeric value. the corresponding values
                          are ordered from the smallest to the biggest.
        numerics_max: dictionary containing as key the name of the numeric value and as value the number of unique values
                     for the said numeric value  
    """
    subfields = df.schema["num_sequence"].dataType.elementType.fieldNames()
    numerics_domains = {}
    numerics_max = {}
    for c in subfields:
        field = "num_sequence." + c
        no_idx = df.select(explode(field).alias(c)).distinct().orderBy(c)
        numerics_domains[c] = no_idx.withColumn("idx", monotonically_increasing_id())
        numerics_max["idx"+c] = numerics_domains[c].count()
    return numerics_domains, numerics_max

def convert_numerics(df, numerics_domains):
    """
    Function to convert the numeric values in df in the correspoding index of their value in the domain. This
    is needed to speed up the generation of generalizations.
    params:
        df: pyspark dataframe containing the data of the entire dataset
        numerics_domains: output of get_numerics(df)
    return
        a dataframe containing two main columns: "_id" and "enc_num_sequence". "_id" is the id of the example in the dataset,
        "enc_num_sequence" is a version of the column "num_sequence" where each value is replaced with the corresponding 
        index. Also, each numeric values is renamed with "idx"+their original name
    """
    workdf = df.select(col("id").alias("_id"),posexplode("num_sequence").alias("pos","exp")).select("_id", "pos", "exp.*")
    needed_columns = [i for i in numerics_domains.keys()]
    needed_columns.append("pos")
    needed_columns.append("_id")
    for kind, unique_df in numerics_domains.items():
        print("processing " + kind + "...")
        expr1 = kind + " as _" + kind
        expr2 = "idx as idx" + kind
        workdf = workdf.join(unique_df.selectExpr(expr1, expr2), col(kind)==col("_"+kind))
        needed_columns.remove(kind)
        needed_columns.append("idx"+kind)
        workdf = workdf.select(needed_columns)
    needed_columns.remove("_id")
    needed_columns.remove("pos")
    workdf = workdf.orderBy("_id", "pos")
    workdf = workdf.groupBy("_id", "pos").agg(collect_list(struct([col(i) for i in needed_columns])).alias("enc_num_sequence"))
    return workdf.groupBy("_id").agg(collect_list(col("enc_num_sequence")[0]).alias("enc_num_sequence"))
    

if __name__ == "__main__":
    """
    this script pre-processes the data.
    IT IS ONLY A GUIDELINE, as it solves the problem considering the quantity of data available.
    
    if the argument added is "gen_json"
    data is first loaded, converted to a .json format and saved locally
    this is done for our specific use-case, as we suppose that a bigger
    dataset could be produced directly as a huge single .json file
    and not in the "proprietary" specific format of the original ".data" file
    and thus easily readable with pyspark
    
    the next steps instead, are always required, as they pre-process the dataset for the seq_scout algorithm
    
    ideally also the dataframe containing the lookup table (see get_numerics) could be serialized and then used 
    in a second moment to convert back the sequences with indexes. It won't be done as it is not explicitly required
    for this project.
    
    The final dataframe should contain the following columns: "id","input_sequence" ,"enc_num_sequence", "class"
    """
    if len(sys.argv) < 3:
        print("ERROR: missing arguments. add \"gen_json\" or \"gen_dataframe\" and the respective target folder")
        quit()
        
    if sys.argv[1] == "gen_json":
        data = read_dataset(sys.argv[2])
        with open("source.json", "w") as file:
            file.write(json.dumps(data))
            
    elif sys.argv[1] == "gen_dataframe": 
        spark = SparkSession.builder.appName("RocketLeagueDP").getOrCreate()
        # reads the dataframe
        
        df = spark.read.format("json").load(sys.argv[2]+"source.json")
        print("dataframe loaded")
        # adds the "id" column, associating an id to each example
        df = df.withColumn("id", monotonically_increasing_id())
        print("added id")
        # recovers unique values
        numerics_domains, numerics_max = get_numerics(df)
        
        # encodes the values with their unique value "id"
        encoded_numerics = convert_numerics(df, numerics_domains)
        print("converted indexes")
        # joins the original dataframe with the encoded one and applies 
        dfj = df.join(encoded_numerics, col("id")==col("_id")).select("id","input_sequence" ,"enc_num_sequence", "class")
        print("result:")
        dfj.show(5)
        with open("numerics_max.pickle", "wb") as file:
            pickle.dump(numerics_max, file)
        # saves the dataframe on the hdfs.
        # this helps to speed up operations, otherwise the spark engine had to
        # redo all the previous operations at each iteration
        # we also need to split the dataset in training and test in order to obtain
        # comparable results
        
        # the sample by method does not return the exact number of expected samples
        # for each class, an samplebykeyexact is not implemented currently in python
        # thus since the dataset is already enough shuffled, we are gonna select the first
        # 0.8 of examples taken by ordering them. It is not a random split, but its
        # the closest thing to a reproducible split we can add
        
        # we take the counts for each class and compute their respective 80% fraction
        counts = df.groupBy("class").count().collect()
        num_ex_required = {i["class"]:round(0.8*i["count"]) for i in counts}
        num_ex_per_class = {i["class"]:i["count"] for i in counts}
        
        # we craft the where expression to be run in order to isolate the correct number of examples required
        where_expr = ""
        j = 0
        for i in sorted(num_ex_per_class.keys()):
            where_expr = where_expr + "(class==" + i + " AND sample_id<" + str(num_ex_required[i]+j) + ") OR "
            j = j + num_ex_per_class[i]
        where_expr = where_expr[:-4] # discarding last or
        
        # isolating the ids
        df_samp = df.orderBy("class", "id")\
                    .withColumn("sample_id",monotonically_increasing_id())\
                    .where(where_expr)\
                    .withColumnRenamed("id", "_id")\
                    .select("_id")
        
        training_df = dfj.join(df_samp, dfj.id == df_samp._id)\
                         .select("id","input_sequence" ,"enc_num_sequence", "class")
                         
        print("showing training dataframe:")
        training_df.show(5)
        training_df.groupBy("class").count().show()
        # saving the training set
        training_df.write.format("json").save(sys.argv[2] + "processed_df")
        
        test_df = dfj.join(df_samp, dfj.id == df_samp._id, "leftanti")
        print("showing test dataframe:")
        test_df.show(5)
        test_df.groupBy("class").count().show()
        
        print("proof of disjunction")
        test_df.join(training_df, training_df.id == test_df.id).show()
        # saving the test
        test_df.write.format("json").save(sys.argv[2] + "processed_test")
        
        spark.quit()
    else:
        print("ERROR: wrong arguments. add \"gen_json\" or \"gen_dataframe\" and the respective target folder")
    

    