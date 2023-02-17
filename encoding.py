"""
Python script containg the procedure to encode the dataset given the found patterns

"""

from utils import is_subsequence
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf,col
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
import sys
import os
import pickle


def encode_dataset(dataframe, patterns):
    """
    Function encoding the dataset given a list of patterns as returned by seq_scout
    
    params:
        dataframe: dataframe containing the pre-processed dataset (see preprocessing.py)
        patterns: list of tuples containing the score and the relative pattern mined by seq_scout
    
    return:
        dataframe: input dataframe with new binary labels indicating 1 if the sequence is generalized by
                   a given pattern in patterns, 0 otherwise. All the columns are labeled from 0 to len(patterns)-1
        index - 1: number of columns added. Needed for subsequent filtering
    """
    index = 0
    for p in patterns:
        udf_subsequence = udf(lambda x,y: is_subsequence(mutable_seq_copy(p[1]),None, x, y, None), IntegerType())
        dataframe = dataframe.withColumn(str(index),udf_subsequence(dataframe.input_sequence, dataframe.enc_num_sequence))
        index = index + 1
    return dataframe, index - 1


if __name__ = "__main__":
    """
    The script will encode the dataset and produce a new dataframe in the same folder of the processed one.
    """
    if len(sys.argv) < 2:
        print("ERROR: missing arguments. Requires the path to the dataframe")
    
    spark = SparkSession.builder.appName("RocketLeagueDE").getOrCreate()
    # reads the dataframe
        
    df = spark.read.format("json").load(sys.argv[2]+ "processed_df")
    test = spark.read.format("json").load(sys.argv[2]+ "processed_test")
    print("dataframes loaded")
    
    classes = df.select(col("class")).distinct().collect()
    class_list = [c["class"] for c in classes]
    # remove the noise class
    class_list.remove("-1")
    class_list.sort()
    patterns = []
    
    # loading the patterns
    for c in class_list:
        with open("patterns_for_class_" + c + ".pickle", "rb") as pat_file:
            patterns = patterns + pickle.load(pat_file)
            
    enc_dataset, num_cols = encode_dataset(df, patterns)
    new_cols = [str(i) for i in range(num_cols+1)]
    enc_dataset = enc_dataset.select(["id","class"] + new_cols)
    va = VectorAssembler(inputCols=new_cols, outputCol="features")
    enc_dataset = va.transform(enc_dataset).select("id","class", "features")
    
    enc_test, num_cols = encode_dataset(test, patterns)
    new_cols = [str(i) for i in range(num_cols+1)]
    enc_test = enc_test.select(["id","class"] + new_cols)
    va = VectorAssembler(inputCols=new_cols, outputCol="features")
    enc_test = va.transform(enc_test).select("id","class", "features")
    
    
    print("showing results:")
    enc_dataset.show(3)
    enc_test.show(3)
    enc_dataset.write.format("json").save(sys.argv[2] + "encoded_df")
    enc_test.write.format("json").save(sys.argv[2] + "encoded_test")
    spark.quit()
        
    
    
    