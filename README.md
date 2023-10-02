# BigDataAnalysis Project: Application of Pattern Mining and Machine Learning on a distributed setting
This repository contains the project for the Data Mining, Text Mining and Big Data Analyisis course (A.Y. 22-23)

## Main aim
The main aim for the project is to port the *Seq Scout* pattern mining algorithm on a distributed setting employing YARN, Hadoop and PySpark and then to reproduce and eventually improve the results of Machine Learning algorithm through Grid Search on the **RocketLeagueSkillshot** dataset.

## Dataset and task
The dataset (available [here](https://archive.ics.uci.edu/dataset/858/rocket+league+skillshots)) consists of 297 instances labeled with 7 different classes. Each instance corresponds to a a sequence of button inputs + in-game information (speed and position of the player, ball acceleration and speed...) describing a so-called "trick shot" in the soccer-like videogame <i>Rocket League</i>. As a consequence, each sequence is labeled with a class describing either a kind of trick shot:
- class 1: Ceiling Shot;
- class 2: Power Shot;
- class 3: Waving Dash;
- class 5: Air Dribbling;
- class 6: Front Flick;
- class 7: Musty Flick;

or indicating the sequence being Noise (class -1).

The main goal of this project is to reproduce and improve the method used in [this paper](https://www.researchgate.net/publication/343852416_A_Behavioral_Pattern_Mining_Approach_to_Model_Player_Skills_in_Rocket_League).
Said method can be broken down in the following 3 steps:
1. Extract general patterns describing each class with the *[Seq Scout](https://ieeexplore.ieee.org/document/8964149)* algorithm;
2. Use those patterns to encode the dataset with binary features (1 if the sequence is generalized by the pattern, 0 otherwise)
3. Apply the following ML algorithms to perform prediction on the class of the sequence: Decision Tree, Random Forest, XGBoost, SVM, Naive bayes.

The main paper focused more on the optimization of the parameters for the *Seq Scout* algorithm, while this project, other than proposing the
same algorithm implemented in a distributed way with PySpark with some minor differences, focuses also on the optimiziation of the ML models through Grid Search. 

## Repository summary

        .
        ├── dataset
        |    ├ encoded_df.json # folder containing the encoded train dataset
        |    ├ encoded_test.json # folder containing the encoded test dataset
        |    ├ processed_df.json # pre-processed training dataset
        |    ├ processed_test.json # pre-processed test dataset
        |    ├ encoded_df_schema.json # schema of the encoded training dataset
        |    ├ encoded_test_schema.json # schema of the encoded test dataset
        |    ├ rocket_league_skillshots.data # original raw dataset
        |    └ source.json # raw dataset converted from .data format to .json
        ├── models
        │    ├ decision_tree
        │    │  ├ base                  # base decision tree model
        │    │  └ hyperParameterTuned   # tuned decision tree model
        │    ├ nb
        │    │  ├ base                  # base naive bayes model
        │    │  └ hyperParameterTuned   # tuned naive bayes model
        │    ├ random_forest
        │    │  ├ base                  # base random forest model
        │    │  └ hyperParameterTuned   # tuned random forest model
        │    ├ svm
        │    │  ├ base                  # base svm model
        │    │  └ hyperParameterTuned   # tuned svm model
        │    └ xgb
        |        ├ base                  # base xgboost model
        |        └ hyperParameterTuned   # tuned xgboost model
        ├── patterns # folder containing the extracted patterns
        |    ├ pattern_for_class_1.pickle
        |    ├ pattern_for_class_2.pickle   
        |    ├ pattern_for_class_3.pickle   
        |    ├ pattern_for_class_5.pickle   
        |    ├ pattern_for_class_6.pickle   
        |    └ pattern_for_class_7.pickle   
        ├── setup scripts # contains scripts for the cluster setup
        |    ├ master # starts.sh and stops.sh are custom script to launch the hdfs+yarn cluster
        |    └ worker
        ├── MachineLearningAlgorithms.ipynb # jupyter notebook containing the code for ML algorithms training and testing
        ├── encoding.py # contains functions and script for the dataset encoding
        ├── numerics_max.pickle # file containing the dictionary of additional informations about numerical variables
        ├── preprocessing.py # contains functions and script for the dataset preprocessing
        ├── seq_scout.py # contains functions and script for the seq scout algorithm
        ├── utils.py # file containing utility functions
        └── README.md


## Quick guide
First of all, the dataset must be pre-processed, split into train and test and converted to a dataframe with the following commands:

    python3 preprocessing.py gen_json [path to .data file]
    [hdfs source folder]\hdfs -put [folder of .data file]/source.json [target folder]
    spark-submit --py-files ./utils.py preprocessing.py gen_dataframe [target folder]

The first command serves for our specific task and translates automatically the data from the .data format to the .json locally in order to easily
read it with PySpark. The second command takes the <code>source.json</code> file generated with the previous folder and moved to <code>target folder</code>
of the hdfs. It then produces two dataframe: <code>processed_df</code> and <code>test_df</code> which are then used as train (80%) and test (20%) set. Also
all the numerical values are converted to indexes in order to speed up the generalization process. To this effect it locally generates <code>numerics_max.pickle</code>
which is a dictionary {numeric_field_name : number_of_unique_values} which is used later.

Then, the algorithm to execute the *Seq Scout* algorithm must be executed with the following command:


    spark-submit --deploy-mode cluster --files [file folder]/numerics_max.pickle --py-files ./utils.py ./seq_scout.py [top_k] [iterations] [theta] [alpha]

where <code>top_k</code> is the number of pattern to keep for each class, <code>iterations</code> is the number of iterations of the algorithm to be run, <code>theta</code> which is the similarity threshold used to filter new pattern and <code>alpha</code> that is the probabilty of removing a constraint on a numerical value of a state in a pattern.
The script will produce six pattern_for_class_n.pickle files where the patterns found by the algoritm are stored.
The six pattern files provided with this repository have been produced considering the suggested values for the paper:
- <code>top_k</code>: 20
- <code>iterations</code>: 10000
- <code>theta</code>: 1 (no filtering)
- <code>alpha</code>: 0.8
patterns are not produced for the noise class, in order to follow closely the setup of the reference paper.

After that, the patterns must be moved locally:

        [path to hdfs]/hdfs dfs -get [pattern folder]/pattern_for_class* ./patterns/

and the dataset must be encoded with the following command:


    spark-submit --deploy-mode cluster --files patterns/pattern_for_class_1.pickle, patterns/pattern_for_class_2.pickle, patterns/pattern_for_class_3.pickle,
    patterns/pattern_for_class_5.pickle, patterns/pattern_for_class_6.pickle, patterns/pattern_for_class_7.pickle --py-files ./utils.py ./encoding.py [path_to_the_dataframes]

Which encodes both the training and the test set with the provided patterns.
Finally, the models can be train and tested using the notebook, executed by means of jupyter.

