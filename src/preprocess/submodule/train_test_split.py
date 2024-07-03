import pandas as pd

TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_STATE = 42

def train_test_split (df, train_size = TRAIN_TEST_SPLIT_RATIO, random_state = RANDOM_STATE) :
    '''
    split dataframe into train and test dataframe

    input
    df : pandas.DataFrame, dataframe to split
    train_size : float, ratio of test data
    random_state : int, random seed

    output
    df : pandas.DataFrame, dataframe with "train" and "test" columns
    '''
    normal_data_in_ratio = df[df["label"] == 1].sample(frac = train_size, random_state = random_state)
    # test_data = df.drop(normal_data_in_ratio.index)

    # normal_data_in_test = test_data[test_data["label"] == 1]
    # abnormal_data_in_test = test_data[test_data["label"] == -1]

    df["train"] = 0
    df["test"] = 0

    df.loc[normal_data_in_ratio.index, "train"] = 1
    df.loc[df.index, "test"] = 1 # use entire dataset for test

    return df