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
    '''
    dataframe을 train과 test로 나눠서 dataframe에 "train"과 "test" column을 추가하는 함수

    input :
        df : pandas.DataFrame, 나눌 dataframe
        train_size : float, test data의 비율
        random_state : int, random seed

    output :
        df : pandas.DataFrame, "train"과 "test" column이 추가된 dataframe
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


def get_train_test_vector_all_feat (df) : 
    '''
    train, test data and labels 을 dataframe에서 추출해서 각각 반환

    input :
        df : pandas.DataFrame, dataframe

    output :
        tuple of train_data (mel), test_data (mel), train_data (chroma), test_data (chroma), y_val
            train_data (mel) : pandas.DataFrame, train data, mel spectrogram
            test_data (mel) : pandas.DataFrame, test data, mel spectrogram
            train_data (chroma) : pandas.DataFrame, train data, chroma scaled
            test_data (chroma) : pandas.DataFrame, test data, chroma scaled
            y_val : list, labels
    '''
    train_data = df[df["train"] == 1]
    test_data = df[df["test"] == 1]
    y_val = df[(df["test"] == 1)]["label"].tolist()

    mel_cols = get_mel_cols(df)
    chroma_cols = get_chroma_cols(df)

    return_list = []
    return_list.extend([train_data[mel_cols], test_data[mel_cols]])
    return_list.extend([train_data[chroma_cols], test_data[chroma_cols]])
    return_list.append(y_val)

    return tuple(return_list)


def get_mel_cols (df) :
    '''
    get mel columns from dataframe

    input :
        df : pandas.DataFrame, dataframe

    output :
        mel_cols : list, mel columns
    '''
    cols = df.columns
    mel_cols = [ col for col in cols if col.startswith("mf") or col.startswith("lm") ]

    return mel_cols


def get_chroma_cols (df) :
    '''
    get chroma columns from dataframe

    input :
        df : pandas.DataFrame, dataframe

    output :
        chroma_cols : list, chroma columns
    '''
    cols = df.columns
    chroma_cols = [ col for col in cols if col.startswith("chroma") ]

    return chroma_cols