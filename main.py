from data import *
from analize import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from gradient_boosting import *
import os


def get_original_data(load=False):
    """
    :param load: if true, we're making all the preprocess. else, getting the ready made csv
    :return: train df balanced, test df
    """
    if load:
        train_df = load_data('/home/student/data/data/train', 'train_new', test=False)
        test_df = load_data('/home/student/data/data/test', 'test_new', test=True)
    else:
        train_df = pd.read_csv('train_new.csv')
        test_df = pd.read_csv('test_new.csv')

    train_df = balanced_data(train_df)
    return train_df, test_df


def optimize_models(train_df, test_df):
    """
    using optuna to find bast hyper parameters for all 3 models
    :param train_df:
    :param test_df:
    :return:
    """
    optimize_gb(train_df, test_df)
    optimize_rf(train_df, test_df)
    optimize_xgb(train_df, test_df)


def main():
    train_df, test_df = get_original_data(load=True)

    optimize_models(train_df, test_df)

    # create sick & healthy plots after imputations
    analizing(train_df, 'After')

    # ~~~~~~~~~~~~~~~ Get df with predictions ~~~~~~~~~~~~~~~
    predicted_df_gradient_boosting = train_model(train_df, test_df, 'gradient_boosting')
    predicted_df_random_forest = train_model(train_df, test_df, 'random_forest')
    predicted_df_xgb = train_model(train_df, test_df, 'xgb')
    #
    # ~~~~~~~~~~~~~~~ Create confusion matrix ~~~~~~~~~~~~~~~
    post_analysis(predicted_df_gradient_boosting)
    post_analysis(predicted_df_random_forest)
    post_analysis(predicted_df_xgb)



if __name__ == "__main__":
    main()