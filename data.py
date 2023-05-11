import os
import pandas as pd
import copy
# from analize import *
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import tqdm


def filter_outliers(df, test=True):
    """
    handeling outliers based on 5 and 95 percentile of distribution.
    :param df:
    :param test: for the train, loosing patients with weird amount of time in the ICU
    :return:
    """
    if df['SepsisLabel'].max() == 0:
        if df['ICULOS'].max() > 72  and not test:
            return False # if its train, we dont want this patient. its less than 95

        if df['HR'].max() > 114:
            df['HR'] = 114

        if df['HR'].max() > 114:
            df['HR'] = 114

    else:
        if df['HR'].max() > 120:
            df['HR'] = 120

        if df['HR'].max() > 120:
            df['HR'] = 120

    if df['SBP'].max() > 170:
        df['SBP'] = 170
    if df['HospAdmTime'].max() < -370:
        df['HospAdmTime'] = -370
    df['MAP'] = df['MAP'].clip(lower=60, upper=111)
    df['Resp'] = df['Resp'].clip(lower=12, upper=30)


def calc_null_per(df):
    """
    calculating for each patient, avg of null per column, and then avg on all columns
    :param df:
    :return:
    """
    null_percent = df.isna().mean()
    return null_percent.mean()


def load_data(data_dir, name, test):
    """
    loading the data, agregating and handeling outliers. imputing null with mean at the end
    :param data_dir:
    :param name: name for the saved csv
    :param test: for the outlier handeling
    :return:
    """
    create_df = True
    for filename in os.listdir(data_dir):
        data_dict = {}
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath, sep='|')
        df = df.head(len(df[df['SepsisLabel']==0])+1)
        df_filtered = filter_outliers(df, test)
        if df_filtered == False:
            continue
        data_dict["ICULOS"] = df["ICULOS"].max()
        data_dict["SepsisLabel"] = df["SepsisLabel"].max()
        start_index = filename.find("_") + 1
        end_index = filename.find(".")
        substring = filename[start_index:end_index]
        data_dict['pid'] = int(substring)
        data_dict["O2Sat"] = df["O2Sat"].min()
        data_dict["HR"] = df["HR"].max()
        data_dict["Resp"] = df["Resp"].mean()
        data_dict["SBP"] = df["SBP"].max()
        data_dict["MAP"] = df["MAP"].mean()
        data_dict["Age"] = df["Age"].max()
        data_dict["Gender"] = df["Gender"].max()
        data_dict["HospAdmTime"] = df["HospAdmTime"].min()
        data_dict['Temp'] = df['Temp'].max()
        data_dict['null_per'] = calc_null_per(df)

        df_columns = list(data_dict.keys())
        if create_df:
            data_df = pd.DataFrame(columns=df_columns)
            create_df = False

        data_df = pd.concat([data_df, pd.DataFrame([data_dict])], ignore_index=True)
    data_df = data_df.fillna(data_df.mean())
    print('finished preproccessing ' + name)
    data_df.to_csv(name+'.csv', index=False)
    return data_df


def separate_sick_healthy(df):
    """
    seperating the data to sick and healthy in order to multiply sick for data balance
    :param df:
    :return:
    """
    sick_mask = df.groupby('pid')['SepsisLabel'].transform(lambda x: (x == 1).any())
    healthy_mask = df.groupby('pid')['SepsisLabel'].transform(lambda x: (x == 0).all())

    sick_patients = df[sick_mask]
    healthy_patients = df[healthy_mask]

    return sick_patients, healthy_patients


def balance_data(df1, df2, ratio=0.35):
    """
    multiplying sick patients to balance the data
    :param df1: sick
    :param df2: healthy
    :param ratio: ratio to multiply sick
    :return:
    """
    num_row_fit = int(ratio * df2.shape[0])
    num_row_needed = num_row_fit - df1.shape[0]
    # randomly select num_row_needed rows
    selected_rows = df1.sample(n=num_row_needed, replace=True)
    # concatenate selected rows with original dataframe
    duplicated_df = pd.concat([df1, selected_rows])
    # reset index of concatenated dataframe
    duplicated_df = duplicated_df.reset_index(drop=True)
    together = pd.concat([duplicated_df, df2]).sample(frac=1).reset_index(drop=True)

    return together


def balanced_data(df):
    """

    :param df:
    :return:
    """
    sick_patients_imp, healthy_patients_imp = separate_sick_healthy(df)
    return balance_data(sick_patients_imp, healthy_patients_imp, ratio=0.4)
