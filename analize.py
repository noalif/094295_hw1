import os
import pandas as pd
import copy
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from pandas_profiling import ProfileReport
from data import *

HIGH_THRESH = 0.7
LOW_THRESH = 0.5

def get_full_data_cols(df):
    '''
    gives us all columns that have no null values
    :param df: dataframe
    :return: list of these columns
    '''
    full_data = []
    for col in df.columns:
      if df[col].isnull().sum() == 0:
        full_data.append(col)
    return full_data


def print_per_dict(not_null_per, full_data_cols):
    '''
    checks which columns contain enough information
    :param not_null_per: percentage of not null value in columns
    :param full_data_cols: columns
    :return: data with enough information
    '''
    high_data = []
    low_data = []
    for feat, per in not_null_per.items():
        if per >= HIGH_THRESH:
            high_data.append(feat)
        if per <= LOW_THRESH:
            low_data.append(feat)

    print('features with a lot of information:')
    print([x for x in high_data if x not in full_data_cols])
    print('features with not enough information:')
    print(low_data)

    return high_data


# def calc_per_null(df, full_data_cols):
#     not_null_per = {}
#     for col in df.columns:
#         has_non_nan = df.groupby('pid')[col].apply(lambda x: x.notna().any())
#         percentage = has_non_nan.mean()
#         not_null_per[col] = percentage
#
#     print_per_dict(not_null_per, full_data_cols)
#
#     return not_null_per


def get_cols_above_threshold(df, threshold):
    '''
    calculate the percentage of non-null values for each column
    :param df: dataframe
    :param threshold:
    :return: list of column names where the percentage of non-null values is higher than the threshold
    '''

    not_null_percentages = df.notnull().mean() * 100
    return list(not_null_percentages[not_null_percentages > threshold].index)


# def calc_per_null_last(df_last, full_data_cols):
#     not_null_per = 1 - (df_last.isna().sum() / len(df_last))
#     not_null_per = not_null_per.to_dict()
#
#     print_per_dict(not_null_per, full_data_cols)
#
#     return not_null_per


def plot_scatter(df ,name_df):
    sns.pairplot(df[['O2Sat', 'SBP', 'MAP', 'Temp', 'null_per', 'Resp', 'HR', 'SepsisLabel']])
    title_scatter = str(name_df)
    plt.suptitle('scatter plot for ' + title_scatter, fontsize=16, y=1.05)

    plt.show()


def correlation_3_max(df):
    '''
    :param df: dataframe
    :return: 3 highest correlations
    '''
    df = df.drop('pid', axis=1)
    df_corr = df.corr()

    df_corr.values[np.tril_indices_from(df_corr)] = np.nan  # set the lower triangle to NaN
    upper_corr = df_corr.stack() # get the upper triangle elements
    upper_corr.dropna(inplace=True)  # drop NaN values

    healthy_top_3_corr = upper_corr.nlargest(3) # get the 3 highest correlation values

    for idx, val in healthy_top_3_corr.items(): # print the results
        row, col = idx
        print(f'{row} and {col}: {val}')


def analizing(patients_df, stage):
    '''
    :param patients_df: dataframe
    :param stage: before imputations or after
    :return: print scatter plot, hists & correlations
    '''

    print('##############  analyze part  #################')

    sick_patients, healthy_patients = separate_sick_healthy(patients_df)
    sub_groups = [[patients_df, 'all_patients'], [sick_patients, 'sick_patients'], [healthy_patients, 'healthy_patients']]
    for group in sub_groups:
        print('~~~~~~~~~~ Analyze ' + str(group[1]) + ' - ' + str(stage) + ' imputations: ~~~~~~~~~~')
        if group[1] == 'all_patients':
            plot_scatter(group[0], group[1])
        prof = ProfileReport(group[0], minimal=True)
        out_root = str(group[1]) + '_description_'+ str(stage) + '_imputations.html'
        prof.to_file(output_file=out_root)

        print('3 top correlations for: ' + str(group[1]))
        correlation_3_max(group[0])
        print('\n')


def create_confusion_matrix(actual, predicted):
    # Create confusion matrix
    cm = confusion_matrix(actual, predicted)

    # Convert to DataFrame for better visualization
    cm_df = pd.DataFrame(cm, columns=['Predicted Healthy', 'Predicted Sick'], index=['Healthy', 'Sick'])
    print(cm_df)


def post_analysis(predicted_df):
    '''
    demonstrates confusion matrix based on value of one column
    :param predicted_df: dataframe
    :return: print confusing matrix
    '''
    # create dataframes based on gender
    female_df = predicted_df[predicted_df['Gender'] == 0]
    male_df = predicted_df[predicted_df['Gender'] == 1]
    female_df_true = female_df[['SepsisLabel']]
    male_df_true = male_df[['SepsisLabel']]
    female_df_pred = female_df[['predictedSepsis']]
    male_df_pred = male_df[['predictedSepsis']]

    print('~~~~~~~~~~ confusion matrix for female: ~~~~~~~~~~')
    create_confusion_matrix(female_df_true, female_df_pred)
    print('~~~~~~~~~~ confusion matrix for male: ~~~~~~~~~~')
    create_confusion_matrix(male_df_true, male_df_pred)

    # create dataframes based on Age
    young_df = predicted_df[predicted_df['Age'] <= 35]
    adult_df = predicted_df[predicted_df['Age'] > 35]
    young_df_true = young_df[['SepsisLabel']]
    adult_df_true = adult_df[['SepsisLabel']]
    young_df_pred = young_df[['predictedSepsis']]
    adult_df_pred = adult_df[['predictedSepsis']]

    print('~~~~~~~~~~ confusion matrix for patients under 35 years old: ~~~~~~~~~~')
    create_confusion_matrix(young_df_true, young_df_pred)
    print('~~~~~~~~~~ confusion matrix for above 35 years old: ~~~~~~~~~~')
    create_confusion_matrix(adult_df_true, adult_df_pred)