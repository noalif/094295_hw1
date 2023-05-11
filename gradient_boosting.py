import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, confusion_matrix, recall_score
import itertools
from tqdm import tqdm
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import data
import optuna
import pickle


def train_model(train_df, test_df, model=['gradient_boosting', 'xgb', 'random_forest']):
    """
    training a model, returning predictions and printing f1 and recall
    :param train_df:
    :param test_df:
    :param model:
    :return:
    """
    print(model)
    X_train = train_df.drop(columns=['pid', 'SepsisLabel'])
    y_train = train_df['SepsisLabel']

    X_test = test_df.drop(['pid', 'SepsisLabel'], axis=1)
    y_test = test_df['SepsisLabel']

    if model == 'gradient_boosting':
        # Best f1: 0.8078
        # Best parameters: {'n_estimators': 870, 'max_depth': 5, 'learning_rate': 0.07871498655349331, 'subsample': 0.8380768213263907}
        clf = GradientBoostingClassifier(random_state=42, n_estimators=870, max_depth=3, learning_rate=0.07871498655349331,
                                         subsample=0.8380768213263907)
    elif model == 'xgb':
        # Best f1 : 0.8087
        # Best parameters: {'n_estimators': 296, 'max_depth': 7, 'learning_rate': 0.019409095935871035, 'subsample': 0.9,
        #              'colsample_bytree': 0.4, 'gamma': 0.48090361387529046, 'reg_alpha': 0.38497858145438274,
        #              'reg_lambda': 0.12743005595006632, 'min_child_weight': 4}

        clf = xgb.XGBClassifier(random_state=42, n_estimators=296, learning_rate=0.019409095935871035, max_depth=7,
                                subsample=0.9, colsample_bytree=0.4, gamma=0.48090361387529046, reg_alpha=0.38497858145438274,
                                reg_lambda=0.12743005595006632, min_child_weight=4)

    elif model == 'random_forest':
        # Best f1: 0.8081
        # Best parameters: {'n_estimators': 880, 'max_depth': 10, 'learning_rate': 0.04143541748879898}
        clf = RandomForestClassifier(random_state=42, n_estimators=880, max_depth=10)

    clf.fit(X_train, y_train)

    # make predictions on the testing set
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    print("Train recall: {:.4f}".format(recall_score(y_train, y_pred_train)))
    print("Train F1-score: {:.4f}".format(f1_score(y_train, y_pred_train)))

    print("Test recall: {:.4f}".format(recall_score(y_test, y_pred)))
    print("Test F1-score: {:.4f}".format(f1_score(y_test, y_pred)))



    res_df = test_df.copy()
    res_df['predictedSepsis'] = y_pred

    # Save the trained classifier
    with open(model+'.pkl', 'wb') as f:
        pickle.dump(clf, f)

    # print(test_df[test_df['pid'] > 6])
    return res_df


def feature_importance(test_df, clf_pickle):
    """
    finding the most important features when predicting test_df
    :param test_df:
    :param clf_pickle:
    :return:
    """
    print(clf_pickle.split('.')[0])
    X_test = test_df.drop(['pid', 'SepsisLabel'], axis=1)
    y_test = test_df['SepsisLabel']
    with open(clf_pickle, 'rb') as f:
        clf = pickle.load(f)

    # Use SelectFromModel to select the most important features
    sfm = SelectFromModel(clf, threshold='median')
    sfm.fit(X_test, y_test)

    selected_feature_indices = sfm.get_support(indices=True)
    selected_feature_names = X_test.columns[selected_feature_indices]

    # Print the indices of the selected features
    print('Selected features:', str(selected_feature_names))


def optimize_xgb(df, test_df):
    """
    finding best hyperparameters for xgb
    :param df:
    :param test_df:
    :return:
    """
    # Define the objective function for Optuna
    def objective(trial):
        # Define the hyperparameters to optimize
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'subsample': trial.suggest_discrete_uniform('subsample', 0.5, 1.0, 0.1),
            'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.1, 1.0, 0.1),
            'gamma': trial.suggest_loguniform('gamma', 0.01, 10),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.01, 10),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.01, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'random_state': 42
        }

        # Train the model with the selected hyperparameters
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Predict and calculate accuracy
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        # Return the negative accuracy as the objective metric to minimize
        return -f1
    X_train = df.drop(['pid', 'SepsisLabel'], axis=1)
    y_train = df['SepsisLabel']


    X_test = test_df.drop(['pid', 'SepsisLabel'], axis=1)
    y_test = test_df['SepsisLabel']

    # Define the study and optimize the hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    # Print the best hyperparameters and accuracy
    print(f"Best f1: {-study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")


def optimize_gb(df, test_df):
    """
    finding best hyperparameters for gb
    :param df:
    :param test_df:
    :return:
    """
    # Define the objective function for Optuna
    def objective(trial):
        # Define the hyperparameters to optimize
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'random_state': 42
        }

        # Train the model with the selected hyperparameters
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Predict and calculate accuracy
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        # Return the negative accuracy as the objective metric to minimize
        return -f1
    X_train = df.drop(['pid', 'SepsisLabel'], axis=1)
    y_train = df['SepsisLabel']


    X_test = test_df.drop(['pid', 'SepsisLabel'], axis=1)
    y_test = test_df['SepsisLabel']

    # Define the study and optimize the hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    # Print the best hyperparameters and accuracy
    print(f"Best f1: {-study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")


def optimize_rf(df, test_df):
    """
    finding best hyperparameters for rf
    :param df:
    :param test_df:
    :return:
    """
    # Define the objective function for Optuna
    def objective(trial):
        # Define the hyperparameters to optimize
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
            'random_state': 42
        }

        # Train the model with the selected hyperparameters
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Predict and calculate accuracy
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        # Return the negative accuracy as the objective metric to minimize
        return -f1
    X_train = df.drop(['pid', 'SepsisLabel'], axis=1)
    y_train = df['SepsisLabel']


    X_test = test_df.drop(['pid', 'SepsisLabel'], axis=1)
    y_test = test_df['SepsisLabel']

    # Define the study and optimize the hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    # Print the best hyperparameters and accuracy
    print(f"Best f1: {-study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")


