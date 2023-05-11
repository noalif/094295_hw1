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
# data_dir = 'train_samples'

# read in the patient data as a pandas dataframe
# df = data.get_data(data_dir)
# df = data.combine_patient_data(df)
# df = data.impute_df_last(df, 'knn', 3)
# define the features and target variables

def search_GB_hyper_parameters(df, test_df):
    ###### best parameters for now:
    # n_estimators = 50, lr = 0.5, max_depth = 1
    # f1 score is: 0.7307692307692307
    X_train = df.drop(['pid', 'SepsisLabel'], axis=1)
    y_train = df['SepsisLabel']

    X_test = test_df.drop(['pid', 'SepsisLabel'], axis=1)
    y_test = test_df['SepsisLabel']

    # split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # train a gradient boosting classifier on the training set
    n_estimators = [50, 100, 150, 200]
    lr = [0.05, 0.1, 0.5]
    max_depth = [1,2,3]
    perm_list = list(itertools.product(n_estimators, lr, max_depth))
    f1_list = []
    class_weight = 1 / (y_train.value_counts(normalize=True))

    for i in tqdm(range(len(perm_list))):
        clf = GradientBoostingClassifier(random_state=42, n_estimators=perm_list[i][0], learning_rate=perm_list[i][1],
                                         max_depth=perm_list[i][2])
        clf.fit(X_train, y_train)

        # make predictions on the testing set
        y_pred = clf.predict(X_test)
        f1_list.append(f1_score(y_test, y_pred))
    max_index = f1_list.index(max(f1_list))
    print(f'n_estimators = {perm_list[max_index][0]}, lr = {perm_list[max_index][1]}, max_depth = {perm_list[max_index][2]}')
    print(f'f1 score is: {max(f1_list)}')
    print(f'recall score is: {recall_score(y_test, y_pred)}')


def train_gradient_boosting(df, test_df):
    X_train = df.drop(['pid', 'SepsisLabel'], axis=1)
    y_train = df['SepsisLabel']


    X_test = test_df.drop(['pid', 'SepsisLabel'], axis=1)
    y_test = test_df['SepsisLabel']
    clf = GradientBoostingClassifier(random_state=42) #'n_estimators': 426, 'max_depth': 3, 'learning_rate': 0.02181204782105362, 'subsample': 0.8321827059136073
    clf.fit(X_train, y_train)

    # make predictions on the testing set
    y_pred = clf.predict(X_test)
    print(f'f1 score is: {f1_score(y_test, y_pred)}')
    print(f'recall score is: {recall_score(y_test, y_pred)}')

def train_xgbg(df, test_df):
    X_train = df.drop(['pid', 'SepsisLabel'], axis=1)
    y_train = df['SepsisLabel']


    X_test = test_df.drop(['pid', 'SepsisLabel'], axis=1)
    y_test = test_df['SepsisLabel']
    clf = xgb.XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.5, max_depth=2)
    clf.fit(X_train, y_train)

    # make predictions on the testing set
    y_pred = clf.predict(X_test)
    print(f'f1 score is: {f1_score(y_test, y_pred)}')
    print(f'recall score is: {recall_score(y_test, y_pred)}')


def train_model(train_df, test_df, model=['gradient_boosting', 'xgb', 'random_forest']):
    print(model)
    X_train = train_df.drop(columns=['pid', 'SepsisLabel'])
    y_train = train_df['SepsisLabel']

    X_test = test_df.drop(['pid', 'SepsisLabel'], axis=1)
    y_test = test_df['SepsisLabel']

    if model == 'gradient_boosting':
        clf = GradientBoostingClassifier(random_state=42, n_estimators=120, max_depth=9, learning_rate=0.09868131222301617,
                                         subsample=0.7158681429515562)
    elif model == 'xgb':
        clf = xgb.XGBClassifier(random_state=42, n_estimators=916, learning_rate=0.0065049906505311155, max_depth=4,
                                subsample=0.6, colsample_bytree=0.3, gamma=2.01506026423888, reg_alpha=1.3434769131059672,
                                reg_lambda=0.04236394239252067, min_child_weight=1)

    elif model == 'random_forest':
        clf = RandomForestClassifier(random_state=42, n_estimators=117, max_depth=10)

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


def select_features(df):
    # Split the data into X (features) and y (target)
    X = df.drop(['pid', 'SepsisLabel'], axis=1)
    y = df['SepsisLabel']

    # Create a Random Forest classifier
    clf = GradientBoostingClassifier(random_state=42, n_estimators=50, learning_rate=0.5, max_depth=1)

    # Perform feature selection using a Random Forest classifier
    selector = SelectFromModel(estimator=clf, threshold=0.6)
    X_selected = selector.fit_transform(X, y)
    # print(selector.estimator_.coef_)

    # Get the selected feature names
    selected_feature_names = X.columns[selector.get_support()]

    # Print the selected feature names
    print("Selected features:", selected_feature_names)
    print(X_selected)

def optimize_xgb(df, test_df):
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
    print(f"Best accuracy: {-study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")


def optimize_gb(df, test_df):
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


