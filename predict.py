from data import *
import sys
import uuid
import os
import pickle


def main(path):

    name = str(uuid.uuid4())
    data = load_data(path, name, test=True)

    X_test = data.drop(['pid', 'SepsisLabel'], axis=1)

    with open('xgb.pkl', 'rb') as f:
        clf = pickle.load(f)

    # make predictions on the testing set
    y_pred = clf.predict(X_test)

    data['prediction'] = [int(pred) for pred in y_pred]
    data = data.sort_values(by='pid')
    data['id'] = data['pid'].apply(lambda x: f"patient_{x}")

    res = data[['id', 'prediction']]
    res.to_csv('prediction.csv', index=False)
    print('your predictions are in the file!')


if __name__ == '__main__':
    main(sys.argv[1])





