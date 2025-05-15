import time
import json
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from typing import List


def classify_with_NNR(data_trn: str, data_vld: str, df_tst: DataFrame) -> List:
    # todo: implement this function
    #  the data_tst dataframe should only(!) be used for the final predictions your return
    print(f'starting classification with {data_trn}, {data_vld}, predicting on {len(df_tst)} instances')

    df_train = pd.read_csv(data_trn)
    df_validation = pd.read_csv(data_vld)
    x_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    x_validation = df_validation.iloc[:, :-1]
    y_validation = df_validation.iloc[:, -1]
    x_test = df_tst.iloc[:, :-1]
    y_test = df_tst.iloc[:, -1]
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_test)

    radi = [round(0.1 * i, 1) for i in range(1, 21)]
    best_Radius = None
    best_Accuracy = 0

    for radius in radi:
        predications = []

        for i in range(len(x_validation_scaled)):
            x_val = x_validation.iloc[i]
            distances = x_train_scaled.apply(lambda x: ((x - x_val) ** 2).sum() ** 0.5, axis=1)
            neighbors = y_train[distances <= radius]

            if not neighbors.empty():
                predicated_label = neighbors.va



    predictions = list()  # todo: return a list of your predictions for test instances
    return predictions



































# todo: fill in your student ids
students = {'id1': '000000000', 'id2': '000000000'}

if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    df = pd.read_csv(config['data_file_test'])
    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  df.drop(['class'], axis=1))

    labels = df['class'].values
    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')
