import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import seaborn as sns


def load_artembev():
    res = pd.read_excel('../data/1Artembev_PhC_sport_and_tripms.xls', header=0)
    res = res[['Physiological Cost (PhC)', 'TRIMP1', 'TRIMP2', 'TRIMP3', 'TRIMP4']]
    res['PhC'] = res['Physiological Cost (PhC)']
    res.drop(columns=["Physiological Cost (PhC)"], inplace=True)
    return res[11::]


def load_others():
    columns_to_remain = ['PhC', "t1(passive,slow)", "t2(passive,fast)", "t3(active,slow)", 't4(active,fast)']
    prokopbev = pd.read_excel('../data/2Prokopbev_PhC_sport_and_tripms.xls', header=0)
    prokopbev = prokopbev[columns_to_remain]
    prokopbev['TRIMP1'] = prokopbev["t1(passive,slow)"]
    prokopbev['TRIMP2'] = prokopbev["t2(passive,fast)"]
    prokopbev['TRIMP3'] = prokopbev["t3(active,slow)"]
    prokopbev['TRIMP4'] = prokopbev['t4(active,fast)']
    volkov = pd.read_excel('../data/3Volkov_PhC_sport_and_tripms.xls', header=0)
    volkov = volkov[columns_to_remain]
    volkov['TRIMP1'] = volkov["t1(passive,slow)"]
    volkov['TRIMP2'] = volkov["t2(passive,fast)"]
    volkov['TRIMP3'] = volkov["t3(active,slow)"]
    volkov['TRIMP4'] = volkov['t4(active,fast)']

    cols_to_remove = ["t1(passive,slow)", "t2(passive,fast)", "t3(active,slow)", 't4(active,fast)']

    prokopbev.drop(columns=cols_to_remove, inplace=True)
    volkov.drop(columns=cols_to_remove, inplace=True)
    return prokopbev[8::], volkov[34::]


def convert_to_3day_mean_format(df, add_day_numbers=False):
    data = df.copy()
    data["PhC"] = data["PhC"][1::]
    cols_to_drop = []
    for col in data.columns:
        data[col + '_mean_3_days'] = np.asarray([data.loc[i - 1:i + 1, col].mean() for i in data.index])
        cols_to_drop.append(col)
    old_phc_colname = "PhC_mean_3_days"
    new_phc_colname = 'PhC_mean_target_3_days'
    data[new_phc_colname] = data[old_phc_colname][3::].tolist() + 3 * [np.nan]
    data[old_phc_colname] = np.asarray([data.loc[i - 2:i, "PhC"].mean() for i in data.index])

    if add_day_numbers:
        data['day_number'] = np.arange(data.shape[0]) + 1

    data.drop(columns=cols_to_drop, inplace=True)
    data.dropna(inplace=True)

    y = data[new_phc_colname]
    X = data.drop(columns=[new_phc_colname])
    return X, y


def train_and_test(model, X, y, test_size=None, shuffle=True, random_state=42, draw=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    if draw:
        print(f'R² на отложенной выборке: {r2}')
        plt.figure(figsize=(7, 7))
        sns.scatterplot(y_test, y_pred)
        sns.scatterplot(y_test, y_test)
        plt.xlabel('Истинное значение')
        plt.ylabel('Прогноз')
    return r2


def train_and_test_nosplit(model, X, y, draw=True):
    model.fit(X, y)

    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)

    if draw:
        print(f'R² на всей выборке: {r2}')
        plt.figure(figsize=(7, 7))
        sns.scatterplot(y, y_pred)
        sns.scatterplot(y, y)
        plt.xlabel('Истинное значение')
        plt.ylabel('Прогноз')

    return r2
