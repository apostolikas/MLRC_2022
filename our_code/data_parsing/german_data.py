import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype


def normalize(df):
    '''
    Rescale between zero and one.

    Params:
        df: pandas dataframe
    
    Return:
        rescaled dataframe
    '''
    result = df.copy()
    max_value = df.max()
    min_value = df.min()
    result = (df - min_value) / (max_value - min_value)
    return result


def get_german_data(filepath, wass_setup=False):
    '''
    Load German dataset splits.

    Params:
        filepath: string
        wass_setup: boolean
    
    Return:
        balanced_x, balanced_y, balanced_p, x_test.values, y_test.values, p_test.values: numpy arrays
    '''
    # From: https://www.kaggle.com/twunderbar/german-credit-risk-classification-with-keras
    data = pd.read_csv(filepath, index_col=0, sep=',')
    labels = data.columns
    # protected_label = 'Sex  # Confirmed that in MMD paper, sensitive value is sex.
    # ^ You can't believe everything you read these days. What they meant was age.
    protected_label = 'Age'
    # lets go through column 2 column
    for col in labels:
        if is_string_dtype(data[col]):
            if col == 'Risk':
                # we want 'Risk' to be a binary variable
                data[col] = pd.factorize(data[col])[0]
                continue
            if col == protected_label:  # Whatever you want for protected, if it's a categorical variable
                data[col] = pd.factorize(data[col])[0]
                continue
            # the other categorical columns should be one-hot encoded
            data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)
            data.drop(col, axis=1, inplace=True)
        else:
            if col == protected_label:
                # The standard is to threshold age by 25, but Wass uses 30
                age_threshold = 30 if wass_setup else 25
                data.loc[data[col] <= age_threshold, col] = 0
                data.loc[data[col] > age_threshold, col] = 1
                continue
            data[col] = normalize(data[col])

    # move 'Risk' back to the end of the df
    data = data[[c for c in data if c not in ['Risk', protected_label]] + [protected_label] + ['Risk']]
    # Shuffle all the data
    data = data.sample(frac=1)
    data_train = data.iloc[:800]
    data_test = data.iloc[800:]
    x_train = data_train.iloc[:, :-2]
    p_train = data_train.iloc[:, -2]
    y_train = data_train.iloc[:, -1]
    x_test = data_test.iloc[:, :-2]
    p_test = data_test.iloc[:, -2]
    y_test = data_test.iloc[:, -1]
    balanced_x, balanced_p, balanced_y = x_train.values, p_train.values, y_train.values
    print("Testing proportion y", np.mean(y_test.values))
    print("Training proportion y", np.mean(balanced_y))
    print("Testing proportion p", np.mean(p_test.values))
    print("Training proportion p", np.mean(balanced_p))


    from sklearn.linear_model import LinearRegression
    y_num = np.reshape(balanced_y, (-1, 1))
    p_num = np.reshape(balanced_p, (-1, 1))
    reg = LinearRegression().fit(y_num, p_num)
    print("Score", reg.score(y_num, p_num))
    print("Coeff", reg.coef_)
    return balanced_x, balanced_y, balanced_p, x_test.values, y_test.values, p_test.values

