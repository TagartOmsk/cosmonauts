import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree._tree import TREE_LEAF


class TreeBoosting:
    def __init__(self, depth, verbose=False):
        self.linears = {}
        self.tree = None
        self.depth = depth
        self.verbose = verbose
        self.possible_values = None

    def fit(self, X, y):
        self.tree = DecisionTreeRegressor(max_depth=self.depth)
        self.tree.fit(X, y)

        leaves_count = np.sum([self.tree.tree_.children_left[i] == TREE_LEAF and
                               self.tree.tree_.children_right[i] == TREE_LEAF
                               for i in range(self.tree.tree_.node_count)])
        y_pred = self.tree.predict(X)

        self.possible_values = np.unique(y_pred)

        for i in range(self.possible_values.shape[0]):
            linreg = LinearRegression(fit_intercept=True)

            X_node = X[y_pred == self.possible_values[i]]
            y_node = y[y_pred == self.possible_values[i]]

            if self.verbose:
                print(f'Output #{i}\nLength {y_node.shape[0]}')

            linreg.fit(X_node, y_node)

            self.linears.update({i: linreg})

    def predict(self, X):
        idx = self.tree.predict(X)

        for i in range(self.possible_values.shape[0]):
            idx[idx == self.possible_values[i]] = i

        res = idx.copy()

        for i in range(res.shape[0]):
            if self.verbose:
                print(f'The chosen leaf was {int(res[i])}')
            res[i] = self.linears[int(res[i])].predict([X.iloc[i]])[0]


        return res