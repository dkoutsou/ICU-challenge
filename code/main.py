import matplotlib
matplotlib.use('TkAgg')
import pickle
import numpy as np
import warnings
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, precision_recall_curve, roc_curve,
        auc)
from sklearn.model_selection import train_test_split
from biosppy.signals.tools import signal_stats
from sklearn.model_selection import GridSearchCV

class ICUChallenge:
    def __init__(self, path_data, path_labels):
        self.initial_data =  pickle.load(open(path_data, 'rb'))
        self.labels =  pickle.load(open(path_labels, 'rb'))
        self.initial_data = np.swapaxes(self.initial_data, 1, 2)
        self.tasks = ['Low SA02', 'Low heartrate', 'Low respiration',
                'Low Systemic Mean', 'High Heartrate', 'High respiration', 
                'High Systemic Mean']
        self.classifiers = []
        # self.data = np.reshape(self.data, (self.data.shape[0], -1))
        self.data = np.zeros((self.initial_data.shape[0], 4 * 8)) 
        for i, sample in enumerate(self.initial_data):
            for j, variable in enumerate(sample):
                stats = signal_stats(variable)
                for k, stat in enumerate(stats):
                    self.data[i][8*j + k] = stat

        self.initial_data = self.initial_data.reshape(
                self.initial_data.shape[0], 4 * 16)
        self.data = np.concatenate((self.data, self.initial_data), axis=1)


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.data, self.labels, test_size=0.2)

    def train_classifier(self):
        tuned_parameters = {
                'max_depth' : [3, 5, 10],
                'n_estimators' : [500, 1000, 2000],
                'reg_alpha': [0.1, 5, 10, 100],
                'reg_lambda': [0.1, 5, 10, 100]
                }
        for i in range(len(self.tasks)):
            clf = GridSearchCV(XGBClassifier(), tuned_parameters, cv=5,
                       scoring='roc_auc')
            clf.fit(self.X_train, self.y_train[:, i])
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

    def evaluate(self):
        for i, model in enumerate(self.classifiers):
            y_pred = model.predict(self.X_test)
            y_pred = [round(x) for x in y_pred]
            precision, recall, thresholds = precision_recall_curve(self.y_test[:, i],
                    y_pred)
            prc_score = auc(recall, precision)
            roc_score = roc_auc_score(self.y_test[:, i], y_pred)
            print(self.tasks[i], roc_score, prc_score)

if __name__ == '__main__':
    # warnings.filterwarnings("ignore", category=DeprecationWarning) 
    np.random.seed(42)
    chal = ICUChallenge('data/samples.pk', 'data/labels.pk')
    chal.train_classifier()
    # chal.evaluate()
