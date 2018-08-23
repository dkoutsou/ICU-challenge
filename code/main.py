import matplotlib
matplotlib.use('TkAgg')
import pickle
import heapq
import numpy as np
import warnings
import argparse
from matplotlib import pyplot
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, precision_recall_curve, roc_curve,
        auc)
from sklearn.model_selection import train_test_split
from biosppy.signals.tools import signal_stats
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from prettytable import PrettyTable

class ICUChallenge:
    """A class that takes as input the data and the labels and then train an
    XGBoost classifier for every of the 7 tasks that we have in the eICU
    challenge."""
    def __init__(self, path_data, path_labels):
        self.data =  pickle.load(open(path_data, 'rb'))
        self.labels =  pickle.load(open(path_labels, 'rb'))
        self.data = np.swapaxes(self.data, 1, 2)
        self.tasks = ['Low SA02', 'Low heartrate', 'Low respiration',
                'Low Systemic Mean', 'High Heartrate', 'High respiration', 
                'High Systemic Mean']
        self.classifiers = []
        self.data = self.data.reshape(
                self.data.shape[0], 4 * 16)

    def tune_classifier(self):
        """A method that runs a grid search to find the best hyperparameters for
        the XGBoost classifier for every one of the tasks."""
        tuned_parameters = {
                'max_depth' : [3, 5, 10],
                'n_estimators' : [500, 1000, 2000],
                'reg_alpha': [0, 0.1, 5, 10, 100],
                'reg_lambda': [0, 0.1, 5, 10, 100]
                }
        for i in range(len(self.tasks)):
            X_train, X_test, y_train, y_test = train_test_split(
                    self.data, self.labels[:, i], test_size=0.2,
                    stratify=self.labels[:, i], shuffle=True, random_state=42)
            clf = GridSearchCV(XGBClassifier(), tuned_parameters, cv=5,
                       scoring='roc_auc')
            clf.fit(X_train, y_train)
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


    def train_classifier_and_evaluate(self):
        """A method that outputs the AUROC and AURPC score for every one of the
        seven tasks after training the classifier and then outputs the most
        important feature that the classification was done for each task."""
        best_params = [
                {'max_depth': 3, 'n_estimators': 500, 'reg_alpha': 0,
                   'reg_lambda': 0.1},
                {'max_depth': 3, 'n_estimators': 500, 'reg_alpha': 0.1,
                    'reg_lambda': 0.1},
                {'max_depth': 5, 'n_estimators': 2000, 'reg_alpha': 0,
                    'reg_lambda': 5},
                {'max_depth': 5, 'n_estimators': 500, 'reg_alpha': 0,
                    'reg_lambda': 0.1},
                {'max_depth': 5, 'n_estimators': 2000, 'reg_alpha': 0,
                    'reg_lambda': 10},
                {'max_depth': 5, 'n_estimators': 500, 'reg_alpha': 0.1,
                    'reg_lambda': 0.1},
                {'max_depth': 3, 'n_estimators': 500, 'reg_alpha': 0.1,
                    'reg_lambda': 0}
                ]
                
        t = PrettyTable(['Task', 'AUROC', 'AURPC'])
        for i in range(len(self.tasks)):
            try:
                model = XGBClassifier(**best_params[i])
            except:
                model = XGBClassifier()
            X_train, X_test, y_train, y_test = train_test_split(
                    self.data, self.labels[:, i], test_size=0.2,
                    stratify=self.labels[:, i], shuffle=True, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            precision, recall, thresholds = precision_recall_curve(y_test,
                    y_pred)
            rpc_score = auc(recall, precision)
            roc_score = roc_auc_score(y_test, y_pred)
            t.add_row([self.tasks[i], roc_score, rpc_score])
            print(self.tasks[i])
            largest = max(model.feature_importances_)
            for i, item in enumerate(model.feature_importances_):
                if abs(largest-item) < 0.0001:
                    largest_index = i
            if largest_index >= 0  and largest_index < 16:
                feature = "Heart rate"
            elif largest_index >= 16  and largest_index < 32:
                feature = "Respiratory rate"
            elif largest_index >= 32  and largest_index < 48:
                feature = "Blood pressure"
            else:
                feature = "Oxygen saturation"
            print("Most important feature, {1} in timestep {2}".format(
                i+1, feature, (largest_index+1)%16))
        print(t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-f",
            "--function",
            choices=('tune', 'train_eval'),
            help="choose the function of the code",
            action="store",
            default="train_eval"
            )
    parser.add_argument(
            "-d",
            "--data",
            help="choose the folder where the data is",
            action="store",
            default='data/samples.pk'
            )
    parser.add_argument(
            "-l",
            "--labels",
            help="choose the folder where the labels are",
            action="store",
            default='data/labels.pk'
            )

    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    np.random.seed(42)
    chal = ICUChallenge(args.data, args.labels)
    chal.train_classifier_and_evaluate()
