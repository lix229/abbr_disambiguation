import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
import warnings
from sklearn.naive_bayes import GaussianNB
import feature_generation, preprocessing
from copy import deepcopy

# Ignore warnings
warnings.filterwarnings("ignore")

#!FIXME: Temparary model
#///-TODO Hyperparameter tuning
#TODO More classifiers MAYBE?
#TODO Pipeline


# Classifiers
def gaussian_naive_bayes(X_train, y_train):
    """
    gaussian_naive_bayes classifier

    :param X_train: training data
    :param y_train: training labels

    :return : clf (classifier) - trained classifier
    """
    gnb_clf = GaussianNB()
    gnb_clf.fit(X_train, y_train)

    kfold = StratifiedKFold(n_splits=5)
    results = cross_val_score(gnb_clf, X_train, y_train, cv=kfold)
    print('--' * 20)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    return gnb_clf

def gradient_boosting(X_train, y_train):
    """
    gradient boost classifier
    Using GridSearchCV to find the best parameters for the classifier
    stratified cv = 5 folds for cross validation

    :param X_train: training data
    :param y_train: training labels

    :return : clf (classifier) - trained gridsearchcv classifier
    """
    y_train  = y_train.squeeze()
    gb_clf = GradientBoostingClassifier(random_state=0)
    params = {
        'n_estimators' : [50, 100, 150, 200],
        'learning_rate' : [0.01, 0.05, 0.1, 0.5, 1],
        'max_features' : [5, 10, 20, 30, 40, 50, 100, len(X_train.columns)],
    }
    grid = GridSearchCV(estimator=gb_clf,
                    param_grid=params,
                    cv=5,
                    scoring="roc_auc",
                    refit=True,
                    verbose=0,
                    n_jobs=-1)
    grid.fit(X_train, y_train)
    kfold = StratifiedKFold(n_splits=5)
    results = cross_val_score(grid, X_train, y_train, cv=kfold)
    print('--' * 20)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    return grid

def random_forest(X_train, y_train):
    """
    random_forest classifier

    Parameters: X_train, y_train - training data

    Using GridSearchCV to find the best parameters for the classifier
    
    stratified cv = 5 folds for cross validation

    Returns: clf (classifier) - trained gridsearchcv classifier
    """
    y_train  = y_train.squeeze()
    rf_clf = RandomForestClassifier(random_state=0)
    params = {
        'n_estimators' : [50, 100, 150, 200],
        'max_features' :  [5, 10, 20, 30, 40, 50, 100, len(X_train.columns)],
        'max_depth': [5, 10, 20, 30],
        'min_samples_leaf' : [1, 3, 5]
    }
    grid = GridSearchCV(estimator=rf_clf,
                    param_grid=params,
                    cv=5,
                    scoring="roc_auc",
                    refit=True,
                    verbose=0,
                    n_jobs=-1)
    grid.fit(X_train, y_train)

    kfold = StratifiedKFold(n_splits=5)
    results = cross_val_score(grid, X_train, y_train, cv=kfold)
    print('--' * 20)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    return grid

def ada_boost(X_train, y_train):
    """
    ada boost classifier
    Using GridSearchCV to find the best parameters for the classifier

    :param X_train: training data
    :param y_train: training labels

    :return : clf (classifier) - trained gridsearchcv classifier
    """
    y_train  = y_train.squeeze()
    ada_clf = AdaBoostClassifier(random_state=0)
    params = {
        'n_estimators' : [50, 100, 150, 200],
        'learning_rate' :  [0.01, 0.05, 0.1, 0.5, 1],
        'max_features' : [5, 10, 20, 30, 40, 50, 100, len(X_train.columns)],
    }
    grid = GridSearchCV(estimator=ada_clf,
                    param_grid=params,
                    cv=5,
                    scoring="f1_macro",
                    refit=True,
                    verbose=0,
                    n_jobs=-1)
    grid.fit(X_train, y_train)

    kfold = StratifiedKFold(n_splits=5)
    results = cross_val_score(grid, X_train, y_train, cv=kfold)
    print('--' * 20)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    return grid

def feature_set_selection(abbr, clf):
    """
    Choose the best combination of features {abbr, n-gram_window_size, feature_type} for `clf`
    :param abbr: abbreviation:= str
    :param clf: classifier:= function
    :return: best combination of features:= tuple
    """
    feature_results = {'clf': clf.__name__}
    state = 12
    test_size = 0.2
    for window_size in range(2, 6):
        # n_gram = preprocessing.get_n_gram(abbr, window_size)
        #///!FIXME n_gram is being changed with option 1 and 2
        #!TESTING
        for feature_type in range(3):
            feature = pd.read_csv("./features/feature_%s_%s.csv" % (window_size, feature_type))
            label = pd.read_csv("./features/label_%s_%s.csv" % (window_size, feature_type))
            # n_gram_copy = deepcopy(n_gram)
            # feature = feature_generation.generate_feature(abbr, n_gram_copy, feature_type)
            # label = feature["sense"]
            # feature.drop("sense", axis=1, inplace=True)
            X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=test_size, random_state=state)
            trained = clf(X_train, y_train)
            predictions = trained.predict(X_test)
            feature_results[(abbr, window_size, feature_type)] = {'f1': f1_score(y_test, predictions), 'roc_auc': roc_auc_score(y_test, predictions)}
            print("Window Size: ", window_size)
            print("Feature Type: ", feature_type)
            print("F1 Score: ", f1_score(y_test, predictions))
            print("ROC AUC Score: ", roc_auc_score(y_test, predictions))
    #* If choosing F1 scores as primary metric
    #* feature_results = {('CVA', 2, 0): {'f1': 0.9701492537313433, 'roc_auc': 0.9580475998386446}, ('CVA', 2, 1): {'f1': 0.9635036496350364, 'roc_auc': 0.9384832593787817}, ('CVA', 2, 2): {'f1': 0.9635036496350364, 'roc_auc': 0.9384832593787817}, ('CVA', 3, 0): {'f1': 0.9777777777777777, 'roc_auc': 0.9655102864058088}, ('CVA', 3, 1): {'f1': 0.9705882352941176, 'roc_auc': 0.9519967728922951}, ('CVA', 3, 2): {'f1': 0.9705882352941176, 'roc_auc': 0.9519967728922951}, ('CVA', 4, 0): {'f1': 0.9705882352941176, 'roc_auc': 0.9519967728922951}, ('CVA', 4, 1): {'f1': 0.949640287769784, 'roc_auc': 0.9114562323517548}, ('CVA', 4, 2): {'f1': 0.9705882352941176, 'roc_auc': 0.9519967728922951}, ('CVA', 5, 0): {'f1': 0.9565217391304348, 'roc_auc': 0.9249697458652683}, ('CVA', 5, 1): {'f1': 0.9635036496350364, 'roc_auc': 0.9384832593787817}, ('CVA', 5, 2): {'f1': 0.9705882352941176, 'roc_auc': 0.9519967728922951}}
    print(feature_results)
    return feature_results

def multi_experiment(model_list, X_train, y_train, X_test, y_test):
    """
    
    train multiple models and evaluate on test set

    Parameters: model_list (list) - list of models to train
                X_train, y_train - training data
                X_test, y_test - test data

    IO: print out the performance of each model on test set
        Using joblib to save the trained model

    Returns: None

    """
    for model in model_list:
        name = model.__name__
        filename = name + '.pkl'
        trained = model(X_train, y_train)
        # joblib.dump(trained, "./trained/"+filename)
        predictions = trained.predict(X_test)
        print('--' * 20)
        print(name+':')
        # F1 Score
        print('F1 Score: ', f1_score(y_test, predictions))
        # Precision
        print('Precision: ', precision_score(y_test, predictions))
        # Recall
        print('Recall: ', recall_score(y_test, predictions))
        # Matthews Correlation Coefficient
        print('Matthews Correlation Coefficient: ', matthews_corrcoef(y_test, predictions))
        # ROC AUC
        print('ROC AUC: ', roc_auc_score(y_test, predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        print("Classification Report")
        print(classification_report(y_test, predictions))
        print('--' * 20)
        # fig =plot_feature_importance(trained).get_figure()
        # figname = name + '_auc.png'
        # fig.savefig('reports/figures/' + figname)
        # fpr, tpr, thresholds = roc_curve(y_test, trained.predict_proba(X_test)[:, 1])
        # plt.plot(fpr, tpr)
        # plt.savefig('reports/fig/' + figname)




if __name__ == '__main__':
    
    state = 12
    test_size = 0.2
    feature = pd.read_csv("./features/feature_%s_%s.csv" % (4, 1))
    label = pd.read_csv("./features/label_%s_%s.csv" % (4, 1))

    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=test_size, random_state=state)

    # model_list = [ada_boost, gaussian_naive_bayes, random_forest, gradient_boosting]
    model_list = [random_forest]
    multi_experiment(model_list, X_train, y_train, X_test, y_test)
    # feature_set_selection("CVA", random_forest)

