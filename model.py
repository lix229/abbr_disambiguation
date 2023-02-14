import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
import warnings
from sklearn.naive_bayes import GaussianNB

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

    Parameters: X_train, y_train - training data

    Returns: clf (classifier) - trained classifier
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

    Parameters: X_train, y_train - training data

    Using GridSearchCV to find the best parameters for the classifier
    
    stratified cv = 5 folds for cross validation

    Returns: clf (classifier) - trained gridsearchcv classifier
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

    Parameters: X_train, y_train - training data

    Using GridSearchCV to find the best parameters for the classifier
    
    stratified cv = 5 folds for cross validation

    Returns: clf (classifier) - trained gridsearchcv classifier
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

def multi_experiment(model_list, X_train, y_train, X_val, y_val):
    """
    
    train multiple models and evaluate on test set

    Parameters: model_list (list) - list of models to train
                X_train, y_train - training data
                X_val, y_val - test data

    IO: print out the performance of each model on test set
        Using joblib to save the trained model

    Returns: None

    """
    for model in model_list:
        name = model.__name__
        filename = name + '.pkl'
        trained = model(X_train, y_train)
        joblib.dump(trained, "./trained/"+filename)
        predictions = trained.predict(X_val)
        print('--' * 20)
        print(name+':')
        # F1 Score
        print('F1 Score: ', f1_score(y_val, predictions))
        # Precision
        print('Precision: ', precision_score(y_val, predictions))
        # Recall
        print('Recall: ', recall_score(y_val, predictions))
        # Matthews Correlation Coefficient
        print('Matthews Correlation Coefficient: ', matthews_corrcoef(y_val, predictions))
        # ROC AUC
        print('ROC AUC: ', roc_auc_score(y_val, predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, predictions))
        print("Classification Report")
        print(classification_report(y_val, predictions))
        print('--' * 20)
        # fig =plot_feature_importance(trained).get_figure()
        # figname = name + '_auc.png'
        # fig.savefig('reports/figures/' + figname)
        # fpr, tpr, thresholds = roc_curve(y_val, trained.predict_proba(X_val)[:, 1])
        # plt.plot(fpr, tpr)
        # plt.savefig('reports/fig/' + figname)




if __name__ == '__main__':
    """
    OPTIONS: {
        0: basic n-gram features
        1: n-gram features with position
        2: n-gram features with position and distance
    }
    """
    OPTION = 2

    feature = pd.read_csv("./features/feature_option_%s.csv" % OPTION)
    label = pd.read_csv("./features/label_option_%s.csv" % OPTION)
    
    state = 12
    test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=test_size, random_state=state)

    model_list = [ada_boost, gaussian_naive_bayes, random_forest, gradient_boosting]
    multi_experiment(model_list, X_train, y_train, X_test, y_test)

