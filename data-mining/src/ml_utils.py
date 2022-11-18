import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SequentialFeatureSelector

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, auc, roc_curve, RocCurveDisplay, accuracy_score, f1_score, make_scorer, plot_confusion_matrix


#import lightgbm as lgb
from IPython.display import Markdown, display

import warnings
warnings.filterwarnings('ignore')


from time import sleep

def apply(
    df,
    model_instance,
    parameter_grid,
    cross_validation=StratifiedKFold(n_splits=5),
    feature_selection=False,
    filter=False,
    oversample=False
):
    scaler = StandardScaler().fit(X=df)
    X = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
    y = df['status']

    instance_parameter_grid = {}

    for parameter_name, parameter_values in parameter_grid.items():
        instance_parameter_grid[f"model__{parameter_name}"] = parameter_values

    parameter_grid = instance_parameter_grid

    pipeline = []

    if feature_selection:
        rfe = SequentialFeatureSelector(model_instance, n_features_to_select=3)

        if filter:
            rfe = SelectKBest(f_classif, k=10)

        pipeline.append(('feature_selection', rfe))

    if oversample:
        pipeline.append(('sampling', SMOTE(n_jobs=-1)))

    pipeline.append(("model", model_instance))

    estimator = Pipeline(steps=pipeline)

    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
    scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(
        accuracy_score), "F1": make_scorer(f1_score)}

    grid_search = GridSearchCV(
        estimator,
        param_grid=parameter_grid,
        cv=cross_validation,
        scoring=scoring,
        refit="AUC"
    )


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 101) 
    grid_search.fit(X_train, y_train)
    grid_predictions = grid_search.predict(X_test) 

    if (feature_selection & oversample):
        display(Markdown("## Feature selection and oversample"))
    elif (feature_selection):
        display(Markdown("## Feature selection"))
    elif (oversample):
        display(Markdown("## Oversample"))
    else:
        display(Markdown("## No oversample nor feature selection"))



    display(Markdown(f"### **Classification report:** \n\n {classification_report(y_test, grid_predictions)}"))
    display(Markdown(f"### **Best score:** \n\n {grid_search.best_score_}"))
    display(Markdown(f"### **Best parameters:** \n\n {grid_search.best_params_}"))
    display(Markdown("### **Confusion matrix:** "))
    plot_confusion_matrix(grid_search, X_test, y_test, cmap="PuBuGn")  
    plt.show()
    display(Markdown("### **ROC curve:** "))

    fpr, tpr, thresholds = roc_curve(y_test, grid_predictions)
    roc_auc = auc(fpr, tpr)
    display_roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                   estimator_name='example estimator')
                                   
    display_roc.plot()
    plt.show()
    display(Markdown("---"))


    return grid_search