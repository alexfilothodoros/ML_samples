import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("data/diabetes.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
_test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=_test_size, random_state=0 
)

model = XGBClassifier(
    learning_rate=0.02,
    subsample=0.8,
    colsample_bynode=0.8,
    n_estimators=600,
    reg_lambda=1e-5,
    objective="binary:logistic",
    nthread=1,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")


def plot_importance(model, features):
    plt.barh(features, model.feature_importances_)
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.show()


def grid_optimizer():
    param_grid = {
        "learning_rate": [0.01, 0.02, 0.05, 0.1,0.5],
        "n_estimators": [100, 200, 400, 600],
        "subsample": [0.5, 0.7, 0.8],
        "colsample_bytree": [0.5, 0.7, 0.8],
        "max_depth": [3, 4, 5],
        "reg_lambda": [13-6,1e-5, 1e-2, 0.1, 1, 100],
    }
    grid = GridSearchCV(model, param_grid, verbose=0, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(grid.best_estimator_)
    print(grid.best_score_)

    return



plot_importance(model, df.columns[:-1])
