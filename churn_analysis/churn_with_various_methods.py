import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRFClassifier
from sklearn.model_selection import cross_val_score


def explo_ana(plot: bool = True, print_summary: bool = True):

    # Read the CSV file
    df = pd.read_csv("archive/file.csv")

    if plot:

        sns.histplot(df["Age"], kde=True)
        plt.show()

        sns.heatmap(df.corr(), annot=True)
        plt.show()

    df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
    df_majority = df[df["Exited"] == 0]
    df_minority = df[df["Exited"] == 1]

    df_minority_upsampled = resample(
        df_minority, replace=True, n_samples=len(df_majority), random_state=42
    )
    df_new = pd.concat([df_majority, df_minority_upsampled])

    if print_summary:
        print(df_new.describe())

    if plot:
        sns.histplot(df_new["Age"], kde=True)
        plt.show()

        sns.heatmap(df_new.corr(), annot=True)
        plt.show()

    for column in df_new.columns:
        if df_new[column].dtype == np.number:
            continue
        df_new[column] = LabelEncoder().fit_transform(df_new[column])

    X = np.array(df_new.drop(["Exited"], axis=1))
    y = np.array(df_new["Exited"])

    churn_scaler = MinMaxScaler()
    churn_scaler.fit(X)
    X = churn_scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, df_new


def dec_tree():

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy before cross validation: {100*accuracy:.2f}%")
    print(f"F1 score before cross validation: {100*f1:.2f}%")

    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        filled=True,
        feature_names=df_new.columns[:-1],
        class_names=["Not Exited", "Exited"],
    )
    plt.show()
    scores = cross_val_score(model, X_train, y_train, cv=5)

    accuracy = scores.mean()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"DT Accuracy after cross validation: {100*accuracy:.2f}%")
    print(f"DT F1 score after cross validation: {100*f1:.2f}%")
    print(f"DT Precision after cross validation: {100*precision:.2f}%")
    print(f"DT Recall after cross validation: {100*recall:.2f}%")
    print(f"DT F1 score after cross validation: {100*f1:.2f}%")

    return accuracy, precision, recall, f1


def rand_forest():

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy before cross validation: {100*accuracy:.2f}%")
    print(f"F1 score before cross validation: {100*f1:.2f}%")

    feature_importances = pd.Series(
        model.feature_importances_, index=df_new.columns[:-1]
    )
    feature_importances_sorted = feature_importances.sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(feature_importances_sorted.index, feature_importances_sorted.values)
    plt.xticks(rotation=90)
    plt.show()

    scores = cross_val_score(model, X_train, y_train, cv=5)
    accuracy = scores.mean()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"RF Accuracy after cross validation: {100*accuracy:.2f}%")
    print(f"RF F1 score after cross validation: {100*f1:.2f}%")
    print(f"RF Precision after cross validation: {100*precision:.2f}%")
    print(f"RF Recall after cross validation: {100*recall:.2f}%")
    print(f"RF F1 score after cross validation: {100*f1:.2f}%")

    return accuracy, precision, recall, f1


def grad_boost():

    X = df_new.drop(["Exited"], axis=1)
    y = df_new["Exited"]

    param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1],
        "max_depth": [3, 5, 7],
    }

    model = XGBRFClassifier(random_state=42)
    model.fit(X_train, y_train)
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    model = grid_search.best_estimator_

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy before cross validation: {100*accuracy:.2f}%")
    print(f"F1 score before cross validation: {100*f1:.2f}%")

    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances_sorted = feature_importances.sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(feature_importances_sorted.index, feature_importances_sorted.values)
    plt.xticks(rotation=90)
    plt.show()

    scores = cross_val_score(model, X_train, y_train, cv=5)
    accuracy = scores.mean()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"GB Accuracy after cross validation: {100*accuracy:.2f}%")
    print(f"GB F1 score after cross validation: {100*f1:.2f}%")
    print(f"GB Precision after cross validation: {100*precision:.2f}%")
    print(f"GB Recall after cross validation: {100*recall:.2f}%")
    print(f"GB F1 score after cross validation: {100*f1:.2f}%")

    return accuracy, precision, recall, f1


def compare_metrics():
    X_train, X_test, y_train, y_test, df_upsampled = explo_ana(
        plot=False, print_summary=False
    )

    models = {
        "Decision Tree": dec_tree,
        "Random Forest": rand_forest,
        "Gradient Boosting": grad_boost,
    }
    results = {}
    for name, model in models.items():
        accuracy, precision, recall, f1 = model()
        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 score": f1,
        }

    print("Results:")
    for name, metrics in results.items():
        print(f"{name}:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.2f}")

    return results


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df_new = explo_ana(
        plot=False, print_summary=False
    )
    dec_tree()
    rand_forest()
    grad_boost()
    df_comp = compare_metrics()
