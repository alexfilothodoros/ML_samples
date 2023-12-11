import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/diabetes.csv")

df.info()

# Let's check for missing values and remove nan values  from the dataset if any
df.isnull().sum()
df = df.dropna()

# Let's check for unblanced data
df["Outcome"].value_counts()

# Turns ours that there is some imbalance. Let's fix it using SMOTE.
smote = SMOTE(random_state=0)
X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]
X_smote, y_smote = smote.fit_resample(X, y)
y_smote.value_counts()


# Let's create a function to remove outliers from the data
def clear_outlier(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    df_out = df.loc[(df[col] > fence_low) & (df[col] < fence_high)]
    return df_out


for col in df.columns:
    df = clear_outlier(df, col)

X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
SSX_train = scaler.fit_transform(X_train)
SSX_test = scaler.transform(X_test)

Rscaler = RobustScaler()
RSX_train = Rscaler.fit_transform(X_train)
RSX_test = Rscaler.transform(X_test)

train_test = [SSX_train, SSX_test]
result_ml_data = pd.DataFrame(columns=["Model_Name", "SS_Score", "RS_Score"])
model_name = ["LR", "KNN", "SVM", "NB", "DT", "RF"]  # Abridgment Names of Models
result_ml_data["Model_Name"] = model_name

# Logistic Regression

lr = LogisticRegression()
lr.fit(SSX_train, y_train)
lr_pred = lr.predict(SSX_test)
lr_score = accuracy_score(y_test, lr_pred)
result_ml_data["SS_Score"][0] = lr_score
lr.fit(RSX_train, y_train)
grid = GridSearchCV(lr, param_grid={"C": [0.001, 0.01, 0.1, 1, 10, 100]}, cv=5)
grid.fit(SSX_train, y_train)
grid.best_params_
grid.best_score_
lr_pred = lr.predict(RSX_test)
lr_score = accuracy_score(y_test, lr_pred)
result_ml_data["RS_Score"][0] = lr_score


# KNN

knn = KNeighborsClassifier()
knn.fit(SSX_train, y_train)
knn_pred = knn.predict(SSX_test)
knn_score = accuracy_score(y_test, knn_pred)
result_ml_data["SS_Score"][1] = knn_score
knn.fit(RSX_train, y_train)
grid = GridSearchCV(knn, param_grid={"n_neighbors": np.arange(1, 50)}, cv=5)
grid.fit(SSX_train, y_train)
grid.best_params_
grid.best_score_


# SVM

svm = SVC()
svm.fit(SSX_train, y_train)
svm_pred = svm.predict(SSX_test)
svm_score = accuracy_score(y_test, svm_pred)
result_ml_data["SS_Score"][2] = svm_score

grid = GridSearchCV(
    svm,
    param_grid={"C": [0.001, 0.01, 0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1]},
    cv=5,
)

grid.fit(SSX_train, y_train)
grid.best_params_
grid.best_score_


# Naive Bayes

nb = GaussianNB()
nb.fit(SSX_train, y_train)
nb_pred = nb.predict(SSX_test)
nb_score = accuracy_score(y_test, nb_pred)
result_ml_data["SS_Score"][3] = nb_score


grid = GridSearchCV(nb, param_grid={"var_smoothing": np.logspace(0, -9, num=100)}, cv=5)

grid.fit(SSX_train, y_train)
grid.best_params_
grid.best_score_
nb_pred = nb.predict(RSX_test)
nb_score = accuracy_score(y_test, nb_pred)
result_ml_data["RS_Score"][3] = nb_score


# Decision Tree

dt = DecisionTreeClassifier()
dt.fit(SSX_train, y_train)
dt_pred = dt.predict(SSX_test)
dt_score = accuracy_score(y_test, dt_pred)
result_ml_data["SS_Score"][4] = dt_score
dt.fit(RSX_train, y_train)

grid = GridSearchCV(dt, param_grid={"max_depth": np.arange(1, 50)}, cv=5)

grid.fit(SSX_train, y_train)
grid.best_params_
grid.best_score_


# Random Forest


rf = RandomForestClassifier()
rf.fit(SSX_train, y_train)
rf_pred = rf.predict(SSX_test)
rf_score = accuracy_score(y_test, rf_pred)
result_ml_data["SS_Score"][5] = rf_score
rf.fit(RSX_train, y_train)
grid = GridSearchCV(rf, param_grid={"n_estimators": [10, 50, 100, 200]}, cv=5)
grid.fit(SSX_train, y_train)
grid.best_params_
grid.best_score_


# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(
    result_ml_data["Model_Name"], result_ml_data["SS_Score"], label="StandardScaler"
)
plt.legend()
plt.show()


plt.figure(figsize=(10, 7))
plt.bar(result_ml_data["Model_Name"], result_ml_data["SS_Score"])
plt.xticks(rotation=0)
plt.xlabel("Model_Name")
plt.ylabel("SS_Score")
plt.title("Result with Standard Scaler")
plt.show()
