import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

df = pd.read_csv("data/sunspot_data.csv", nrows=500)

df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
df.drop(
    [
        "Unnamed: 0",
        "Year",
        "Month",
        "Day",
        "Date In Fraction Of Year",
        "Observations",
        "Indicator",
    ],
    axis=1,
    inplace=True,
)
df.set_index('Date', inplace=True)
df.resample('M').sum()
df['months elapsed'] = [i for i in range(len(df))]
df = df[df["Number of Sunspots"] != -1]
plt.plot(df["Date"], df["Number of Sunspots"])


training_mask = df["Date"] < "1819-05-11"
training_data = df.loc[training_mask]
print(training_data.shape)

testing_mask = df["Date"] >= "1819-05-11"
testing_data = df.loc[testing_mask]
print(testing_data.shape)


figure, ax = plt.subplots(figsize=(20, 5))
training_data.plot(ax=ax, label="Training", x="Date", y="Number of Sunspots")
testing_data.plot(ax=ax, label="Testing", x="Date", y="Number of Sunspots")
plt.show()


X_train = training_data["Date"]
y_train = training_data["Number of Sunspots"]

X_test = testing_data["Date"]
y_test = testing_data["Number of Sunspots"]


# XGBoost
cv_split = TimeSeriesSplit(n_splits=4, test_size=5)
model = XGBRegressor()
parameters = {
    "max_depth": [3, 4, 6, 5, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "n_estimators": [100, 300, 500, 700, 900, 1000],
    "colsample_bytree": [0.3, 0.5, 0.7],
}


X_train = [i for i in range(len(X_train))]
X_train = np.reshape(X_train, (len(X_train), 1))
grid_search = GridSearchCV(estimator=model, cv=cv_split, param_grid=parameters)
grid_search.fit(X_train, y_train)

# https://medium.com/mlearning-ai/time-series-forecasting-with-xgboost-and-lightgbm-predicting-energy-consumption-460b675a9cee

X_test = [292, 293, 294, 295, 296]
grid_search.predict(X_test)

plt.plot(X_train, y_train, label="Training")
plt.plot(X_test, y_test, label="Testing")
plt.plot(X_test, grid_search.predict(X_test), label="Prediction")
plt.legend()
plt.show()
