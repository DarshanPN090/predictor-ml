import pandas as pd
from sklearn.linear_model import LogisticRegression

# data
data = pd.DataFrame({
    "hours": [1, 2, 3, 4, 5],
    "pass":  [0, 0, 0, 1, 1]
})

X = data[["hours"]]
y = data["pass"]

# model
model = LogisticRegression()
model.fit(X, y)

# test
print(model.predict([[3]]))