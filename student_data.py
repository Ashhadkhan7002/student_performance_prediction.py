import pandas as pd

data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'attendance': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'previous_score': [40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
    'final_score': [42, 47, 52, 57, 62, 67, 72, 77, 82, 87]
}

df = pd.DataFrame(data)

print(df)

# STEP 2: Features & Target
X = df[['study_hours', 'attendance', 'previous_score']]
y = df['final_score']

print(X)
print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Model create
model = LinearRegression()

# Model train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Predicted:", y_pred.values)
print("Actual:", y_test.values)

# Accuracy check
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

new_student = [[6, 80, 65]]
prediction = model.predict(new_student)

print("Predicted Final Score:", prediction[0])

from sklearn.linear_model import LinearRegression

# Model create
model = LinearRegression()

# Model train
model.fit(X_train, y_train)

print("Model trained successfully!")
