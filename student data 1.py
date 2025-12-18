import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Data
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Model create
model = LinearRegression()

# Model train
model.fit(X_train, y_train)

# Prediction on test set
y_pred = model.predict(X_test)

print("Predicted (test):", y_pred)
print("Actual (test):", y_test.values)

# Accuracy check
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict for a new student 1
new_student_1 = pd.DataFrame(
    [[6, 80, 65]],
    columns=['study_hours', 'attendance', 'previous_score']
)
prediction_1 = model.predict(new_student_1)
print("Predicted Final Score (6, 80, 65):", prediction_1[0])

# Predict for a new student 2
new_student_2 = pd.DataFrame(
    [[7, 85, 75]],
    columns=['study_hours', 'attendance', 'previous_score']
)
prediction_2 = model.predict(new_student_2)
print("Predicted Final Score (7, 85, 75):", prediction_2[0])

# Final accuracy print (optional, already printed above)
accuracy = r2_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
