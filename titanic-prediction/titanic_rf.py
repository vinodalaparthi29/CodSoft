import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# 2. Pick columns to work with
df = df[['Survived', 'Pclass', 'Sex', 'Age']]

# 3. Fill missing ages with the average
df['Age'].fillna(df['Age'].median(), inplace=True)

# 4. Convert male/female to numbers (0 and 1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 5. Separate input features and the answer
X = df[['Pclass', 'Sex', 'Age']]    # the input features
y = df['Survived']                  # to predict


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 8. Test the model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))