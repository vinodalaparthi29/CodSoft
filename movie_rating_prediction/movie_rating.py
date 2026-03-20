import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("imdb_moviesIndia.csv", encoding='latin1')

df.dropna(subset=['Rating'], inplace=True)

df['Votes'] = df['Votes'].str.replace(',', '')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Votes'] = df['Votes'].fillna(df['Votes'].median())

df['Year'] = df['Year'].str.replace(r'[^\d]', '', regex=True)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Year'] = df['Year'].fillna(df['Year'].median())

df['Duration'] = df['Duration'].str.replace(' min', '').str.strip()
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
df['Duration'] = df['Duration'].fillna(df['Duration'].median())

df['Genre'] = df['Genre'].fillna('Unknown')
df['Genre'] = df['Genre'].astype('category').cat.codes

director_avg = df.groupby('Director')['Rating'].mean()
df['Director_avg_rating'] = df['Director'].map(director_avg)
df['Director_avg_rating'] = df['Director_avg_rating'].fillna(df['Rating'].mean())
df['Log_votes'] = np.log1p(df['Votes'])

X = df[['Year', 'Duration', 'Log_votes', 'Genre', 'Director_avg_rating']]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Model Score:", round(model.score(X_test, y_test), 2))
print("\nSample Predictions vs Actual:")
results = pd.DataFrame({
    'Actual'   : y_test.values[:8],
    'Predicted': predictions[:8].round(1)
})
print(results)

#Actual vs Predicted 
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, alpha=0.4, color='steelblue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.title('Actual vs Predicted Ratings')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()

#Predicted Rating Distribution
plt.figure(figsize=(8, 4))
plt.hist(y_test, bins=30, alpha=0.6, color='steelblue', label='Actual')
plt.hist(predictions, bins=30, alpha=0.6, color='green', label='Predicted')
plt.title('Actual vs Predicted Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('rating_distribution.png')
plt.show()

print("\nAll charts saved!")