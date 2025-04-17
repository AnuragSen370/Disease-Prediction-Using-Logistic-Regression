import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Typo fixed: 'symtom_data.csv' -> 'symptom_data.csv'
df = pd.read_csv('symptom_data.csv')

# Encode disease labels
le = LabelEncoder()
df['disease'] = le.fit_transform(df['disease'])

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:  # Typo: 'pk1' -> 'pkl'
    pickle.dump(le, f)

# Split data
X = df.drop('disease', axis=1)
y = df['disease']

# Typo fixed: missing comma between arguments
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model
with open('symptom_model.pkl', 'wb') as f:  # Typo: 'pk1' -> 'pkl'
    pickle.dump(model, f)

print("Logistic Regression model trained and saved.")
