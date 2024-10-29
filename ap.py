import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Drop unnecessary columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Encode labels (ham=0, spam=1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Vectorize the text data (convert text to numeric features)
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Initialize the Naive Bayes classifier
nb = MultinomialNB()

# Train the classifier
nb.fit(X_train_vect, y_train)

# Make predictions on the test data
y_pred = nb.predict(X_test_vect)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Function to predict if a given text message is spam or not
def predict_message(message):
    vect_message = vectorizer.transform([message])
    prediction = nb.predict(vect_message)
    return 'Spam' if prediction[0] == 1 else 'Ham'

# Prompt the user for a text message to analyze
user_message = input("Enter a text message to analyze: ")
result = predict_message(user_message)
print(f'The message "{user_message}" is classified as: {result}')
