import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from flask import Flask, request, render_template

app = Flask(__name__)

# Set the working directory to the directory of this script
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

# Load the dataset and remove the first unnamed column
df = pd.read_csv('email_db.csv')
del df[df.columns[0]]

def clean_text(text):
    """Clean the email text by removing HTML tags, non-alphabet characters, converting to lowercase, and stripping whitespaces."""
    if pd.isna(text):
        return ""  # Return empty string for NaN values
    text = str(text)  # Ensure the input is treated as a string
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)  # Keep only letters and spaces
    text = text.lower()  # Convert text to lowercase
    text = text.strip()  # Strip leading/trailing whitespaces
    return text

df['cleaned_text'] = df['Email Text'].apply(clean_text)
df['label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})

vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])

X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df['label'], test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Output the classification report and confusion matrix to the console
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None  # Initialize the prediction variable
    error = None  # Initialize the error variable for input validation messages
    if request.method == 'POST':
        email_text = request.form.get('email_text', '').strip()  # Get email text input and strip any leading/trailing whitespaces
        if not email_text:
            error = "Please enter some text to analyze."  # Set error message if input is empty
        elif len(email_text) > 5000:
            error = "Input is too long. Please enter less than 5000 characters."  # Set error message if input is too long
        else:
            cleaned_text = clean_text(email_text)
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_text)[0]
            prediction = 'Phishing Email' if prediction == 1 else 'Safe Email'

    # Return the template with error and prediction information
    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
