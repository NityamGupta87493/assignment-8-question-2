from flask import Flask, render_template, request
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the Naive Bayes model and TF-IDF vectorizer
model = joblib.load('model/naive_bayes_model.joblib')
vectorizer = joblib.load('model/tfidf_vectorizer.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    email_text = ""
    
    if request.method == 'POST':
        email_text = request.form['email_text']
        
        # Transform input using TF-IDF vectorizer
        email_tfidf = vectorizer.transform([email_text])
        
        # Predict using Naive Bayes model
        prediction = model.predict(email_tfidf)[0]
        
        # Convert prediction to human-readable result
        result = "🚫 Spam" if prediction == 1 else "✅ Not Spam"

    return render_template('index.html', result=result, email_text=email_text)

if __name__ == '__main__':
    app.run(debug=True)
