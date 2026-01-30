import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

class SpamDetector:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=3000)
        self.bow_vectorizer = CountVectorizer(max_features=3000)
        self.multinomial_tfidf = MultinomialNB()
        self.multinomial_bow = MultinomialNB()
        self.gaussian_nb = GaussianNB()
        
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        return ' '.join(tokens)
    
    def load_and_preprocess_data(self, filepath):
        df = pd.read_csv(filepath, encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'text']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        return df
    
    def prepare_features(self, X_train, X_test):
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        
        X_train_bow = self.bow_vectorizer.fit_transform(X_train)
        X_test_bow = self.bow_vectorizer.transform(X_test)
        
        return X_train_tfidf, X_test_tfidf, X_train_bow, X_test_bow
    
    def train_models(self, X_train_tfidf, X_train_bow, y_train):
        self.multinomial_tfidf.fit(X_train_tfidf, y_train)
        self.multinomial_bow.fit(X_train_bow, y_train)
        self.gaussian_nb.fit(X_train_tfidf.toarray(), y_train)
        
    def evaluate_model(self, model, X_test, y_test, model_name):
        if model_name == "Gaussian NB":
            X_test = X_test.toarray()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        }
        return metrics, y_pred, y_proba
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'd:\\spam detection\\confusion_matrix_{model_name.replace(" ", "_")}.png')
        plt.close()
    
    def plot_roc_curve(self, y_test, y_proba, model_name):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(f'd:\\spam detection\\roc_curve_{model_name.replace(" ", "_")}.png')
        plt.close()
    
    def predict_new_emails(self, emails):
        processed = [self.preprocess_text(email) for email in emails]
        X_tfidf = self.tfidf_vectorizer.transform(processed)
        predictions = self.multinomial_tfidf.predict(X_tfidf)
        probabilities = self.multinomial_tfidf.predict_proba(X_tfidf)
        
        results = []
        for i, email in enumerate(emails):
            results.append({
                'Email': email[:100] + '...' if len(email) > 100 else email,
                'Prediction': 'SPAM' if predictions[i] == 1 else 'HAM',
                'Spam Probability': f'{probabilities[i][1]:.2%}'
            })
        return pd.DataFrame(results)

def main():
    detector = SpamDetector()
    
    print("=" * 80)
    print("SPAM EMAIL DETECTION USING NAIVE BAYES")
    print("=" * 80)
    
    print("\n[1] Loading and Preprocessing Data...")
    df = detector.load_and_preprocess_data('d:\\spam detection\\spam.csv')
    print(f"Dataset loaded: {len(df)} emails ({df['label'].sum()} spam, {len(df) - df['label'].sum()} ham)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print("\n[2] Feature Extraction (TF-IDF & Bag of Words)...")
    X_train_tfidf, X_test_tfidf, X_train_bow, X_test_bow = detector.prepare_features(X_train, X_test)
    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")
    print(f"BoW features: {X_train_bow.shape[1]}")
    
    print("\n[3] Training Models...")
    detector.train_models(X_train_tfidf, X_train_bow, y_train)
    print("✓ Multinomial NB (TF-IDF)")
    print("✓ Multinomial NB (BoW)")
    print("✓ Gaussian NB (TF-IDF)")
    
    print("\n[4] Evaluating Models...")
    results = []
    
    metrics_mnb_tfidf, pred_mnb_tfidf, proba_mnb_tfidf = detector.evaluate_model(
        detector.multinomial_tfidf, X_test_tfidf, y_test, "Multinomial NB (TF-IDF)"
    )
    results.append(metrics_mnb_tfidf)
    detector.plot_confusion_matrix(y_test, pred_mnb_tfidf, "Multinomial_NB_TFIDF")
    detector.plot_roc_curve(y_test, proba_mnb_tfidf, "Multinomial_NB_TFIDF")
    
    metrics_mnb_bow, pred_mnb_bow, proba_mnb_bow = detector.evaluate_model(
        detector.multinomial_bow, X_test_bow, y_test, "Multinomial NB (BoW)"
    )
    results.append(metrics_mnb_bow)
    detector.plot_confusion_matrix(y_test, pred_mnb_bow, "Multinomial_NB_BoW")
    detector.plot_roc_curve(y_test, proba_mnb_bow, "Multinomial_NB_BoW")
    
    metrics_gnb, pred_gnb, proba_gnb = detector.evaluate_model(
        detector.gaussian_nb, X_test_tfidf, y_test, "Gaussian NB"
    )
    results.append(metrics_gnb)
    detector.plot_confusion_matrix(y_test, pred_gnb, "Gaussian_NB")
    detector.plot_roc_curve(y_test, proba_gnb, "Gaussian_NB")
    
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(results_df.to_string(index=False))
    results_df.to_csv('d:\\spam detection\\performance_metrics.csv', index=False)
    
    print("\n[5] Testing on New Emails...")
    new_emails = [
        "Congratulations! You've won a $1000 gift card. Click here to claim now!",
        "Hi, can we schedule a meeting tomorrow at 3pm to discuss the project?",
        "URGENT: Your account will be suspended. Verify your identity immediately!",
        "Thanks for your email. I'll review the document and get back to you."
    ]
    
    predictions = detector.predict_new_emails(new_emails)
    print("\n" + predictions.to_string(index=False))
    predictions.to_csv('d:\\spam detection\\new_email_predictions.csv', index=False)
    
    print("\n" + "=" * 80)
    print("✓ All reports and visualizations saved successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
