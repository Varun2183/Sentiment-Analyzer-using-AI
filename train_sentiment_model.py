import joblib
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE

# Class for Sentiment Classification
class SentimentClassifier:
    def __init__(self, file_path):
        self.file_path = file_path
        self.vectorizer = None
        self.model = None

    # Function to clean text
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Load and preprocess data
    def load_and_prepare_data(self):
        print("Loading dataset...")
        try:
            df = pd.read_csv(
                self.file_path,
                encoding='utf-8',
                usecols=["Sentiment", "Text"],
                on_bad_lines='skip'
            )
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            exit(1)

        print("Sample data:")
        print(df.head())

        print("Mapping sentiment values...")
        df['Sentiment'] = df['Sentiment'].map({"Positive": 1, "Negative": -1, "Neutral": 0}).astype(int)

        print("Checking and cleaning null values...")
        df.dropna(subset=["Text"], inplace=True)

        print("Cleaning text data...")
        df['Text'] = df['Text'].apply(self.clean_text)

        return df

    # Vectorize text data
    def vectorize_data(self, df):
        print("Vectorizing text data with TF-IDF...")
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(df["Text"])
        y = df["Sentiment"]
        return X, y

    # Train and evaluate the model
    def train_and_evaluate(self, X, y):
        print("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        print("Splitting dataset into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

        print("Hyperparameter tuning with GridSearchCV...")
        param_grid = {'C': [0.1, 1, 10]}
        grid = GridSearchCV(LinearSVC(max_iter=10000, class_weight='balanced', dual=False), param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)

        print("Best parameters found:", grid.best_params_)
        self.model = grid.best_estimator_

        print("Cross-validating the model...")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation mean accuracy: {cv_scores.mean():.4f}")

        print("Evaluating model on test data...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Macro F1-score: {macro_f1:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Save model, vectorizer, and accuracy
        print("Saving model, vectorizer, and accuracy...")
        joblib.dump(self.model, 'sentiment_model.pkl')
        joblib.dump(self.vectorizer, 'vectorizer.pkl')
        with open("model_accuracy.txt", "w") as f:
            f.write(str(accuracy))
        print("Model, vectorizer, and accuracy saved successfully.")

# Run the classifier
if __name__ == "__main__":
    file_path = r"C:\\Final_Project\\Sentiment-Analyzer-using-AI-main\\custom_sentiment_dataset.csv"
    classifier = SentimentClassifier(file_path)
    df = classifier.load_and_prepare_data()
    X, y = classifier.vectorize_data(df)
    classifier.train_and_evaluate(X, y)