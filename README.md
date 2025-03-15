# Sentiment Analyzer using AI

## 📌 Overview
This project is a **Sentiment Analysis System** built using **Python** and **Machine Learning** techniques. It classifies text data (such as product reviews or social media posts) into **Positive**, **Negative**, or **Neutral** sentiments. The project utilizes TF-IDF for text vectorization and LinearSVC for classification, with additional preprocessing and class balancing techniques.

## 🚀 Features
- Text Preprocessing (Cleaning, Normalizing)
- TF-IDF Vectorization with n-gram features
- Class Balancing using SMOTE
- Hyperparameter tuning using GridSearchCV
- Model Training and Evaluation with detailed metrics
- Saving trained model and vectorizer for future inference
- Prediction on new data using trained model (via `main.py`)
- Streamlit-based interactive web application

## 📁 Project Structure
```
├── train_sentiment_model.py     # Main training pipeline
├── main.py                     # Streamlit web app for sentiment prediction
├── sentiment_model.pkl         # Trained sentiment classification model
├── vectorizer.pkl              # Trained TF-IDF vectorizer
├── model_accuracy.txt          # Accuracy of the trained model
├── custom_sentiment_dataset.csv# Input dataset for training (custom dataset)
├── README.md                   # Project documentation
```

## 📊 Dataset Format
The input dataset should be a CSV file with the following columns:

| Sentiment  | Text               |
|------------|--------------------|
| Positive   | I love this!       |
| Negative   | Very disappointing |
| Neutral    | It's okay.         |

## ⚙️ Installation & Setup
1. **Clone the repository**:
```
git clone <repository-url>
cd <repository-directory>
```

2. **Prepare your dataset**:
Place your dataset as `custom_sentiment_dataset.csv` in the root folder.

## 💻 How to Run
Run the training script:
```
python train_sentiment_model.py
```

## ✅ Output
- **sentiment_model.pkl**: Trained sentiment analysis model.
- **vectorizer.pkl**: TF-IDF vectorizer for text transformation.
- **model_accuracy.txt**: Accuracy of the trained model.

## 🌐 Interactive Web App (main.py)
`main.py` is a **Streamlit-based interactive web application** that allows users to analyze sentiment through various modes. It uses the trained model and vectorizer for real-time predictions.

### 💡 Features of main.py:
- **Text Analysis:** Enter a sentence and get sentiment prediction instantly.
- **Voice Input:** Speak into the microphone to analyze sentiment of spoken words.
- **CSV Analysis:** Upload a CSV file containing a `text` column to perform batch sentiment analysis.
- **WhatsApp Chat Analysis:** Upload WhatsApp chat export files (`.txt`) to analyze sentiment of each message.

### ⚙️ How to Run the Web App:
```
streamlit run main.py
```

### 🌟 App Functionalities:
| Feature                  | Description                                          |
|-------------------------|-----------------------------------------------------|
| **Text Analysis ✍**       | Analyze sentiment of manually entered text.          |
| **Voice Input 🎤**         | Real-time sentiment analysis from speech input.     |
| **CSV Analysis 📄**        | Batch analyze sentiment from CSV file with 'text' column. |
| **WhatsApp Chat 💬**      | Analyze sentiment of messages in WhatsApp chats.    |

### 🚀 Example of Usage in App:
- **Enter Text:** _"I love this app!"_ → Positive 😊
- **Voice Input:** _Say: "This is frustrating"_ → Negative 😢
- **CSV Upload:** Upload a CSV, view analyzed sentiments, download results.
- **Chat File Upload:** Upload `.txt` file of WhatsApp chats, analyze sentiments of messages.

## 📈 Evaluation Metrics
- **Accuracy**
- **Balanced Accuracy**
- **Macro F1-Score**
- **Classification Report** with Precision, Recall, F1-score per class.

## 🚀 Future Improvements
- Integration with REST API for real-time predictions.
- Support for multi-language sentiment analysis.
- Adding deep learning models (LSTM, BERT).

## 🧑‍🎓 Contributors

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repository and submit pull requests.
