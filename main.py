import os
import re
from pathlib import Path

import joblib
import nltk
import pandas as pd
import requests

from dotenv import load_dotenv

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

FAKE_FILE = DATA_DIR / "Fake.csv"
REAL_FILE = DATA_DIR / "True.csv"

MODEL_FILE = MODEL_DIR / "fake_news_model.joblib"
VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"

# Load API key from the .env file.
load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("NEWS_API_KEY")

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 2. NLTK RESOURCE SETUP
# ============================================================

def setup_nltk():
    """Download required NLTK resources if they are missing."""

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]

    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK resource: {package_name}")

            success = nltk.download(
                package_name,
                quiet=True,
            )

            if not success:
                raise RuntimeError(
                    f"Could not download NLTK resource: {package_name}"
                )


# Initialize NLP tools.
print("Checking NLTK resources...")
setup_nltk()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

print("NLTK resources are ready.")


# ============================================================
# 3. TEXT PREPROCESSING
# ============================================================

def clean_text(text):
    """
    Normalize text, remove URLs and punctuation,
    tokenize words, remove stopwords, and lemmatize.
    """

    text = str(text).lower()

    # Remove URLs.
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep English letters and numbers.
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Tokenize text.
    tokens = word_tokenize(text)

    # Remove stopwords and short words, then lemmatize.
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(tokens)


# ============================================================
# 4. LOAD AND VALIDATE DATASETS
# ============================================================

def load_dataset():
    """Load fake and real news CSV files and assign labels."""

    if not FAKE_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {FAKE_FILE}\n"
            "Place Fake.csv inside the data folder."
        )

    if not REAL_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {REAL_FILE}\n"
            "Place True.csv inside the data folder."
        )

    print("\nLoading datasets...")

    try:
        fake = pd.read_csv(FAKE_FILE)
        real = pd.read_csv(REAL_FILE)
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            "One of the CSV files is empty. "
            "Replace blank files with the actual downloaded datasets."
        ) from error

    required_columns = {"title", "text"}

    for filename, dataframe in [
        ("Fake.csv", fake),
        ("True.csv", real),
    ]:
        missing_columns = required_columns - set(dataframe.columns)

        if missing_columns:
            raise ValueError(
                f"{filename} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

    # Labels: 0 = Fake, 1 = Real.
    fake["label"] = 0
    real["label"] = 1

    # Combine datasets.
    data = pd.concat(
        [fake, real],
        ignore_index=True,
    )

    # Safely combine title and article text.
    data["content"] = (
        data["title"].fillna("").astype(str)
        + " "
        + data["text"].fillna("").astype(str)
    )

    # Remove records with no usable text.
    data = data[
        data["content"].str.strip().ne("")
    ].copy()

    # Shuffle reproducibly.
    data = data.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    if data.empty:
        raise ValueError("The datasets contain no usable news articles.")

    if data["label"].nunique() < 2:
        raise ValueError(
            "Both fake and real news samples are required."
        )

    print("Datasets loaded successfully!")
    print(f"Total articles: {len(data)}")

    print("\nClass distribution:")
    print(
        data["label"]
        .value_counts()
        .rename(index={0: "Fake", 1: "Real"})
    )

    return data


# ============================================================
# 5. TRAIN AND EVALUATE THE MODEL
# ============================================================

def train_model():
    """Train TF-IDF and Logistic Regression, then evaluate."""

    data = load_dataset()

    X = data["content"]
    y = data["label"]

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Cleaning training text...")
    X_train_clean = X_train.apply(clean_text)

    print("Cleaning testing text...")
    X_test_clean = X_test.apply(clean_text)

    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_df=0.8,
        min_df=5,
        ngram_range=(1, 2),
    )

    # Fit vocabulary only on training data.
    X_train_vectorized = vectorizer.fit_transform(X_train_clean)

    # Transform test data without fitting on it.
    X_test_vectorized = vectorizer.transform(X_test_clean)

    print("Training Logistic Regression...")

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_vectorized, y_train)

    # Evaluate the model on held-out test data.
    y_pred = model.predict(X_test_vectorized)

    print("\n========== MODEL PERFORMANCE ==========")

    print(
        "Accuracy:",
        round(accuracy_score(y_test, y_pred), 4),
    )

    print("\nConfusion Matrix (rows=true, columns=predicted):")
    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Fake News", "Real News"],
            zero_division=0,
        )
    )

    # Save trained artifacts for future use.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print("\nModel saved to:", MODEL_FILE)
    print("TF-IDF vectorizer saved to:", VECTORIZER_FILE)

    return model, vectorizer


# ============================================================
# 6. PREDICT NEWS
# ============================================================

def predict_news(news, model, vectorizer):
    """Predict whether supplied news resembles fake or real training data."""

    if not isinstance(news, str) or not news.strip():
        return "Please enter some news text."

    cleaned_news = clean_text(news)

    if not cleaned_news.strip():
        return (
            "Not enough usable text. "
            "Please enter a longer headline or article."
        )

    news_vectorized = vectorizer.transform([cleaned_news])

    prediction = model.predict(news_vectorized)[0]
    probabilities = model.predict_proba(news_vectorized)[0]

    if prediction == 0:
        result = (
            f"Fake News prediction "
            f"(model probability: {probabilities[0]:.2%})"
        )
    else:
        result = (
            f"Real News prediction "
            f"(model probability: {probabilities[1]:.2%})"
        )

    return (
        result
        + "\nNote: This is a model prediction, "
        "not independent verification of the article."
    )


# ============================================================
# 7. FETCH LIVE NEWS FROM NEWSAPI
# ============================================================

def get_live_news():
    """Fetch the latest available Indian headlines from NewsAPI."""

    if not API_KEY:
        print(
            "\nNEWS_API_KEY is not configured. "
            "Check the .env file."
        )
        return []

    url = "https://newsapi.org/v2/top-headlines"

    params = {
        "country": "in",
        "pageSize": 5,
        "apiKey": API_KEY,
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15,
        )

        response.raise_for_status()
        result = response.json()

    except requests.exceptions.RequestException as error:
        print("NewsAPI request failed:", error)
        return []

    except ValueError:
        print("NewsAPI returned invalid JSON.")
        return []

    if result.get("status") != "ok":
        print(
            "NewsAPI error:",
            result.get("message", "Unknown error"),
        )
        return []

    news_list = []

    for article in result.get("articles", []):
        title = article.get("title") or ""
        description = article.get("description") or ""
        content = article.get("content") or ""

        full_text = " ".join(
            part
            for part in [title, description, content]
            if part.strip()
        )

        if full_text:
            news_list.append(full_text)

    return news_list


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

def main():
    """Run training, live news predictions, and manual predictions."""

    print("\n" + "=" * 50)
    print("          FAKE NEWS DETECTION SYSTEM")
    print("=" * 50)

    # Train and save model on each run.
    model, vectorizer = train_model()

    # Test model on live headlines.
    print("\n========== LIVE NEWS FROM API ==========")

    live_news = get_live_news()

    if not live_news:
        print(
            "No live news fetched. "
            "Check API key, internet connection, and API access."
        )
    else:
        for index, news in enumerate(live_news, start=1):
            print(f"\nNews {index}: {news[:200]}...")
            print(
                "Prediction:",
                predict_news(news, model, vectorizer),
            )

    # Interactive manual prediction loop.
    print("\n========== FAKE NEWS DETECTOR ==========")
    print("Enter news text or type 'exit' to quit.")

    while True:
        try:
            user_input = input("\nEnter news text: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nProgram exited.")
            break

        if user_input.lower() == "exit":
            print("Program exited!")
            break

        print(
            "\nPrediction:",
            predict_news(user_input, model, vectorizer),
        )


if __name__ == "__main__":
    main()
