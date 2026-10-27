# Fake News Detection System

## Project Overview
This project uses Machine Learning and Natural Language Processing (NLP) to classify news articles as Fake News or Real News.

The system uses TF-IDF for text feature extraction and Logistic Regression for classification. It also includes an option to fetch live news headlines using NewsAPI.

## Technologies Used
- Python
- Pandas
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression
- NewsAPI

## Model Performance
The current model achieved the following results on the test dataset:

- **Accuracy:** 98.69%
- **Fake News F1-score:** 0.99
- **Real News F1-score:** 0.99

These are preliminary results. Further evaluation is needed to check for duplicate articles, data leakage, and generalization to unseen news sources.

## Project Structure

```text
Fake_News_Detection/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Place the required datasets in the `data/` directory.
5. Configure your NewsAPI key in a local `.env` file:

```env
NEWS_API_KEY=your_api_key_here
```

6. Run the project:

```bash
python main.py
```

## Dataset
The project uses separate CSV files for fake and real news articles. The CSV files should contain at least `title` and `text` columns.

The datasets are not included in this repository. Obtain them from an authorized source and check their licence before use.

## Limitations
- Predictions are based on patterns learned from the training dataset.
- A prediction does not independently verify whether a news article is true or false.
- Live news fetching depends on NewsAPI access and account restrictions.
- The reported performance may differ on news from unseen sources.

## Future Improvements
- Evaluate performance on additional independent datasets.
- Check for duplicate articles and data leakage.
- Improve error handling and model reuse.
- Add a user-friendly interface.

## Author
Developed as an MSc Computer Science project.
