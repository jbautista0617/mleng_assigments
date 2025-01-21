
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.exceptions import InconsistentVersionWarning
import sys
import re
import os
from datetime import datetime
import warnings
import mimetypes

def check_arguments():
    if len(sys.argv) != 3:
        print("\nError: Incorrect number of arguments. Please run as 'python score_headlines.py <HEADLINE_FILE> <SOURCE_NAME> \n")
        sys.exit(1)
    elif not os.path.exists(sys.argv[1]):
        print(f"\nError: {sys.argv[1]} does not exist.\n")
        sys.exit(1)
    elif mimetypes.guess_type(sys.argv[1])[0] != 'text/plain':
        print(f"\nError: {sys.argv[1]} is not a text file. Please make sure <HEADLINE_FILE> is a text file which will contain one headline per line.\n")
        sys.exit(1)

def embed_headlines(headline_list):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(headline_list)
    return embeddings

def analyze_headline_sentiment(headline_list, embeddings):
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    sentiment_model = joblib.load('svm.joblib')
    sentiments = sentiment_model.predict(embeddings)

    score_df = pd.DataFrame({
        "Sentiment": sentiments,
        "Headline": headline_list
    })

    return score_df

def extract_date(filename):
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    if match:
        date_part = match.group()
        date = date_part.replace('-', '_')
    else:
        print("No date found in file; Using today's date.")
        date = datetime.datetime.today().strftime('%Y-%m-%d')

    return date

def main():
    # Ensures arguments are correct
    check_arguments()
    filename = sys.argv[1]
    source   = sys.argv[2]

    # SentenceTransformer expects input as list of strings
    headline_list = pd.read_csv(filename, delimiter='\t', header=None)[0].tolist()
    embeddings    = embed_headlines(headline_list)
    score_df      = analyze_headline_sentiment(headline_list, embeddings)
    
    date = extract_date(filename)
    score_df.to_csv(f'headlines_scores_{source}_{date}.txt', index=False, sep='\t')
    print(f'\nHeadlines scores have successfully been saved to headlines_scores_{source}_{date}.txt \n')
    
if __name__ == "__main__":
    main()
