#test comment
# Standard library
import sys
import re
import os
from datetime import datetime
import warnings
import mimetypes

# Third-party
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.exceptions import InconsistentVersionWarning

def check_arguments():
    """Ensures command-line arguments are correct.

    Raises:
        Error: If incorrect number of arguments is used; 
            if the file does not exist; 
            if the file is not a text file.
    """
    if len(sys.argv) != 3:
        print(
            "\nError: Incorrect number of arguments. Please run as "
            "'python score_headlines.py <HEADLINE_FILE> <SOURCE_NAME>' \n"
        )
        sys.exit(1)

    elif not os.path.exists(sys.argv[1]):
        print(f"\nError: {sys.argv[1]} does not exist.\n")
        sys.exit(1)

    elif mimetypes.guess_type(sys.argv[1])[0] != 'text/plain':
        print(
            f"\nError: {sys.argv[1]} is not a text file. Please make sure "
            "the <HEADLINE_FILE> is a text file containing one headline per line.\n"
        )
        sys.exit(1)

def encode_headlines(headline_list):
    """Encodes headlines using the all-MiniLM-L6-v sentence transformer model.

    Args:
        headline_list: List of scraped headlines.

    Returns:
        embeddings: Array of encoded headlines.
    """
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = encoder.encode(headline_list)
    return embeddings

def analyze_headline_sentiment(headline_list, embeddings):
    """Predicts the general sentiment for a list of encoded headlines.

    Args:
        headline_list: List of scraped headlines.
        embeddings: Array of encoded headlines.

    Returns:
        score_df: A Pandas dataframe containing the sentiment in the first column
                  and its corresponding headline in the second.
    """
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    sentiment_model = joblib.load('svm.joblib')
    sentiments = sentiment_model.predict(embeddings)

    score_df = pd.DataFrame({
        "Sentiment": sentiments,
        "Headline": headline_list
    })

    return score_df

def extract_date(filename):
    """Extracts the date from the name of the file.

    Args:
        filename: File submitted as an argument when .py file is ran. 

    Returns:
        date: The extracted date as YYYY_MM_DD.

    Raises:
        Error: If no date is found in the filename, today's date is returned instead.
    """
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    if match:
        date_part = match.group()
        date = date_part.replace('-', '_')
    else:
        print("Error: No date found in file; Using today's date instead.")
        date = datetime.datetime.today().strftime('%Y-%m-%d')

    return date

def main():
    """Main function to handle the pipeline for scoring headlines."""
    check_arguments()
    filename = sys.argv[1]
    source   = sys.argv[2]

    # SentenceTransformer expects input as list of strings
    headline_list = pd.read_csv(filename, delimiter='\t', header=None)[0].tolist()

    embeddings    = encode_headlines(headline_list)
    score_df      = analyze_headline_sentiment(headline_list, embeddings)

    date = extract_date(filename)
    output_file = f'headlines_scores_{source}_{date}.txt'
    score_df.to_csv(output_file, index=False, sep=',')
    print(f'\nHeadlines scores have successfully been saved to {output_file} \n')

if __name__ == "__main__":
    main()
