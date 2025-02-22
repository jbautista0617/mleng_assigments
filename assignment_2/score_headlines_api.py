# Run using "fastapi run score_headlines_api.py" or "python score_headlines_api.py"
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer
import uvicorn

# Initializing logging
logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(message)s",
    level = logging.INFO
)

# Initializing FastAPI
app = FastAPI()

# Loading the models at startup
logging.info("Loading transformer and sentiment models...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
sentiment_model = joblib.load("../assignment_1/svm.joblib")

class HeadlineRequest(BaseModel):
    """ Defines the input data model as a list of string """
    headlines: list[str]

@app.get("/status")
def status():
    """ Returns service status. """
    logging.info("Status check received")
    return {"status": "OK"}

@app.post("/score_headlines")
def score_headlines(request: HeadlineRequest):
    """ Receives a list of headlines, encodes them, and predicts sentiment labels. """

    logging.info("Processing %d headlines.", len(request.headlines))

    if len(request.headlines) == 0:
        logging.warning("No headlines received, returning empty list")
        return {"labels": []}  # Return empty list instead of an error

    # Encodes headlines and predicts sentiment
    embeddings = encoder.encode(request.headlines)
    labels = sentiment_model.predict(embeddings).tolist()

    return {"labels": labels}

if __name__ == "__main__":
    # In case of running it using Python instead of FastAPI
    uvicorn.run(app, host="localhost", port=8090)
