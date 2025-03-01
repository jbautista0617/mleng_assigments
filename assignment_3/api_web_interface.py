# NOTE: Must run "score_headlines_api.py" before this
# Run this using "streamlit run api_web_interface.py --server.port 9090"
import io
import requests
import pandas as pd
import streamlit as st

API_URL = "http://localhost:8090/score_headlines"

def analyze_headlines(headlines):
    """ Sends headlines to API for sentiment analysis. """
    try:
        response = requests.post(API_URL, json = {"headlines": headlines}, timeout = 10)
        response.raise_for_status()
        return response.json().get("labels", [])

    except requests.exceptions.ConnectionError:
        st.error("Error: Could not connect to the API. Ensure the server is running.")

    # Returns "Unknown" sentiment for every headline if API fails
    return ["Unknown"] * len(headlines)

def generate_csv(headlines, sentiment_labels):
    """ Generates csv content for download. """
    csv_data = pd.DataFrame({"Headline": headlines, "Sentiment": sentiment_labels})
    csv_buffer = io.StringIO()
    csv_data.to_csv(csv_buffer, index = False)
    return csv_buffer.getvalue()

# Ensure session state variables exist at the start
if "headlines" not in st.session_state:
    st.session_state.headlines = []
if "submitted" not in st.session_state:
    st.session_state.submitted = False

st.title("Headline Sentiment Analyzer")

# Show input screen if headlines haven't been submitted
if not st.session_state.submitted:
    st.subheader("Enter your headlines:")
    input_method = st.radio("Choose input method:", ["Paste text", "Upload .txt file"])

    # Option 1: Paste multiple headlines in a textbox
    if input_method == "Paste text":
        text_input = st.text_area("Paste headlines (separate by new lines):")
        if text_input.strip():
            st.session_state.headlines = [
                line.strip() for line in text_input.split("\n") if line.strip()
            ]

    # Option 2: Upload a .txt file
    elif input_method == "Upload .txt file":
        uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
        if uploaded_file:
            file_content = uploaded_file.read().decode("utf-8")
            st.session_state.headlines = [
                line.strip() for line in file_content.split("\n") if line.strip()
            ]

    # Button to move to next screen
    if st.button("Next"):
        if st.session_state.headlines:
            st.session_state.submitted = True
            st.rerun()
        else:
            st.warning("Please enter or upload at least one headline before submitting.")

# Once headlines are submitted, show analysis screen
else:
    st.subheader("If necessary, edit your headlines before final submission:")

    for i, headline in enumerate(st.session_state.headlines):
        st.session_state.headlines[i] = st.text_input(
            f"Headline {i+1}", headline, key = f"headline_{i}"
        )

    if st.button("Analyze Headlines"):
        if not st.session_state.headlines:
            st.warning("No headlines to analyze.")
        else:
            sentiments = analyze_headlines(st.session_state.headlines)
            st.subheader("Results:")
            for headline, sentiment in zip(st.session_state.headlines, sentiments):
                st.write(f"**{headline}** - {sentiment}")

            # Generate csv for download
            csv_content = generate_csv(st.session_state.headlines, sentiments)
            st.download_button(
                "Download Results as CSV",
                data = csv_content,
                file_name = "headline_sentiments.csv",
                mime = "text/csv"
            )

    # Start over button (to reset and allow new submission)
    if st.button("Start Over"):
        st.session_state.submitted = False
        st.session_state.headlines = []
        st.rerun()
