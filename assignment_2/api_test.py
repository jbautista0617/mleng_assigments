# Run as "python api_test.py" after running "fastapi run score_headlines_.py --port 8090"
import requests

HOST = 'localhost'
PORT = 8090

def test_status():
    """ Tests the /status API call """
    response = requests.get(url=f'http://{HOST}:{PORT}/status', timeout=5)
    print(f"Result of doing a GET request from http://{HOST}:{PORT}/status:\n")
    print(response.text)

def test_score_headlines():
    """ Tests the /test_score_headlines API call in two different ways """

    test_headlines = {"headlines":["This is a test headline", "This headlines is amazing!!"]}
    no_headlines = {"headlines":[]}

    response_test = requests.post(url=f'http://{HOST}:{PORT}/score_headlines',
                                  json=test_headlines, timeout=5)

    print(f"Result of doing a POST request to http://{HOST}:{PORT}/score_headlines\n")

    print(f"Sending: {test_headlines}")
    print(f"Received: {response_test.text}\n")

    response_empty = requests.post(url=f'http://{HOST}:{PORT}/score_headlines',
                                   json=no_headlines, timeout=5)

    print(f"Sending: {no_headlines}")
    print(f"Received: {response_empty.text}")

def main():
    """ Runs tests on all possible API calls """
    test_status()
    print("\n-------------\n")
    test_score_headlines()

if __name__=='__main__':
    main()
