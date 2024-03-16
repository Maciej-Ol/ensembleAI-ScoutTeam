from PIL import Image
import requests
from PIL import Image
import io

def model_stealing(image: Image):
    SERVER_URL = "[paste server url here]"
    ENDPOINT = "/modelstealing"
    URL = SERVER_URL + ENDPOINT

    TEAM_TOKEN = "[paste your team token here]"


    response = requests.get(
        URL, files={"file": io.BufferedReader(image.tobytes())}, headers={"token": TEAM_TOKEN}
    )

    if response.status_code == 200:
        return response.content["representation"]
    else:
        raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")