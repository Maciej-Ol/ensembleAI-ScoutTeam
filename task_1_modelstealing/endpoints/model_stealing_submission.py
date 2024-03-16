import requests

def model_stealing_submission(path_to_onnx_file: str):
    SERVER_URL = "http://34.71.138.79:9090"
    ENDPOINT = "/modelstealing/submit"
    URL = SERVER_URL + ENDPOINT

    TEAM_TOKEN = "l5pvMfL4ZID1QHmn"

    with open(path_to_onnx_file, "rb") as onnx_file:
        response = requests.post(
            URL, files={"file": onnx_file}, headers={"token": TEAM_TOKEN}
        )

        if response.status_code == 200:
            return response.content["score"]
        else:
            raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")