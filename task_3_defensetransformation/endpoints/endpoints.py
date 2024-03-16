# Spamming with any request or attempts to attack infrastructure will result in team ban.
# PS once we find you, ban won't be your biggest trouble ;P

# Submission examples are available here: https://github.com/WojciechBarczynski/ensembleAI_apidoc

from typing import List
from dotenv import load_dotenv


import numpy as np
import requests
import os

load_dotenv()

SERVER_URL = os.getenv("http://34.71.138.79:9090")
TEAM_TOKEN = os.getenv("l5pvMfL4ZID1QHmn")


# Be careful. This can be done only once an hour.
# Computing this might take a few minutes. Be patient.
# Make sure your file has proper content.
def defense_submit(path_to_npz_file: str):
    endpoint = "/defense/submit"
    url = SERVER_URL + endpoint
    with open(path_to_npz_file, "rb") as f:
        response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN})
        if response.status_code == 200:
            print("Request ok")
            print(response.json())
        else:
            raise Exception(
                f"Defense submit failed. Code: {response.status_code}, content: {response.json()}"
            )

# Call examples:

# sybil(
#     [101031, 8526, 43127, 191394, 298792, 121086, 149475, 102605, 163605, 101855],
#     "home",
#     "affine",
# )
# sybil_submit("example_submissions/sybil.npz", "affine")
# sybil_reset("binary")
# defense_submit("example_submissions/defense.npz")
# model_stealing("test.png")
# model_stealing_submit("example_submissions/model_stealing.onnx")
# model_stealing_reset()
