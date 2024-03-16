import requests
from typing import List
import json

def sybil_attack(ids: List[int], home_or_defense: str, binary_or_affine: str):
    if home_or_defense not in ["home", "defense"] or binary_or_affine not in ["binary", "affine"]:
        raise "Invalid endpoint"
    SERVER_URL = "http://34.71.138.79:9090"
    ENDPOINT = f"/sybil/{binary_or_affine}/{home_or_defense}"
    URL = SERVER_URL + ENDPOINT
    
    TEAM_TOKEN = "l5pvMfL4ZID1QHmn"
    ids = ids = ",".join(map(str, ids))

    response = requests.get(
        URL, params={"ids": ids}, headers={"token": TEAM_TOKEN}
    )

    if response.status_code == 200:
        return json.loads(response.content.decode())["representations"]
    else:
        raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")


api_ret = sybil_attack(ids=[101031],
                 home_or_defense='home',
                 binary_or_affine='affine')
print(len(api_ret[0]))