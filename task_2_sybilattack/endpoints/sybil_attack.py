def sybil_attack(ids: List[int], home_or_defense: str, binary_or_affine: str):
    if home_or_defense not in ["home", "defense"] or binary_or_affine not in ["binary", "affine"]:
        raise "Invalid endpoint"
    
    SERVER_URL = "[paste server url here]"
    ENDPOINT = f"/sybil/{binary_or_affine}/{home_or_defense}"
    URL = SERVER_URL + ENDPOINT
    
    TEAM_TOKEN = "[paste your team token here]"
    ids = ids = ",".join(map(str, ids))

    response = requests.get(
        URL, params={"ids": ids}, headers={"token": TEAM_TOKEN}
    )

    if response.status_code == 200:
        return response.content["representations"]
    else:
        raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")
