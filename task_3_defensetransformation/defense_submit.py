import numpy as np
import requests
import time

SERVER_URL = "http://34.71.138.79:9090"
TEAM_TOKEN = "l5pvMfL4ZID1QHmn"

def defense_submit(path_to_npz_file: str):
    endpoint = "/defense/submit"
    url = SERVER_URL + endpoint
    with open(path_to_npz_file, "rb") as f:
        try:
            response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN})
            if response.status_code == 200:
                print("Request ok")
                print(response.json())
                answer = f"Request ok, response: {response.json()}"
            else:
                answer = f"Defense submit failed. Code: {response.status_code}, content: {response.json()}"
        except Exception as e:
            answer = f"Server down. Exception: {e}"
    return answer


def apply_defense_transformation_3(X,padding_size=100,st=0.3):
    n = X.shape[1]

    matrix = np.random.rand(n,n)
    matrix = matrix/np.linalg.norm(matrix)*st+np.eye(n)

    X = np.dot(X, matrix)
    # add padding
    padding = np.ones((X.shape[0], padding_size))
    padding = np.random.normal(np.mean(X), np.std(X), X.shape[0]*padding_size).reshape(X.shape[0], padding_size)
    X = np.hstack((X, padding))
    return X

def transform_file(st):
    data_sub = np.load("task_3_defensetransformation/data/DefenseTransformationSubmit.npz")
    rep = apply_defense_transformation_3(data_sub["representations"],st=st)

    # Convert from float64 to float32                                
    rep = rep.astype(np.float32)
    return rep

def save_file(rep):           
    # Apply defense transformation (e.g., your transformation)
    np.savez(
        "task_3_defensetransformation/data/example_submission.npz",
        representations=rep
    )

def save_response(response, i):
    # Append the parameter to a text file
    output_file = "task_3_defensetransformation/data/output.txt"  # Name of the output text file
    with open(output_file, mode='a') as file:
        file.write(response + "\n")
        file.write(f"Step: {i}" + "\n")




if __name__ == "__main__":
    list = [0.1, 0.2, 0.05, 0.15, 0.3, 0.4, 0.5]

    for i in list:
        rep = transform_file(i)
        save_file(rep)
        respone = defense_submit("task_3_defensetransformation/data/example_submission.npz")
        print("I have finished defense_submit successfully!")
        save_response(respone, i)
        print("Response saved!")
        time.sleep(3660)  # Sleep for about 1 hour



    