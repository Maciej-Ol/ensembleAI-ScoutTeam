import numpy as np
import requests
import time

SERVER_URL = "http://34.71.138.79:9090"
TEAM_TOKEN = "l5pvMfL4ZID1QHmn"

def defense_submit(path_to_npz_file: str):
    endpoint = "/defense/submit"
    url = SERVER_URL + endpoint
    with open(path_to_npz_file, "rb") as f:
        response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN})
        if response.status_code == 200:
            print("Request ok")
            print(response.json())
            answer = f"Request ok, response: {response.json()}"
        else:
            answer = f"Defense submit failed. Code: {response.status_code}, content: {response.json()}"
    return response.json()


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



def save_file(st):
    data_sub = np.load("task_3_defensetransformation/data/DefenseTransformationSubmit.npz")
    rep = apply_defense_transformation_3(data_sub["representations"],st=st)

    # Convert from float64 to float32                                
    rep = rep.astype(np.float32)
                                    
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
        file.write("Step: "+i + "\n")




if __name__ == "__main__":
    list = [0.1, 0.2, 0.05, 0.15, 0.3, 0.4, 0.5]

    for i in list:
        save_file(i)
        respone = defense_submit("task_3_defensetransformation/data/example_submission.npz")
        save_response(respone, i)
        print("I have finished defense_submit successfully!")
        time.sleep(3660)  # Sleep for 4000 seconds (about 1 hour)



    