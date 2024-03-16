import numpy as np
import requests
import os
import sys



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
        else:
            raise Exception(
                f"Defense submit failed. Code: {response.status_code}, content: {response.json()}"
            )
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
    data_sub = np.load("data/DefenseTransformationSubmit.npz")
    rep = apply_defense_transformation_3(data_sub["representations"],st=st)

    # Convert from float64 to float32                                
    rep = rep.astype(np.float32)
                                    
    # Apply defense transformation (e.g., your transformation)
    np.savez(
        "data/example_submission.npz",
        representations=rep
    )

def save_response(response):
    # Append the parameter to a text file
    output_file = "data/output.txt"  # Name of the output text file
    with open(output_file, mode='a') as file:
        file.write(response + "\n")
        file.write("Step: "+a + "\n")




if __name__ == "__main__":
    # Access command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python your_script.py <parameter>")
        sys.exit(1)

    # Extract the parameter
    parameter = sys.argv[1]

    # list
    list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    a = list[int(parameter)]

    save_file(a)
    respone = defense_submit("data/example_submission.npz")
    save_response(respone, a)

    # Now you can use the parameter in your script as needed
    print("Parameter passed from Bash script:", parameter)