import numpy as np

if __name__ == "__main__":
    data = np.load(
        "task_3_defensetransformation/data/ExampleDefenseTransformationEvaluate.npz"
    )
    print(data["labels"], data["representations"].shape)

    data = np.load("task_3_defensetransformation/data/ExampleDefenseTransformationSubmit.npz")
    print(data["representations"].shape)
