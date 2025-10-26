import pickle


def save_model(model, path: str):
    with open(path + ".pkl", "wb") as f:
        pickle.dump(model, f)
