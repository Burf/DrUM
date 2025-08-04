import pickle

def load_pickle(path, mode = "rb", encoding = None):
    with open(path, mode = mode, encoding = encoding) as file:
        result = pickle.loads(file.read())
    return result

def save_pickle(data, path , mode = "wb", encoding = None):
    with open(path, mode = mode, encoding = encoding) as file:
        file.write(pickle.dumps(data))
    return path