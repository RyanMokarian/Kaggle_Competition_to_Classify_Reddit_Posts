import pickle


def write_object_to_file(obj, file_name):
    with open(file_name, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def read_object_from_file( file_name):
    with open(file_name, 'rb') as input:
        return pickle.load(input)
