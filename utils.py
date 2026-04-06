import json


def read_ingredients(file_path):
    with open(file_path, "r") as file:
        return file.read()


def save_json(data, output_path):
    with open(output_path, "w") as file:
        json.dump(data, file, separators=(",", ":"))