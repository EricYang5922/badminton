import json
import os

def search(json_data, label):
    for term in json_data:
        if term['label'] == label and term['shape_type'] == 'point':
            return term['points']
    return []

def load_json(filepath):
    if not os.path.exists(filepath):
        return []
    file = open(filepath, 'r')
    json_data = json.load(file)['shapes']
    file.close()
    return reverse(search(json_data, 'shuttle'))


def reverse(points):
    result = [[point[1], point[0]] for point in points]
    return result
