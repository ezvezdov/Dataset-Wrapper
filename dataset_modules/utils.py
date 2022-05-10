import json
import os

file_path = os.path.join(os.getcwd(), "resources", "categories-category2id.json")
file = open(file_path, "r")
categories = dict()
categories = json.load(file)

def unificate_category_list():
    pass


def get_unificated_category_id(category: str):
    """
    Transform category to unificated category

    :param category: string of category
    :return: unificated category id
    """
    if category in categories.keys():
        return categories[category]
    return None

