import json
import os
import numpy as np

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

def get_point_mask(pcl, bbox, x_add=(0., 0.), y_add=(0., 0.), z_add=(0., 0.)):
    '''
    :param pcl: x,y,z ...
    :param bbox: x,y,z,l,w,h,yaw
    :param x_add:
    :param y_add:
    :param z_add:
    :return: Segmentation mask
    '''

    angle = bbox[6]
    Rot_z = np.array(([np.cos(angle), - np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]))
    s = pcl.copy()
    s[:, :3] -= bbox[:3]
    s[:, :3] = (s[:, :3] @ Rot_z)[:, :3]
    size = np.array((-bbox[3]/2, bbox[3]/2, -bbox[4]/2, bbox[4]/2, -bbox[5]/2, bbox[5]/2))
    point_mask = (size[0] - x_add[0] <= s[:, 0]) & (s[:, 0] <= size[1] + x_add[1]) & (size[2] - y_add[0] <= s[:, 1])\
                 & (s[:, 1] <= size[3] + y_add[1]) & (size[4] - z_add[0] <= s[:,2]) & (s[:,2] <= size[5] + z_add[1])

    return point_mask


