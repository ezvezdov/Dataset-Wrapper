import parser
import json
import pprint
from os import path
import numpy as np
import numpy.linalg as lag
import glob


class A2D2Parser(parser.Parser):

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        with open(path.join(self.dataset_path, 'cams_lidars.json'), 'r') as f:
            self.config = json.load(f)

    def get_coordinates(self):
        cam_lid_sb_path = path.join(self.dataset_path, 'camera_lidar_semantic_bboxes')
        file_names = sorted(glob.glob(path.join(cam_lid_sb_path, '*/lidar/cam_front_center/*.npz')))
        file_name_lidar = file_names[7]
        lidar_front_center = np.load(file_name_lidar)
        points = lidar_front_center['points']

        return points

    def get_data(self, scene_number: int, frame_number: int):
        coord = self.get_coordinates()
        categories = self.get_categories()


    def get_categories(self):
        cam_lid_sb_path = path.join(self.dataset_path, 'camera_lidar_semantic_bboxes')
        with open(path.join(cam_lid_sb_path, 'class_list.json'), 'r') as f:
            class_dict = json.load(f)
        print(class_dict)

        return class_dict
