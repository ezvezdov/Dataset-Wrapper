from os import path

import nuscenes_based.nuscenes_parser

import numpy as np
from lyft_dataset_sdk.lyftdataset import LyftDataset
# from lyft_dataset_sdk.lyftdataset import LyftDatasetExplorer
# from lyft_dataset_sdk.utils import data_classes
from pyquaternion import Quaternion
import numpy as np

import nuscenes_based.nuscenes_flags as nf


class LyftParser(nuscenes_based.nuscenes_parser.NuScenesParser):

    def __init__(self, dataset_path: str):
        self.lyft = LyftDataset(data_path=dataset_path, json_path=path.join(dataset_path, 'data'),
                                verbose=True)
        self.dataset_path = dataset_path

    def get_data(self, scene_number: int, frame_number: int):
        """
        :param scene_number: Number of scene
        :param frame_number: Number of frame

        :return: Dictionary with coordinates numpy array and labels list
                {'coordinates' : numpy array, 'labels' : labels list}
        """

        scene = self.lyft.scene[scene_number]
        sample = self._get_nth_sample(self.lyft, scene, frame_number)
        coord = self.get_coordinates(sample)
        boxes = self.get_boxes(self.lyft, sample)
        transformation_matrix = self.get_transformation_matrix(self.lyft, sample)
        data = {'coordinates': coord, 'transformation_matrix': transformation_matrix, 'boxes': boxes, 'labels': ''}

        return data

    def get_coordinates(self, sample: dict):
        """
        :param sample: Nuscenes sample
        :return coordinates numpy array coord[dim][num]
            dim - dimension, {x,y,z}
            num - number of point
        """

        lidar_top_data = self.lyft.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_filename = path.join(self.dataset_path, lidar_top_data['filename'])

        scan = np.fromfile(str(lidar_filename), dtype=np.float32)
        points = scan.reshape((-1, 5))[:, : 4]  # 4 is number of dimensions
        points = points[:3, :]  # cut-off intensity
        return points

    def get_categories(self):
        """
        Returns categories of dataset objects
        :return: categories list, category consist of name, description
                [{'name': str, 'description': str},{}]
        """
        categories = self.lyft.category
        for category in categories:
            del category[nf.TOKEN]

        return categories

    