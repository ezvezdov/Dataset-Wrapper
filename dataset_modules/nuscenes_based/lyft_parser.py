from os import path
from os import listdir
import numpy as np
from pyquaternion import Quaternion

import dataset_modules.nuscenes_based.nuscenes_parser
import dataset_modules.nuscenes_based.nuscenes_flags as nf

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix


class LyftParser(dataset_modules.nuscenes_based.nuscenes_parser.NuScenesParser):

    def __init__(self, dataset_path: str):
        self.lyft = LyftDataset(data_path=dataset_path, json_path=path.join(dataset_path, nf.DATA),
                                verbose=True)
        self.dataset_path = dataset_path

    def get_data(self, scene_number: int, frame_number: int):
        """
        :param scene_number: Number of scene
        :param frame_number: Number of frame

        :return: Dictionary with coordinates numpy array and labels list {'dataset_type': str,
        'motion_flow_annotation': ndarray, 'coordinates' : ndarray, 'transformation_matrix': ndarray, 'labels': list
        'boxes': list}
        """

        scene = self.lyft.scene[scene_number]
        sample = self._get_nth_sample(self.lyft, scene, frame_number)

        # Points coordinates in global frame
        coord = self.get_coordinates(sample)

        # Matrix to project coordinates in 3D coordinates
        transformation_matrix = np.eye(4)

        # Motion flow annotation for points inside boxes
        motion_flow_annotation = self.get_motion_flow_annotation(self.lyft, sample, coord)

        # labels for points
        labels = []

        # Boxes list for current frame
        boxes = self.get_boxes(self.lyft, sample)

        # Type of dataset
        dataset_type = self.get_dataset_type()

        data = {'dataset_type': dataset_type, 'coordinates': coord, 'motion_flow_annotation': motion_flow_annotation,
                'transformation_matrix': transformation_matrix, 'boxes': boxes, 'labels': labels}

        return data

    def get_coordinates(self, sample: dict):
        """
        :param sample: Lyft sample
        :return coordinates numpy array coord[num][dim]
            num - number of point
            dim - dimension, {x,y,z}
        """

        available_lidars = [nf.LIDAR_TOP, nf.LIDAR_FRONT_RIGHT, nf.LIDAR_FRONT_LEFT]
        global_coordinates = []

        for lidar in available_lidars:
            sample_lidar_token = sample["data"][lidar]
            lidar_data = self.lyft.get("sample_data", sample_lidar_token)
            lidar_filepath = self.lyft.get_sample_data_path(sample_lidar_token)

            lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)

            ego_pose = self.lyft.get("ego_pose", lidar_data["ego_pose_token"])
            calibrated_sensor = self.lyft.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

            # Homogeneous transformation matrix from car frame to world frame.
            global_from_car = transform_matrix(ego_pose['translation'],
                                               Quaternion(ego_pose['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            car_from_sensor = transform_matrix(calibrated_sensor['translation'],
                                               Quaternion(calibrated_sensor['rotation']),
                                               inverse=False)

            lidar_pointcloud.transform(car_from_sensor)
            lidar_pointcloud.transform(global_from_car)

            points = np.swapaxes(lidar_pointcloud.points, 0, 1)  # change axes from points[dim][num] to points[num][dim]
            points = np.delete(points, 3, axis=1)  # cut-off intensity

            global_coordinates.append(points)

        global_coordinates = np.concatenate(global_coordinates)

        return global_coordinates

    def get_map(self):
        mask_map_list = []
        for i in self.lyft.map:
            mask_map_list.append(i['mask'])
        if len(mask_map_list) == 0:
            print("This dataset has no map!")
        return mask_map_list

    def get_dataset_type(self):
        dataset_files = listdir(self.dataset_path)
        for file in dataset_files:
            if "train" in file or "training" in file:
                return "train"
            elif "test" in file or "testing" in file:
                return "test"
            elif "valid" in file or "validation" in file:
                return "valid"
        return "unrecognized"

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
