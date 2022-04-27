import parser

import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

import open3d as o3

tf.enable_eager_execution()

import matplotlib.pyplot as plt
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from waymo_open_dataset.protos import segmentation_metrics_pb2
from waymo_open_dataset.protos import segmentation_submission_pb2


class WaymoParser(parser.Parser):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    # TODO delete create_open3d_pc()
    def create_open3d_pc(self, points):
        # create open3d point cloud
        pcd = o3.geometry.PointCloud()
        # assign point coordinates
        pcd.points = o3.utility.Vector3dVector(points)
        return pcd

    def __get_nth_scene(self, scene_number):
        scenes_filenames = os.listdir(self.dataset_path)
        scene_filename = os.path.join(self.dataset_path, scenes_filenames[scene_number])
        scene = tf.data.TFRecordDataset(scene_filename, compression_type='')

        return scene

    def __get_nth_frame(self, scene, frame_number):
        counter = 0
        for data in scene:
            if counter != frame_number:
                counter += 1
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            break
        return frame

    def get_data(self, scene_number: int, frame_number: int):
        scene = self.__get_nth_scene(scene_number)
        frame = self.__get_nth_frame(scene, frame_number)
        print(frame.context)

        coord = self.get_coordinates(frame)

        # TODO transformation matrix
        transformation_matrix = []

        # TODO boxes
        boxes = []

        # TODO labels
        labels = []

        data = {'coordinates': coord, 'transformation_matrix': transformation_matrix, 'boxes': boxes, 'labels': labels}

    def get_coordinates(self, frame):
        (range_images, camera_projections, _, range_image_top_pose) = (
            frame_utils.parse_range_image_and_camera_projection(frame))
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame,
                                                                           range_images,
                                                                           camera_projections,
                                                                           range_image_top_pose)
        # 3d points in vehicle frame.
        points_all = np.concatenate(points, axis=0)

        tmp_pcd = self.create_open3d_pc(points_all)
        o3.visualization.draw_geometries([tmp_pcd])
        return points_all

    def get_categories(self):
        pass
