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

import json

# List of Waymo categories, id = index
categories_list = ["UNDEFINED", "CAR", "TRUCK", "BUS", "OTHER_VEHICLE", "MOTORCYCLIST", "BICYCLIST", "PEDESTRIAN",
                   "SIGN", "TRAFFIC_LIGHT", "POLE", "CONSTRUCTION_CONE", "BICYCLE", "MOTORCYCLE", "BUILDING",
                   "VEGETATION", "TREE_TRUNK", "CURB", "ROAD", "LANE_MARKER", "OTHER_GROUND", "WALKABLE", "SIDEWALK"]
#
# categories_dict = {"UNDEFINED": "Undefined point", "CAR": "-", "TRUCK": "-", "OTHER_VEHICLE": "Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction vehicles, RV, limo, tram).",
#                    "MOTORCYCLIST": "-", "BICYCLIST": "-",
#                    "PEDESTRIAN": "-", "SIGN": "-", "TRAFFIC_LIGHT": "-", "POLE": "Lamp post, traffic sign pole etc.",
#                    "CONSTRUCTION_CONE": "Construction cone/pole.", "BICYCLE": "-", "MOTORCYCLE": "-",
#                    "BUILDING": "-", "VEGETATION": "Bushes, tree branches, tall grasses, flowers etc.", "TREE_TRUNK": "-",
#                    "CURB": "Curb on the edge of roads. This does not include road boundaries if there’s no curb.", "ROAD": "Surface a vehicle could drive on. This include the driveway connecting parking lot and road over a section of sidewalk.", "LANE_MARKER": "Marking on the road that’s specifically for defining lanes such as single/double white/yellow lines.", "OTHER_GROUND": "Marking on the road other than lane markers, bumps, cateyes, railtracks etc.",
#                    "WALKABLE": "Most horizontal surface that’s not drivable, e.g. grassy hill, pedestrian walkway stairs etc.", "SIDEWALK": "Nicely paved walkable surface when pedestrians most likely to walk on."
#                    }


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
            # TODO remove this construction and uncomment break
            if frame.lasers[0].ri_return1.segmentation_label_compressed:
                break
            # break
        return frame

    def get_data(self, scene_number: int, frame_number: int):
        scene = self.__get_nth_scene(scene_number)
        frame = self.__get_nth_frame(scene, frame_number)
        # print(frame.context)

        (range_images, camera_projections, segmentation_labels,
         range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

        coord = self.get_coordinates(frame)

        # TODO transformation matrix
        transformation_matrix = []

        # TODO boxes
        boxes = []

        # TODO labels
        labels = []
        print("KEYS ", segmentation_labels.keys())
        if len(segmentation_labels) != 0:
            print("Segmentation label was detected!")
            point_labels = self.convert_range_image_to_point_cloud_labels(
                frame, range_images, segmentation_labels)
            # np.set_printoptions(threshold=np.inf)
            # print(point_labels)
            point_labels_all = np.concatenate(point_labels, axis=0)
            print(point_labels_all)

            # tmp_pcd = self.create_open3d_pc(point_labels_all)
            # o3.visualization.draw_geometries([tmp_pcd])

        data = {'coordinates': coord, 'transformation_matrix': transformation_matrix, 'boxes': boxes, 'labels': labels}

    def convert_range_image_to_point_cloud_labels(self, frame,
                                                  range_images,
                                                  segmentation_labels,
                                                  ri_index=0):
        """Convert segmentation labels from range images to point clouds.

        Args:
          frame: open dataset frame
          range_images: A dict of {laser_name, [range_image_first_return,
             range_image_second_return]}.
          segmentation_labels: A dict of {laser_name, [range_image_first_return,
             range_image_second_return]}.
          ri_index: 0 for the first return, 1 for the second return.

        Returns:
          point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
            points that are not labeled.
        """
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        point_labels = []
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            range_image_mask = range_image_tensor[..., 0] > 0

            if c.name in segmentation_labels:
                sl = segmentation_labels[c.name][ri_index]
                sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
                print(sl_tensor)
                sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
            else:
                num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
                sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

            point_labels.append(sl_points_tensor.numpy())
        return point_labels

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
        with open(os.path.join(os.getcwd(), "waymo_module", "categories.json"), 'r') as f:
            categories = json.load(f)
        return categories
