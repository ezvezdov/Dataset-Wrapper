import os
import json
import numpy as np

import parser
from dataset_modules.utils import get_unificated_category_id, get_point_mask

import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import keypoint_data
from waymo_open_dataset.utils import transform_utils

# waymo-open-dataset-tf-2-6-0
# tensorflow==2.6.0
# keras==2.6.0

# List of Waymo categories, id = index
categories_list = ["UNDEFINED", "CAR", "TRUCK", "BUS", "OTHER_VEHICLE", "MOTORCYCLIST", "BICYCLIST", "PEDESTRIAN",
                   "SIGN", "TRAFFIC_LIGHT", "POLE", "CONSTRUCTION_CONE", "BICYCLE", "MOTORCYCLE", "BUILDING",
                   "VEGETATION", "TREE_TRUNK", "CURB", "ROAD", "LANE_MARKER", "OTHER_GROUND", "WALKABLE", "SIDEWALK"]

bboxes_categories_list = ["TYPE_UNSET", "TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST", "TYPE_OTHER"]


class WaymoParser(parser.Parser):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

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
                continue
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            break
        return frame

    def get_data(self, scene_number: int, frame_number: int):
        scene = self.__get_nth_scene(scene_number)
        frame = self.__get_nth_frame(scene, frame_number)
        prev_frame = self.__get_nth_frame(scene, frame_number - 1) if frame_number - 1 >= 0 else None

        (range_images, camera_projections, segmentation_labels,
         range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

        coord = self.get_coordinates(frame, range_images, camera_projections, range_image_top_pose)

        # transformation matrix for global (vehicle) view
        transformation_matrix = cur_ego_to_global = np.reshape(np.array(frame.pose.transform), [4, 4])

        boxes = self.get_boxes(frame)

        motion_flow_annotation = self.get_motion_flow_annotation(frame, prev_frame, coord)

        labels = self.get_labels(frame, range_images, segmentation_labels)

        dataset_type = self.get_dataset_type()

        data = {'dataset_type': dataset_type, 'motion_flow_annotation': motion_flow_annotation, 'coordinates': coord,
                'transformation_matrix': transformation_matrix,
                'boxes': boxes, 'labels': labels}
        return data

    def convert_range_image_to_point_cloud_labels(self, frame, range_images, segmentation_labels, ri_index=0):
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
                sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
            else:
                num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
                sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

            point_labels.append(sl_points_tensor.numpy())
        return point_labels

    def get_coordinates(self, frame, range_images, camera_projections, range_image_top_pose):
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections,
                                                                           range_image_top_pose)
        # 3d points in vehicle frame.
        points_all = np.concatenate(points, axis=0)

        return points_all

    def get_motion_flow_annotation(self, cur_frame, prev_frame, coordinates):
        # https://deepai.org/publication/scalable-scene-flow-from-point-clouds-in-the-real-world
        # https://arxiv.org/pdf/2103.01306v3.pdf
        # chapter 3.2

        motion_flow_annotation = np.full(coordinates.shape[0], None)

        if not prev_frame:
            return motion_flow_annotation

        # 10 Hz, ~0.1 s
        time_delta = cur_frame.timestamp_micros - prev_frame.timestamp_micros  # value in microseconds
        time_delta /= 1000000  # time in seconds

        cur_labels = keypoint_data.group_object_labels(cur_frame)
        prev_labels = keypoint_data.group_object_labels(prev_frame)

        for cur_box_id in cur_labels.keys():
            cur_box = cur_labels[cur_box_id]
            prev_box = None
            for prev_box_id in prev_labels.keys():
                if cur_box_id == prev_box_id:
                    prev_box = prev_labels[prev_box_id]

            if not prev_box:
                continue

            cur_center = [cur_box.laser.box.center_x, cur_box.laser.box.center_y, cur_box.laser.box.center_z]
            cur_size = [cur_box.laser.box.width, cur_box.laser.box.length, cur_box.laser.box.height]
            cur_yaw = cur_box.laser.box.heading
            prev_yaw = prev_box.laser.box.heading

            # TESTED np.linalg.inv(cur_box_to_cur_ego) @ np.array(cur_center) = [0,0,0]
            cur_box_to_cur_ego = np.eye(4)
            cur_box_to_cur_ego[:3, :3] = transform_utils.get_yaw_rotation(cur_yaw)
            cur_box_to_cur_ego[:3, 3] = cur_center

            # TESTED np.linalg.inv(prev_box_to_prev_ego) @ np.array(prev_center)
            prev_box_to_prev_ego = np.eye(4)
            prev_box_to_prev_ego[:3, :3] = transform_utils.get_yaw_rotation(prev_yaw)
            prev_box_to_prev_ego[:3, 3] = [prev_box.laser.box.center_x, prev_box.laser.box.center_y,
                                           prev_box.laser.box.center_z]

            cur_ego_to_global = np.reshape(np.array(cur_frame.pose.transform), [4, 4])
            prev_ego_to_global = np.reshape(np.array(prev_frame.pose.transform), [4, 4])

            cur_ego_to_prev_ego = np.linalg.inv(prev_ego_to_global) @ cur_ego_to_global

            prev_box_to_cur_ego = np.linalg.inv(cur_ego_to_prev_ego) @ prev_box_to_prev_ego

            cur_box_to_prev_box = np.linalg.inv(prev_box_to_cur_ego) @ cur_box_to_cur_ego

            # adding 1 to end of each point end
            coordinates_reshape = np.concatenate((coordinates, np.ones((coordinates.shape[0], 1))), axis=1)

            point_mask = get_point_mask(coordinates,
                                        [cur_center[0], cur_center[1], cur_center[2], cur_size[0], cur_size[1],
                                         cur_size[2], cur_yaw])

            true_cnt = 0
            for point_index in range(len(coordinates)):
                if point_mask[point_index]:
                    pm1 = cur_box_to_prev_box @ np.transpose(coordinates_reshape[point_index])
                    pm1 = np.delete(pm1, 3)
                    motion_flow_annotation[point_index] = coordinates[point_index] - pm1
                    motion_flow_annotation[point_index] /= time_delta
                    true_cnt += 1

        return motion_flow_annotation

    def get_boxes(self, frame):
        labels = keypoint_data.group_object_labels(frame)

        boxes_list = []
        for box_id in labels.keys():
            obj = labels[box_id]
            box_inf = dict()
            box_inf['category_id'] = get_unificated_category_id(bboxes_categories_list[obj.laser.object_type])
            box_inf['size'] = [obj.laser.box.length, obj.laser.box.width, obj.laser.box.height]
            box_inf['center'] = [obj.laser.box.center_x, obj.laser.box.center_y, obj.laser.box.center_z]
            box_inf['orientation'] = obj.laser.box.heading
            boxes_list.append(box_inf)

        return boxes_list

    def get_labels(self, frame, range_images, segmentation_labels):
        if len(segmentation_labels) == 0:
            return []
        point_labels = self.convert_range_image_to_point_cloud_labels(frame, range_images, segmentation_labels)
        point_labels_all = np.concatenate(point_labels, axis=0)

        labels = []
        for label in point_labels_all:
            if label[0] == 0 and label[1] == 0:
                labels.append(get_unificated_category_id(categories_list[0]))
            else:
                labels.append(get_unificated_category_id(categories_list[label[1]]))
        return labels

    def get_dataset_type(self):
        folder_name = self.dataset_path.split()[-1]
        if "training" in folder_name or "train" in folder_name:
            return "train"
        elif "testing" in folder_name or "test" in folder_name:
            return "test"
        elif "validation" in folder_name or "valid" in folder_name:
            return "valid"
        else:
            return "unrecognized"

    def get_categories(self):
        with open(os.path.join(os.getcwd(), "dataset_modules", "waymo_module", "categories.json"), 'r') as f:
            categories = json.load(f)
        return categories
