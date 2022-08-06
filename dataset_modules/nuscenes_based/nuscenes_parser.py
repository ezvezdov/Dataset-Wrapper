from os import path
from pyquaternion import Quaternion
import numpy as np

import parser
import dataset_modules.nuscenes_based.nuscenes_flags as nf
from dataset_modules.utils import get_unificated_category_id, get_point_mask

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import data_classes
from nuscenes.utils import splits as nusc_dataset_type
from nuscenes.utils.geometry_utils import transform_matrix


class NuScenesParser(parser.Parser):

    def __init__(self, dataset_path: str):
        self.nusc = NuScenes(dataroot=dataset_path, verbose=True)
        self.dataset_path = dataset_path

    def get_map(self):
        mask_map_list = []
        for i in self.nusc.map:
            mask_map_list.append(i['mask'])
        if len(mask_map_list) == 0:
            print("This dataset has no map!")
        return mask_map_list

    def _get_nth_sample(self, dataset_module, scene: dict, frame_number: int):
        sample = dataset_module.get(nf.SAMPLE, scene[nf.FIRST_SAMPLE_TOKEN])
        for i in range(frame_number):
            sample = dataset_module.get(nf.SAMPLE, sample[nf.NEXT])
        return sample

    def get_data(self, scene_number: int, frame_number: int):
        """
        :param scene_number: Number of scene
        :param frame_number: Number of frame

        :return: Dictionary with coordinates numpy array and labels list {'dataset_type': str,
        'motion_flow_annotation': ndarray, 'coordinates' : ndarray, 'transformation_matrix': ndarray, 'labels': list
        'boxes': list}
        """
        scene = self.nusc.scene[scene_number]
        sample = self._get_nth_sample(self.nusc, scene, frame_number)

        # Points coordinates in global frame
        coord = self.get_coordinates(sample)

        # Matrix to project coordinates in 3D coordinates
        transformation_matrix = np.eye(4)

        # Motion flow annotation for points inside boxes
        motion_flow_annotation = self.get_motion_flow_annotation(self.nusc, sample, coord)

        # labels for points
        labels = self.get_label_list(sample)

        # Boxes list for current frame
        boxes = self.get_boxes(self.nusc, sample)

        # Type of dataset
        dataset_type = self.get_dataset_type(scene['name'])

        data = {'dataset_type': dataset_type, 'motion_flow_annotation': motion_flow_annotation, 'coordinates': coord,
                'transformation_matrix': transformation_matrix,
                'boxes': boxes, 'labels': labels}

        return data

    def get_coordinates(self, sample: dict):
        """
        :param sample: Nuscenes sample
        :return coordinates numpy array coord[num][dim]
            num - number of point
            dim - dimension, {x,y,z}
        """

        lidar_top_data = self.nusc.get(nf.SAMPLE_DATA, sample[nf.DATA][nf.LIDAR_TOP])
        pcd = data_classes.LidarPointCloud.from_file(path.join(self.dataset_path, lidar_top_data[nf.FILENAME]))

        # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/scripts/export_pointclouds_as_obj.py

        # transform the point cloud to the ego vehicle frame
        cs_record = self.nusc.get(nf.CALIBRATED_SENSOR, lidar_top_data[nf.CALIBRATED_SENSOR_TOKEN])
        pcd.transform(transform_matrix(cs_record['translation'], Quaternion(cs_record[nf.ROTATION])))

        # transform from car ego to the global frame
        ego_pose = self.nusc.get(nf.EGO_POSE, lidar_top_data[nf.EGO_POSE_TOKEN])
        pcd.transform(transform_matrix(ego_pose['translation'], Quaternion(ego_pose[nf.ROTATION])))

        points = np.swapaxes(pcd.points, 0, 1)  # change axes from points[dim][num] to points[num][dim]
        points = np.delete(points, 3, axis=1)  # cut-off intensity
        return points

    def get_motion_flow_annotation(self, dataset_module, cur_sample, coordinates):
        # https://deepai.org/publication/scalable-scene-flow-from-point-clouds-in-the-real-world
        # https://arxiv.org/pdf/2103.01306v3.pdf
        # chapter 3.2

        motion_flow_annotation = np.full(coordinates.shape[0], None)

        if len(cur_sample[nf.PREV]) == 0:
            return motion_flow_annotation

        prev_sample = dataset_module.get(nf.SAMPLE, cur_sample[nf.PREV])

        # 2 Hz, ~0.5 s for NuScenes dataset
        # 5 Hz, ~0.2 s for Lyft dataset
        time_delta = cur_sample[nf.TIMESTAMP] - prev_sample[nf.TIMESTAMP]  # value in microseconds
        time_delta /= 1000000  # value in seconds

        for box_token in cur_sample[nf.ANNS]:
            cur_box_metadata = dataset_module.get(nf.SAMPLE_ANNOTATION, box_token)

            if len(cur_box_metadata[nf.PREV]) == 0:
                continue

            prev_box_metadata = dataset_module.get(nf.SAMPLE_ANNOTATION, cur_box_metadata[nf.PREV])

            if prev_box_metadata[nf.SAMPLE_TOKEN] != prev_sample[nf.TOKEN]:
                continue

            # Transformation matrix from Current box view to global view
            # TESTED, np.linalg.inv(cur_box_to_global) @ cur_center = [0 0 0]
            cur_box_to_global = np.eye(4)
            cur_box_to_global[:3, :3] = Quaternion(cur_box_metadata[nf.ROTATION]).rotation_matrix
            cur_box_to_global[:3, 3] = cur_box_metadata[nf.TRANSLATION]
            global_to_cur_box = np.linalg.inv(cur_box_to_global)

            # Transformation matrix from Previous box view to global view
            # TESTED, np.linalg.inv(prev_box_to_global) @ prev_center = [0 0 0]
            prev_box_to_global = np.eye(4)
            prev_box_to_global[:3, :3] = Quaternion(prev_box_metadata[nf.ROTATION]).rotation_matrix
            prev_box_to_global[:3, 3] = prev_box_metadata[nf.TRANSLATION]

            # T_delta, transformation matrix from current box to previous box
            # TESTED, prev_center - (cur_box_to_prev_box @ cur_center) = [0 0 0]
            cur_box_to_prev_box = prev_box_to_global @ global_to_cur_box

            # adding 1 to end of each point end
            coordinates_reshape = np.concatenate((coordinates, np.ones((coordinates.shape[0], 1))), axis=1)

            center = cur_box_metadata["translation"]
            size = cur_box_metadata["size"]
            yaw = Quaternion(cur_box_metadata[nf.ROTATION]).yaw_pitch_roll[0]

            point_mask = get_point_mask(coordinates, [center[0], center[1], center[2], size[0], size[1], size[2], yaw])

            for point_index in range(len(coordinates)):
                if point_mask[point_index]:
                    pm1 = cur_box_to_prev_box @ np.transpose(coordinates_reshape[point_index])
                    pm1 = np.delete(pm1, 3)
                    motion_flow_annotation[point_index] = coordinates[point_index] - pm1
                    motion_flow_annotation[point_index] /= time_delta

        return motion_flow_annotation

    def get_label_list(self, sample: dict):
        """
        :param sample: Nuscenes sample
        :return labels list labels[num]
                    num - number of point in coordinates array
        """
        lidar_top_data = self.nusc.get(nf.SAMPLE_DATA, sample[nf.DATA][nf.LIDAR_TOP])
        lidar_token = lidar_top_data[nf.TOKEN]
        try:
            lidarseg_labels_filename = path.join(self.dataset_path,
                                                 self.nusc.get(nf.LIDARSEG, lidar_token)[nf.FILENAME])
        except:
            return []

        points_label = data_classes.load_bin_file(lidarseg_labels_filename)
        id2label_dict = self.nusc.lidarseg_idx2name_mapping

        labels_list = []
        for label in points_label:
            labels_list.append(get_unificated_category_id(id2label_dict[label]))

        return labels_list

    def get_boxes(self, dataset_module, sample):
        boxes_list = []

        for box_token in sample[nf.ANNS]:
            box_metadata = dataset_module.get(nf.SAMPLE_ANNOTATION, box_token)
            box_inf = dict()
            box_inf['category_id'] = get_unificated_category_id(box_metadata['category_name'])
            box_inf['wlh'] = box_metadata['size']
            box_inf['center'] = box_metadata[nf.TRANSLATION]

            yaw = Quaternion(box_metadata[nf.ROTATION]).yaw_pitch_roll[0]
            box_inf['orientation'] = yaw

            boxes_list.append(box_inf)

        return boxes_list

    def get_dataset_type(self, scene_name):
        if scene_name in nusc_dataset_type.train or scene_name in nusc_dataset_type.mini_train:
            return "train"
        elif scene_name in nusc_dataset_type.test:
            return "test"
        elif scene_name in nusc_dataset_type.val or scene_name in nusc_dataset_type.mini_val:
            return "valid"
        else:
            return "unrecognized"

    def get_categories(self):
        """
        Returns categories of dataset objects
        :return: categories list, category consist of name, description
                [{'name': str, 'description': str},{}]
        """
        categories = self.nusc.category
        for category in categories:
            del category[nf.TOKEN]
            del category[nf.INDEX]

        return categories
