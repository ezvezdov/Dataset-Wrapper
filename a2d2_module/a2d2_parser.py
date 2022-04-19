import parser
import json
from os import path
import os
import numpy as np
import numpy.linalg as la
import glob

import open3d as o3

dataset_types_list = ['camera_lidar', 'camera_lidar_semantic', 'camera_lidar_semantic_bboxes']

#TODO: finish transformation to global view
class A2D2Parser(parser.Parser):

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        with open(path.join(self.dataset_path, 'cams_lidars.json'), 'r') as f:
            self.config = json.load(f)

        self.dataset_type = sorted([dir_name for dir_name in os.listdir(self.dataset_path) if
                                    os.path.isdir(os.path.join(self.dataset_path, dir_name))])[0]

        # DEBUG TYPE
        # TODO: comment out debug
        #self.dataset_type = dataset_types_list[0]

        self.points_flag = -1

        self.dataset_path_type = os.path.join(self.dataset_path, self.dataset_type)
        self.scenes_list = [dir_name for dir_name in os.listdir(self.dataset_path_type) if
                            os.path.isdir(os.path.join(self.dataset_path_type, dir_name))]
        print(self.scenes_list)

        self.vehicle_view = self.config['vehicle']['view']  # global view

    def _get_nth_sample(self, scene_number):
        scene = self.scenes_list[scene_number]
        sample_path = os.path.join(self.dataset_path_type, scene)
        return sample_path

    def get_data(self, scene_number: int, frame_number: int):
        sample_path = self._get_nth_sample(scene_number)

        # TODO data from different lidar sensors
        cam_lid_sb_path = path.join(self.dataset_path, self.dataset_type)
        file_names = sorted(glob.glob(path.join(cam_lid_sb_path, '*/lidar/cam_front_center/*.npz')))
        file_name_lidar = file_names[7]

        # coord = self.get_coordinates(file_name_lidar)
        # transformation_matrix = self.get_transformation_matrix(view)
        coord = self.get_coordinates(sample_path, frame_number)

        transformation_matrix = self.get_transformation_matrix(self.vehicle_view)
        boxes = [] if self.dataset_type != dataset_types_list[2] else self.get_boxes(file_name_lidar)

        data = {'coordinates': coord, 'transformation_matrix': transformation_matrix, 'boxes': boxes, 'labels': []}
        return data

    def get_coordinates(self, sample_path, frame_number):
        global_coordinates = np.array(2)

        available_lidars = [dir_name for dir_name in os.listdir(os.path.join(sample_path, 'lidar')) if
                            os.path.isdir(os.path.join(sample_path, 'lidar', dir_name))]

        lidars_list = list(self.config['lidars'].keys())
        print(lidars_list)

        for current_folder in available_lidars:
            current_lidar_name = current_folder[4:]
            if current_lidar_name not in lidars_list:
                continue
            view = self.config['lidars'][current_lidar_name]['view']
            transformation_matrix = self.get_transformation_matrix(view)

            frames_list = sorted(glob.glob(os.path.join(sample_path, 'lidar', current_folder, '*.npz')))
            lidar_frame_path = frames_list[frame_number]

            # current_coord = self.__project_lidar_from_to(lidar_frame_path,view,self.vehicle_view)



            # tmp_current_lidar = np.load(lidar_frame_path)
            # print(list(tmp_current_lidar.keys()))
            # tmp_pcd_front_center = self.create_open3d_pc(tmp_current_lidar)
            # o3.visualization.draw_geometries([tmp_pcd_front_center])

        return global_coordinates

    def __get_lidar_coordinates(self, frame_path):
        current_lidar = np.load(frame_path)
        if self.points_flag == -1:
            for key in current_lidar.keys():
                if 'points' in key:
                    self.points_flag = key

        points = current_lidar[self.points_flag]
        return points

    # def __project_lidar_from_to(self,lidar, src_view, target_view):
    #     lidar = dict(lidar)
    #     trans = self.__transform_from_to(src_view, target_view)
    #     points = lidar['points']
    #     points_hom = np.ones((points.shape[0], 4))
    #     points_hom[:, 0:3] = points
    #     points_trans = (np.dot(trans, points_hom.T)).T
    #     lidar['points'] = points_trans[:, 0:3]
    #
    #     return lidar

    # TMP
    # def create_open3d_pc(self, lidar, cam_image=None):
    #     # create open3d point cloud
    #     pcd = o3.geometry.PointCloud()
    #
    #     # assign point coordinates
    #     pcd.points = o3.utility.Vector3dVector(lidar[self.points_flag])
    #
    #     # assign colours
    #     if cam_image is None:
    #         median_reflectance = np.median(lidar['pcloud_attr.reflectance'])
    #
    #         # clip colours for visualisation on a white background
    #     else:
    #         rows = (lidar['pcloud.row'] + 0.5).astype(np.int)
    #         cols = (lidar['pcloud.col'] + 0.5).astype(np.int)
    #
    #     pcd.colors = o3.utility.Vector3dVector()
    #
    #     return pcd

    def get_categories(self):
        if self.dataset_type != dataset_types_list[2]:
            return []
        cam_lid_sb_path = path.join(self.dataset_path, self.dataset_type)
        with open(path.join(cam_lid_sb_path, 'class_list.json'), 'r') as f:
            class_dict = json.load(f)
        print(class_dict)

        return class_dict

    def get_transformation_matrix(self, view):
        # get axes
        x_axis, y_axis, z_axis = self.__get_axes_of_a_view(view)

        # get origin
        origin = self.__get_origin_of_a_view(view)
        transform_to_global = np.eye(4)

        # rotation
        transform_to_global[0:3, 0] = x_axis
        transform_to_global[0:3, 1] = y_axis
        transform_to_global[0:3, 2] = z_axis

        # origin
        transform_to_global[0:3, 3] = origin

        return transform_to_global

    def __get_axes_of_a_view(self, view):
        EPSILON = 1.0e-10  # norm should not be small

        x_axis = view['x-axis']
        y_axis = view['y-axis']

        x_axis_norm = la.norm(x_axis)
        y_axis_norm = la.norm(y_axis)

        if x_axis_norm < EPSILON or y_axis_norm < EPSILON:
            raise ValueError("Norm of input vector(s) too small.")

        # normalize the axes
        x_axis = x_axis / x_axis_norm
        y_axis = y_axis / y_axis_norm

        # make a new y-axis which lies in the original x-y plane, but is orthogonal to x-axis
        y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)

        # create orthogonal z-axis
        z_axis = np.cross(x_axis, y_axis)

        # calculate and check y-axis and z-axis norms
        y_axis_norm = la.norm(y_axis)
        z_axis_norm = la.norm(z_axis)

        if (y_axis_norm < EPSILON) or (z_axis_norm < EPSILON):
            raise ValueError("Norm of view axis vector(s) too small.")

        # make x/y/z-axes orthonormal
        y_axis = y_axis / y_axis_norm
        z_axis = z_axis / z_axis_norm

        return x_axis, y_axis, z_axis

    def __get_origin_of_a_view(self, view):
        return view['origin']

    def get_boxes(self, file_name_lidar):
        seq_name = file_name_lidar.split('/')[-4]

        file_name_bboxes = self.__extract_bboxes_file_name_from_lidar_file_name(file_name_lidar)
        file_name_bboxes = path.join(self.dataset_path, self.dataset_type, seq_name, 'label3D/cam_front_center/',
                                     file_name_bboxes)
        boxes = self.__read_bounding_boxes(file_name_bboxes)
        boxes = self.__reformate_boxes(boxes)
        return boxes

    def __reformate_boxes(self, boxes):
        boxes_list = []
        for i in range(len(boxes)):
            box_inf = dict()
            box_inf['name'] = boxes[i]['class']
            box_inf['wlh'] = boxes[i]['size']
            box_inf['center'] = boxes[i]['center']
            box_inf['orientation'] = boxes[i]['rotation']
            boxes_list.append(box_inf)
        return boxes_list

    def __skew_sym_matrix(self, u):
        return np.array([[0, -u[2], u[1]],
                         [u[2], 0, -u[0]],
                         [-u[1], u[0], 0]])

    def __axis_angle_to_rotation_mat(self, axis, angle):
        return np.cos(angle) * np.eye(3) + \
               np.sin(angle) * self.__skew_sym_matrix(axis) + \
               (1 - np.cos(angle)) * np.outer(axis, axis)

    def __extract_bboxes_file_name_from_lidar_file_name(self, file_name_lidar):
        file_name_bboxes = file_name_lidar.split('/')
        file_name_bboxes = file_name_bboxes[-1].split('.')[0]
        file_name_bboxes = file_name_bboxes.split('_')
        file_name_bboxes = file_name_bboxes[0] + '_' + \
                           'label3D_' + \
                           file_name_bboxes[2] + '_' + \
                           file_name_bboxes[3] + '.json'

        return file_name_bboxes

    def __read_bounding_boxes(self, file_name_bboxes):
        # open the file
        with open(file_name_bboxes, 'r') as f:
            bboxes = json.load(f)

        boxes = []  # a list for containing bounding boxes

        for bbox in bboxes.keys():
            bbox_read = dict()  # a dictionary for a given bounding box
            bbox_read['class'] = bboxes[bbox]['class']
            bbox_read['truncation'] = bboxes[bbox]['truncation']
            bbox_read['occlusion'] = bboxes[bbox]['occlusion']
            bbox_read['alpha'] = bboxes[bbox]['alpha']
            bbox_read['top'] = bboxes[bbox]['2d_bbox'][0]
            bbox_read['left'] = bboxes[bbox]['2d_bbox'][1]
            bbox_read['bottom'] = bboxes[bbox]['2d_bbox'][2]
            bbox_read['right'] = bboxes[bbox]['2d_bbox'][3]
            bbox_read['center'] = np.array(bboxes[bbox]['center'])
            bbox_read['size'] = np.array(bboxes[bbox]['size'])
            angle = bboxes[bbox]['rot_angle']
            axis = np.array(bboxes[bbox]['axis'])
            bbox_read['rotation'] = self.__axis_angle_to_rotation_mat(axis, angle)
            boxes.append(bbox_read)

        return boxes
