import parser
import json
import pprint
from os import path
import numpy as np
import numpy.linalg as la
import glob


class A2D2Parser(parser.Parser):

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        with open(path.join(self.dataset_path, 'cams_lidars.json'), 'r') as f:
            self.config = json.load(f)

    def get_data(self, scene_number: int, frame_number: int):
        view = self.config['cameras']['front_left']['view']
        coord = self.get_coordinates()
        categories = self.get_categories()
        transformation_matrix = self.get_transformation_matrix(view)

        data = {'coordinates': coord, 'transformation_matrix': transformation_matrix, 'boxes': '', 'labels': ''}
        return data

    def get_coordinates(self):
        cam_lid_sb_path = path.join(self.dataset_path, 'camera_lidar_semantic_bboxes')
        file_names = sorted(glob.glob(path.join(cam_lid_sb_path, '*/lidar/cam_front_center/*.npz')))
        file_name_lidar = file_names[7]
        lidar_front_center = np.load(file_name_lidar)
        points = lidar_front_center['points']

        return points

    def get_categories(self):
        cam_lid_sb_path = path.join(self.dataset_path, 'camera_lidar_semantic_bboxes')
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