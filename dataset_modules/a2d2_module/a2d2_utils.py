import json
from os import path
import numpy as np
import numpy.linalg as la
import glob

import cv2

from dataset_modules.utils import get_unificated_category_id


def __get_axes_of_a_view(view):
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


def get_frame_id(sample_path, frame_number):
    frames_list = sorted(glob.glob(path.join(sample_path, 'lidar/cam_front_center/*.npz')))
    frame_path = frames_list[frame_number]
    frame_lidar = frame_path.split('/')[-1]
    frame_id = frame_lidar.split('_')[-1]
    frame_id = frame_id.split('.')[0]
    return frame_id


def __get_origin_of_a_view(view):
    return view['origin']


def get_transform_to_global(view):
    # get axes
    x_axis, y_axis, z_axis = __get_axes_of_a_view(view)

    # get origin
    origin = __get_origin_of_a_view(view)
    transform_to_global = np.eye(4)

    # rotation
    transform_to_global[0:3, 0] = x_axis
    transform_to_global[0:3, 1] = y_axis
    transform_to_global[0:3, 2] = z_axis

    # origin
    transform_to_global[0:3, 3] = origin

    return transform_to_global


def __get_transform_from_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    trans = np.eye(4)
    rot = np.transpose(transform_to_global[0:3, 0:3])
    trans[0:3, 0:3] = rot
    trans[0:3, 3] = np.dot(rot, -transform_to_global[0:3, 3])

    return trans


def __transform_from_to(src, target):
    transform = np.dot(__get_transform_from_global(target), get_transform_to_global(src))

    return transform


def project_lidar_from_to(points, src_view, target_view):
    trans = __transform_from_to(src_view, target_view)
    points_hom = np.ones((points.shape[0], 4))
    points_hom[:, 0:3] = points
    points_trans = (np.dot(trans, points_hom.T)).T
    points = points_trans[:, 0:3]

    return points


###################################################
# boxes
def __skew_sym_matrix(u):
    return np.array([[0, -u[2], u[1]],
                     [u[2], 0, -u[0]],
                     [-u[1], u[0], 0]])


def __axis_angle_to_rotation_mat(axis, angle):
    return np.cos(angle) * np.eye(3) + \
           np.sin(angle) * __skew_sym_matrix(axis) + \
           (1 - np.cos(angle)) * np.outer(axis, axis)


def read_bounding_boxes(file_name_bboxes):
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
        bbox_read['rotation'] = __axis_angle_to_rotation_mat(axis, angle)
        boxes.append(bbox_read)

    return boxes


def reformate_boxes(boxes):
    boxes_list = []
    for i in range(len(boxes)):
        box_inf = dict()
        box_inf['category_id'] = get_unificated_category_id(boxes[i]['class'])
        box_inf['wlh'] = boxes[i]['size']
        box_inf['center'] = boxes[i]['center']
        box_inf['orientation'] = boxes[i]['rotation']
        boxes_list.append(box_inf)
    return boxes_list


#######################################################
# For segmentation labels
def undistort_image(config, image, cam_name):
    if cam_name in ['front_left', 'front_center', 'front_right', 'side_left', 'side_right', 'rear_center']:
        # get parameters from config file
        intr_mat_undist = \
            np.asarray(config['cameras'][cam_name]['CamMatrix'])
        intr_mat_dist = \
            np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
        dist_parms = \
            np.asarray(config['cameras'][cam_name]['Distortion'])
        lens = config['cameras'][cam_name]['Lens']

        if lens == 'Fisheye':
            return cv2.fisheye.undistortImage(image, intr_mat_dist, D=dist_parms, Knew=intr_mat_undist)
        elif lens == 'Telecam':
            return cv2.undistort(image, intr_mat_dist, distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    else:
        return image


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
