import json
import os
import numpy as np
import open3d as o3


def get_unificated_category_id(category: str):
    """
    Transform category to unificated category

    :param category: string of category
    :return: unificated category id
    """

    file_path = os.path.join(os.getcwd(), "resources", "categories-category2id.json")
    file = open(file_path, "r")
    categories = json.load(file)

    if category in categories.keys():
        return categories[category]
    return None


def get_point_mask(pcl, bbox, x_add=(0., 0.), y_add=(0., 0.), z_add=(0., 0.)):
    """
    :param pcl: x,y,z ...
    :param bbox: x,y,z,l,w,h,yaw
    :param x_add:
    :param y_add:
    :param z_add:
    :return: Segmentation mask
    """

    angle = bbox[6]
    Rot_z = np.array(([np.cos(angle), - np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]))
    s = pcl.copy()
    s[:, :3] -= bbox[:3]
    s[:, :3] = (s[:, :3] @ Rot_z)[:, :3]
    size = np.array((-bbox[3] / 2, bbox[3] / 2, -bbox[4] / 2, bbox[4] / 2, -bbox[5] / 2, bbox[5] / 2))
    point_mask = (size[0] - x_add[0] <= s[:, 0]) & (s[:, 0] <= size[1] + x_add[1]) & (size[2] - y_add[0] <= s[:, 1]) &\
                 (s[:, 1] <= size[3] + y_add[1]) & (size[4] - z_add[0] <= s[:, 2]) & (s[:, 2] <= size[5] + z_add[1])

    return point_mask


def update_motion_flow_annotation(coordinates, point_mask, motion_flow_annotation,
                                  cur_box_to_prev_box, time_delta):
    """
    Update motion flow annotation.

    :param coordinates: Points coordinates np.array (N,3)
    :param point_mask: Segmentation mask np.array (N)
    :param motion_flow_annotation: np.array with motion flow annotation
    :param cur_box_to_prev_box:  transformation matrix from current box to previous box np.array (4,4)
    :param time_delta: time difference of current and previous frame
    :return: Upgraded motion_flow_annotation array
    """

    # adding 1 to end of each point end (for multiplication)
    coordinates_reshape = np.concatenate((coordinates, np.ones((coordinates.shape[0], 1))), axis=1)

    for point_index in range(len(coordinates)):
        if point_mask[point_index]:
            # xyz of current point in previous frame
            pm1 = cur_box_to_prev_box @ np.transpose(coordinates_reshape[point_index])

            # removing 1 from end of point
            pm1 = np.delete(pm1, 3)

            # distance delta from current point to current point in previous frame
            motion_flow_annotation[point_index] = coordinates[point_index] - pm1

            # point speed in m/s
            motion_flow_annotation[point_index] /= time_delta

    return motion_flow_annotation


def __get_points(bbox):
    half_size = np.array(bbox['size']) / 2.

    if half_size[0] > 0:
        # calculate unrotated corner point offsets relative to center
        brl = np.asarray([-half_size[0], +half_size[1], -half_size[2]])
        bfl = np.asarray([+half_size[0], +half_size[1], -half_size[2]])
        bfr = np.asarray([+half_size[0], -half_size[1], -half_size[2]])
        brr = np.asarray([-half_size[0], -half_size[1], -half_size[2]])
        trl = np.asarray([-half_size[0], +half_size[1], +half_size[2]])
        tfl = np.asarray([+half_size[0], +half_size[1], +half_size[2]])
        tfr = np.asarray([+half_size[0], -half_size[1], +half_size[2]])
        trr = np.asarray([-half_size[0], -half_size[1], +half_size[2]])

        # rotate points
        rotation_matrix = np.array([
            [np.cos(bbox['orientation']), -np.sin(bbox['orientation']), 0.0],
            [np.sin(bbox['orientation']), np.cos(bbox['orientation']), 0.0],
            [0.0, 0.0, 1.0]])
        points = np.asarray([brl, bfl, bfr, brr, trl, tfl, tfr, trr])
        points = np.dot(points, rotation_matrix.T)

        # add center position
        points = points + bbox['center']

    return points


def _get_bboxes_wire_frames(bboxes, linesets=None, color=None):
    num_boxes = len(bboxes)

    # initialize linesets, if not given
    if linesets is None:
        linesets = [o3.geometry.LineSet() for _ in range(num_boxes)]

    # set default color
    if color is None:
        # color = [1, 0, 0]
        color = [0, 0, 1]

    assert len(linesets) == num_boxes, "Number of linesets must equal number of bounding boxes"

    # point indices defining bounding box edges
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [0, 4], [1, 5], [2, 6], [3, 7],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [5, 2], [1, 6]]

    # loop over all bounding boxes
    for i in range(num_boxes):
        # get bounding box corner points
        points = __get_points(bboxes[i])
        # update corresponding Open3d line set
        colors = [color for _ in range(len(lines))]
        line_set = linesets[i]
        line_set.points = o3.utility.Vector3dVector(points)
        line_set.lines = o3.utility.Vector2iVector(lines)
        line_set.colors = o3.utility.Vector3dVector(colors)

    return linesets


def _create_open3d_pc(points):
    pcd = o3.geometry.PointCloud()
    pcd.points = o3.utility.Vector3dVector(points)

    return pcd
