import numpy as np

import parser
from os import getcwd, listdir
from dataset_modules.a2d2_module.a2d2_utils import *
from dataset_modules.utils import get_unificated_category_id, get_point_mask

dataset_types_list = ['camera_lidar', 'camera_lidar_semantic', 'camera_lidar_semantic_bboxes']


class A2D2Parser(parser.Parser):

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        with open(path.join(getcwd(), 'dataset_modules', 'a2d2_module', 'cams_lidars.json'), 'r') as f:
            self.config = json.load(f)

        self.dataset_type = sorted([dir_name for dir_name in listdir(self.dataset_path) if
                                    path.isdir(path.join(self.dataset_path, dir_name))])[0]

        # DEBUG TYPE
        self.dataset_type = dataset_types_list[2]
        self.points_flag = -1

        dataset_path_type = path.join(self.dataset_path, self.dataset_type)
        self.scenes_list = sorted(
            [dir_name for dir_name in listdir(dataset_path_type) if path.isdir(path.join(dataset_path_type, dir_name))])

        self.vehicle_view = self.config['vehicle']['view']  # global view
        self.categories = self.__get_categories()

    def __get_nth_sample(self, scene_number):
        scene = self.scenes_list[scene_number]
        sample_path = path.join(self.dataset_path, self.dataset_type, scene)
        return sample_path

    def get_data(self, scene_number: int, frame_number: int):
        sample_path = self.__get_nth_sample(scene_number)
        frame_id = get_frame_id(sample_path, frame_number)

        if frame_number-1 >= 0:
            previous_frame_id = get_frame_id(sample_path,frame_number-1)
        else:
            previous_frame_id = -1

        coord = self.get_coordinates(sample_path, frame_id)
        transformation_matrix = get_transform_to_global(self.vehicle_view)
        boxes = [] if self.dataset_type != dataset_types_list[2] else self.get_boxes(sample_path, frame_id)

        motion_flow_anotation = self.get_motion_flow_annotation(sample_path, frame_id, previous_frame_id, boxes,coord)

        labels = self.get_labels(sample_path, frame_id)
        dataset_type = self.get_dataset_type()

        data = {'dataset_type':dataset_type,'coordinates': coord, 'transformation_matrix': transformation_matrix, 'boxes': boxes, 'labels': labels}

        return data

    def get_labels(self, sample_path, frame_id):
        if self.dataset_type == dataset_types_list[0]:
            return []
        lidar_path = glob.glob(path.join(sample_path, 'lidar/cam_front_center', '*' + frame_id + '*'))[0]
        current_lidar = np.load(lidar_path)

        file_name_label_image = glob.glob(path.join(sample_path, 'label/cam_front_center/', '*' + frame_id + '*'))[0]
        label_image = cv2.imread(file_name_label_image)
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        label_image = undistort_image(self.config, label_image, 'front_center')

        # get labels from colors on photo
        rows = (current_lidar['row'] + 0.5).astype(np.int)
        cols = (current_lidar['col'] + 0.5).astype(np.int)
        colours = label_image[rows, cols, :]

        labels_list = []
        for label in colours:
            hex_color = rgb_to_hex(tuple(label))
            if hex_color not in self.categories.keys():
                # undefined category
                labels_list.append(0)
            else:
                labels_list.append(get_unificated_category_id(self.categories[hex_color]))

        return labels_list

    def get_coordinates(self, sample_path, frame_id):
        global_coordinates = []

        available_lidars = [dir_name for dir_name in listdir(path.join(sample_path, 'lidar')) if
                            path.isdir(path.join(sample_path, 'lidar', dir_name))]



        for current_folder in available_lidars:
            current_lidar_name = current_folder[4:]
            lidar_view = self.config['cameras'][current_lidar_name]['view']

            lidar_frame_path = \
                sorted(glob.glob(path.join(sample_path, 'lidar', current_folder, '*' + frame_id + '.npz')))[0]

            current_coord = self.__get_lidar_coordinates(lidar_frame_path)
            current_coord = project_lidar_from_to(current_coord, lidar_view, self.vehicle_view)

            current_coord = np.array(current_coord)

            if len(global_coordinates) == 0:
                global_coordinates = current_coord
            else:
                global_coordinates = np.concatenate((global_coordinates, current_coord))


        print(global_coordinates.shape)
        return global_coordinates

    def __get_lidar_coordinates(self, lidar_path):
        current_lidar = np.load(lidar_path)
        if self.points_flag == -1:
            for key in current_lidar.keys():
                if 'points' in key:
                    self.points_flag = key

        points = current_lidar[self.points_flag]

        return points

    def get_motion_flow_annotation(self, sample_path, frame_id, previous_frame_id, boxes, coordinates):

        motion_flow_annotation = np.full(coordinates.shape[0], None)
        if previous_frame_id == -1 or self.dataset_type != dataset_types_list[2]:
            return motion_flow_annotation

        # tmp_cnt = []

        time_delta = 1

        available_lidars = [dir_name for dir_name in listdir(path.join(sample_path, 'lidar')) if
                            path.isdir(path.join(sample_path, 'lidar', dir_name))]
        for current_folder in available_lidars:
            cur_lidar_frame_path = \
            sorted(glob.glob(path.join(sample_path, 'lidar', current_folder, '*' + frame_id + '.npz')))[0]
            prev_lidar_frame_path = \
            sorted(glob.glob(path.join(sample_path, 'lidar', current_folder, '*' + previous_frame_id + '.npz')))[0]
            current_lidar = np.load(cur_lidar_frame_path)
            prev_lidar = np.load(prev_lidar_frame_path)

            cur_timestamp = np.mean(current_lidar['timestamp'])
            prev_timestamp = np.mean(prev_lidar['timestamp'])
            time_delta = cur_timestamp - prev_timestamp  # value in microseconds
            time_delta /= 1000000 # value in seconds

            print(cur_timestamp)
            print(prev_timestamp)
        print(time_delta)

        # Get raw boxes from previous frame (to check class top/left/bottom etc)
        file_name_bboxes = glob.glob(path.join(sample_path, 'label3D/cam_front_center/', '*' + previous_frame_id + '*'))[0]
        boxes2 = read_bounding_boxes(file_name_bboxes)
        boxes2 = reformate_boxes(boxes2)

        box_number = 0
        for box in boxes:
            prev_box = get_same_box(boxes2, boxes[box_number])
            if prev_box is None:
                continue

            # Transformation matrix from Current box view to global view
            # TESTED, np.linalg.inv(cur_box_to_global) @ cur_center = [0 0 0]
            cur_box_to_global = np.eye(4)
            cur_box_to_global[:3, :3] = box['rotation_matrix']
            cur_box_to_global[:3, 3] = box['center']
            global_to_cur_box = np.linalg.inv(cur_box_to_global)



            # Transformation matrix from Previous box view to global view
            # TESTED, np.linalg.inv(prev_box_to_global) @ prev_center = [0 0 0]
            prev_box_to_global = np.eye(4)
            prev_box_to_global[:3, :3] = prev_box['rotation_matrix']
            prev_box_to_global[:3, 3] = prev_box['center']

            # T_delta, transformation matrix from current box to previous box
            # TESTED, prev_center - (cur_box_to_prev_box @ cur_center) = [0 0 0]
            cur_box_to_prev_box = prev_box_to_global @ global_to_cur_box

            coordinates_reshape = np.concatenate((coordinates, np.ones((coordinates.shape[0], 1))), axis=1)


            center = box['center']
            size = box['wlh']
            yaw = box['orientation']

            point_mask = get_point_mask(coordinates, [center[0], center[1], center[2], size[0], size[1], size[2], yaw])

            # tmp_true_cnt = 0
            for point_index in range(len(coordinates)):
                if point_mask[point_index]:
                    pm1 = cur_box_to_prev_box @ np.transpose(coordinates_reshape[point_index])
                    pm1 = np.delete(pm1, 3)
                    motion_flow_annotation[point_index] = coordinates[point_index] - pm1
                    motion_flow_annotation[point_index] /= time_delta
                    print(motion_flow_annotation[point_index])
                    # tmp_true_cnt += 1
            # tmp_cnt.append(tmp_true_cnt)
            box_number+=1
        # print("CNT",tmp_cnt)

    def get_boxes(self, sample_path, frame_id):
        file_name_bboxes = glob.glob(path.join(sample_path, 'label3D/cam_front_center/', '*' + frame_id + '*'))[0]
        boxes = read_bounding_boxes(file_name_bboxes)
        print(boxes[0])
        boxes = reformate_boxes(boxes)
        return boxes

    def get_dataset_type(self):
        if self.dataset_type == dataset_types_list[0]:
            return "test"
        elif self.dataset_type == dataset_types_list[1]:
            return "valid"
        elif self.dataset_type == dataset_types_list[2]:
            return "train"
        return None

    def get_categories(self):
        return self.categories

    def __get_categories(self):
        if self.dataset_type == dataset_types_list[0]:
            return dict()
        class_list_path = path.join(self.dataset_path, self.dataset_type, 'class_list.json')
        with open(class_list_path, 'r') as f:
            class_dict = json.load(f)
        return class_dict
