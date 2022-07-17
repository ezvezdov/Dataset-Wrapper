import parser
from os import getcwd, listdir
from dataset_modules.a2d2_module.a2d2_utils import *
from dataset_modules.utils import get_unificated_category_id

dataset_types_list = ['camera_lidar', 'camera_lidar_semantic', 'camera_lidar_semantic_bboxes']


class A2D2Parser(parser.Parser):

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        with open(path.join(getcwd(), 'dataset_modules', 'a2d2_module', 'cams_lidars.json'), 'r') as f:
            self.config = json.load(f)

        self.dataset_type = sorted([dir_name for dir_name in listdir(self.dataset_path) if
                                    path.isdir(path.join(self.dataset_path, dir_name))])[0]

        # DEBUG TYPE
        # self.dataset_type = dataset_types_list[2]
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

        coord = self.get_coordinates(sample_path, frame_id)
        transformation_matrix = get_transform_to_global(self.vehicle_view)
        boxes = [] if self.dataset_type != dataset_types_list[2] else self.get_boxes(sample_path, frame_id)
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

            if len(global_coordinates) == 0:
                global_coordinates = current_coord
            else:
                global_coordinates = np.concatenate((global_coordinates, current_coord))



        return global_coordinates

    def __get_lidar_coordinates(self, lidar_path):
        current_lidar = np.load(lidar_path)
        if self.points_flag == -1:
            for key in current_lidar.keys():
                if 'points' in key:
                    self.points_flag = key

        points = current_lidar[self.points_flag]

        return points

    def get_boxes(self, sample_path, frame_id):
        file_name_bboxes = glob.glob(path.join(sample_path, 'label3D/cam_front_center/', '*' + frame_id + '*'))[0]
        boxes = read_bounding_boxes(file_name_bboxes)
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
