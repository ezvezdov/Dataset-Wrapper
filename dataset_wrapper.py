import json
import os

# Datasets names
NUSCENES_NAME = "nuscenes"
LYFT_NAME = "lyft"
WAYMO_NAME = "waymo"
A2D2_NAME = "a2d2"


class DatasetWrapper:
    def __init__(self, dataset_name: str, dataset_path: str):
        self.dataset_name = dataset_name.lower()
        self.dataset_path = dataset_path
        self.parser = None
        self.categories_cat2id = dict()
        self.categories_id2cat = dict()

        self.__init_parser__()
        self.__init_categories__()

    def __init_parser__(self):
        if self.dataset_name == NUSCENES_NAME:
            # TODO add catching exceptions for non lidarseg edition
            import dataset_modules.nuscenes_based.nuscenes_parser as num
            self.parser = num.NuScenesParser(self.dataset_path)
        elif self.dataset_name == LYFT_NAME:
            # TODO add catching exceptions
            import dataset_modules.nuscenes_based.lyft_parser as lyp
            self.parser = lyp.LyftParser(self.dataset_path)
        elif self.dataset_name == WAYMO_NAME:
            import dataset_modules.waymo_module.waymo_parser as wp
            self.parser = wp.WaymoParser(self.dataset_path)
        elif self.dataset_name == A2D2_NAME:
            import dataset_modules.a2d2_module.a2d2_parser as a2d2p
            self.parser = a2d2p.A2D2Parser(self.dataset_path)
        else:
            print("Error: unknown dataset!")
            print("All possibles datasets:", NUSCENES_NAME, LYFT_NAME, A2D2_NAME, WAYMO_NAME)
            exit(1)

    def __init_categories__(self):
        file_path = os.path.join(os.getcwd(), "resources", "categories-category2id.json")
        file = open(file_path, "r")
        self.categories_cat2id = json.load(file)

        file_path = os.path.join(os.getcwd(), "resources", "categories-id2category.json")
        file = open(file_path, "r")
        self.categories_id2cat = json.load(file)


    def get_item(self, scene_number: int, frame_number: int):
        """
        Return main lidar data of selected frame in scene: coordinates and each point category

        :param scene_number: Number of scene
        :param frame_number: Number of frame in selected scene
        :return: Dictionary with coordinates numpy array and labels list
                {'dataset_type' : str,'coordinates' : numpy array, transformation_matrix' : numpy array,
                'boxes': list, 'labels' : list}
                possible dataset_type values: 'unrecognized','train', 'valid', 'test'

        """
        data = self.parser.get_data(scene_number, frame_number)
        return data

    def get_map(self):
        if self.dataset_name != NUSCENES_NAME:
            print("This dataset has no map!")
            return
        return self.parser.get_map()

    def get_unificated_categories(self):
        """
        Returns unificated categories of objects

        :return: categories list, category consist of id, and category string
                { id : category, ... }
        """
        return self.categories_id2cat

    def get_dataset_categories(self):
        """
        Returns dataset's categories of objects
        :return: categories list, category consist of name, description
                [{'name': str, 'description': str} ...]
        """
        return self.parser.get_categories()

    def get_category_by_id(self, id: int):
        """
        Returns unificated category by id

        :param id: id of category, returned from dataset
        :return: string with category
        """
        return self.categories_id2cat[str(id)]