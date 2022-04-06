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
        self.__init_parser__()

    def __init_parser__(self):
        if self.dataset_name == NUSCENES_NAME:
            import nuscenes_module.nuscenes_parser as num
            self.parser = num.NuScenesParser(self.dataset_path)
        elif self.dataset_name == LYFT_NAME:
            # TODO: add lyft init
            pass
        elif self.dataset_name == WAYMO_NAME:
            # TODO: add waymo init
            pass
        elif self.dataset_name == A2D2_NAME:
            # TODO: add a2d2 init
            pass
        else:
            # TODO: unknown_dataset
            pass

    def get_item(self, scene_number: int, frame_number: int):
        """
        Return main lidar data of selected frame in scene: coordinates and each point category

        :param scene_number: Number of scene
        :param frame_number: Number of frame in selected scene
        :return: Dictionary with coordinates numpy array and labels list
                {'coordinates' : numpy array, 'labels' : labels list}
        """
        data = self.parser.get_data(frame_number, scene_number)
        return data

    def get_categories(self):
        """
        Returns categories of dataset objects
        :return: categories list, category consist of name, description
                [{'name': str, 'description': str},{}]
        """
        return self.parser.get_categories()
