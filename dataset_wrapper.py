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
            #TODO add catching exceptions for non lidarseg edition
            import nuscenes_based.nuscenes_parser as num
            self.parser = num.NuScenesParser(self.dataset_path)
        elif self.dataset_name == LYFT_NAME:
            # TODO add catching exceptions
            import nuscenes_based.lyft_parser as lyp
            self.parser = lyp.LyftParser(self.dataset_path)

        elif self.dataset_name == WAYMO_NAME:
            # TODO: add waymo init
            import waymo_module.waymo_parser as wp
            self.parser = wp.WaymoParser(self.dataset_path)
        elif self.dataset_name == A2D2_NAME:
            import a2d2_module.a2d2_parser as a2d2p
            self.parser = a2d2p.A2D2Parser(self.dataset_path)
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
        data = self.parser.get_data(scene_number, frame_number)
        return data

    def get_categories(self):
        """
        Returns categories of dataset objects
        :return: categories list, category consist of name, description
                [{'name': str, 'description': str},{}]
        """
        return self.parser.get_categories()
