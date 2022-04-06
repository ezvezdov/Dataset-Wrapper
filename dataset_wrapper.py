NUSCENES_NAME = "nuscenes"
LYFT_NAME = "lyft"
WAYMO_NAME = "waymo"
A2D2_NAME = "a2d2"


class DatasetWrapper:
    def __init__(self, dataset_name: str, dataset_path):
        self.dataset_name = dataset_name.lower()
        self.dataset_path = dataset_path
        self.parser = None
        self.__init_parser()

    def __init_parser(self):
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

    def get_item(self, frame_number, scene_number):
        data = self.parser.get_data(frame_number, scene_number)
        # print(data)
        return data
