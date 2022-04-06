NUSCENES_NAME = "nuscenes_module"
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
            pass
            # TODO: add lyft init
        elif self.dataset_name == WAYMO_NAME:
            pass
            # TODO: add waymo init
        elif self.dataset_name == A2D2_NAME:
            pass
            # TODO: add a2d2 init

    def get_item(self, frame_number, scene):
        self.parser.set_frame_number(frame_number)
        self.parser.set_scene_number(scene)
        print(self.parser.get_coordinates())
