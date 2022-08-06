class Parser:
    def get_data(self, scene_number: int, frame_number: int):
        """
        :param scene_number: Number of scene
        :param frame_number: Number of frame

        :return: Dictionary with coordinates numpy array and labels list {'dataset_type': str,
        'motion_flow_annotation': ndarray, 'coordinates' : ndarray, 'transformation_matrix': ndarray, 'labels': list
        'boxes': list}
        """
        pass

    def get_map(self):
        print("This dataset has no map!")
        return []

    def get_categories(self):
        pass

