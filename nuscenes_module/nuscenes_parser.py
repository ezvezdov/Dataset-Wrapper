from os import path
import parser

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import data_classes
import nuscenes_module.nuscenes_flags as nf


class NuScenesParser(parser.Parser):

    def __init__(self, dataset_path: str):
        self.nusc = NuScenes(dataroot=dataset_path, verbose=True)
        self.dataset_path = dataset_path

    def __get_nth_sample(self, scene, frame_number):
        sample = self.nusc.get(nf.SAMPLE, scene[nf.FIRST_SAMPLE_TOKEN])
        for i in range(frame_number):
            sample = self.nusc.get(nf.SAMPLE, sample[nf.NEXT])
        return sample

    def get_data(self, frame_number, scene_number):
        scene = self.nusc.scene[scene_number]
        sample = self.__get_nth_sample(scene, frame_number)
        coord = self.get_coordinates(sample)
        labels = self.get_label_list(sample)
        data = {'coordinates': coord, 'labels': labels}

        return data

    def get_coordinates(self, sample):
        """
        Return coordinates numpy array coord[dim][num]
            dim - dimension, {x,y,z}
            num - number of point
        :param sample: Nuscenes sample
        :return: coord: numpy array
        """
        lidar_top_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pcd = data_classes.LidarPointCloud.from_file(path.join(self.dataset_path, lidar_top_data['filename']))
        pcd.points = pcd.points[:3, :]  # cut-off intensity

        return pcd.points

    def get_label_list(self, sample):
        """
        Returns labels list lebels[num]
            num - number of point in coordinates array
        :param sample: Nuscenes sample
        :return: labels: list
        """
        lidar_top_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_token = lidar_top_data[nf.TOKEN]
        lidarseg_labels_filename = path.join(self.nusc.dataroot,
                                             self.nusc.get(nf.LIDARSEG, lidar_token)[nf.FILENAME])
        points_label = data_classes.load_bin_file(lidarseg_labels_filename)
        id2label_dict = self.nusc.lidarseg_idx2name_mapping

        labels_list = []
        for label in points_label:
            labels_list.append(id2label_dict[label])

        return labels_list
