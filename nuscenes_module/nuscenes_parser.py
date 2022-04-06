from nuscenes.nuscenes import NuScenes
from nuscenes.utils import data_classes
from os import path
import nuscenes_module.nuscenes_flags as nf
import parser


class NuScenesParser(parser.Parser):

    def __init__(self, dataset_path: str):
        self.nusc = NuScenes(dataroot=dataset_path, verbose=True)
        self.dataset_path = dataset_path
        self.frame_number = None
        self.scene_number = None

    def set_frame_number(self, frame_number):
        self.frame_number = frame_number

    def set_scene_number(self, scene_number):
        self.scene_number = scene_number

    def __get_nth_sample(self, scene):
        sample = self.nusc.get(nf.SAMPLE, scene[nf.FIRST_SAMPLE_TOKEN])
        for i in range(self.frame_number):
            sample = self.nusc.get(nf.SAMPLE, sample[nf.NEXT])
        return sample

    def get_coordinates(self):
        scene = self.nusc.scene[self.scene_number]
        sample = self.__get_nth_sample(scene)
        ###
        lidar_top_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pcd = data_classes.LidarPointCloud.from_file(path.join(self.dataset_path, lidar_top_data['filename']))
        #np.set_printoptions(threshold=np.inf)  # print full array
        #print(pcd.points.size)
        ###
        # sample_annotation = sample[nf.NuScenesFlags.ANNS]

        # print(lidar_top_data)

        return pcd.points
