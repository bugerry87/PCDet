#Build In
import os
import sys
import pickle
import copy
import random

# Installed
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import torch
import spconv
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader

# Local
from pcdet.utils import box_utils, object3d_utils, calibration, common_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.config import cfg
from pcdet.datasets.data_augmentation.dbsampler import DataBaseSampler
from pcdet.datasets import DatasetTemplate


def shuffle_log(log:ArgoverseTrackingLoader):
    index = np.arange(log.num_lidar_frame)
    random.shuffle(index)
    for idx in index:
        lidar = log.get_lidar(idx)
        label = log.get_label_object(idx)
        yield idx, lidar, label, log


class BaseArgoDataset(DatasetTemplate):
    def __init__(self, root_path, subsets:list):
        super().__init__()
        self.root_path = root_path
        self.atls = {subset:ArgoverseTrackingLoader(Path(self.root_path) / subset) for subset in subsets}
        self._len = 0
        pass

    def __len__(self):
        if self._len is 0:
            for atl in self.atls.values():
                for log in iter(atl):
                    self._len += log.num_lidar_frame
        return self._len

    def __iter__(self):
        for atl in self.atls.values():
            for log in iter(atl):
                for idx in range(atl.num_lidar_frame):
                    lidar = log.get_lidar(idx)
                    label = log.get_label_object(idx)
                    yield idx, lidar, label, log
        pass

    def shuffle(self, seed=0):
        random.seed = seed
        generators = [(shuffle_log(log) for log in iter(atl)) for atl in self.atls.values()]
        random.shuffle(generators)
        has_next = True
        while has_next:
            has_next = False
            for generator in generators:
                item = next(generator, False)
                if item is not False:
                    has_next = True
                    yield item

    def create_gt_parts(self, root=None):
        if root is None:
            root = Path(self.root_path)
        for idx, lidar, label, log in iter(self):
            save_path = root / log.current_log / 'gt_parts'
            save_path.mkdir(parents=True, exist_ok=True)

            gt_boxes = np.zeros((len(label, 7)))
            for i, obj in enumerate(label):
                loc = obj.translation
                quat = obj.quaternion
                dim = (obj.width, obj.length, obj.height)
                rot = R.from_quat(quat).as_euler('zyx')
                gt_boxes[i] = np.hstack((loc, dim, rot[0]))

            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(torch.from_numpy(lidar[:, :3]), torch.from_numpy(gt_boxes)).numpy()

            for i, obj in enumerate(label):
                filename = save_path / '{}_{}_{}.bin'.format(idx, obj.label_class, obj.track_id)
                gt_points = lidar[point_indices[i] > 0]
                gt_points -= gt_points.mean(axis=0)

                with open(filename, 'wb') as f:
                    gt_points.tofile(f)


class ArgoDataset(BaseArgoDataset):
    def __init__(self, root_path, subsets:list, class_names:dict, training=True):
        """
        :param root_path: ARGO AI data path
        :param split:
        """
        super().__init__(root_path, subsets)
        self.class_names = class_names
        self.training = training
        self.mode = 'TRAIN' if self.training else 'TEST'

        # Support spconv 1.0 and 1.1
        try:
            VoxelGenerator = spconv.utils.VoxelGeneratorV2
        except:
            VoxelGenerator = spconv.utils.VoxelGenerator
        
        vg_cfg = cfg.DATA_CONFIG.VOXEL_GENERATOR
        self.voxel_generator = VoxelGenerator(
            voxel_size=vg_cfg.VOXEL_SIZE,
            point_cloud_range=vg_cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
            max_num_points=vg_cfg.MAX_POINTS_PER_VOXEL,
            max_voxels=cfg.DATA_CONFIG[self.mode].MAX_NUMBER_OF_VOXELS
        )
        pass
    
    def __getitem__(self, index):
        def create_input_dict(log, subset, idx):
            label = []
            for obj in log.get_label_object(idx):
                if obj.label_class in self.class_names.keys():
                    obj.class_id = self.class_names[obj.label_class]
                    label.append(obj)

            points = log.get_lidar(idx)
            gt_boxes = np.zeros((len(label), 7))
            occluded = np.zeros(len(label), dtype=int)
            for i, obj in enumerate(label):
                loc = obj.translation
                quat = obj.quaternion
                dim = (obj.width, obj.length, obj.height)
                rot = R.from_quat(quat).as_euler('zyx')
                gt_boxes[i] = np.hstack((loc, dim, rot[0], obj.class_id))
                occluded[i] = obj.occlusion

            voxel_grid = self.voxel_generator.generate(points)
            if isinstance(voxel_grid, dict):
                voxels = voxel_grid["voxels"]
                coordinates = voxel_grid["coordinates"]
                num_points = voxel_grid["num_points_per_voxel"]
            else:
                voxels, coordinates, num_points = voxel_grid
            
            voxel_centers = (coordinates[:, ::-1] + 0.5) * self.voxel_generator.voxel_size + self.voxel_generator.point_cloud_range[:3]

            return {
                'voxels': voxels,
                'voxel_senters': voxel_centers,
                'coordinates': coordinates,
                'num_points': num_points,
                'points': points,
                'subset': subset,
                'sample_idx': idx,
                'occluded': occluded,
                'gt_names': np.array([obj.label_class for obj in label]),
                'gt_box2d': None,
                'gt_boxes': gt_boxes
            }

        for subset, atl in self.atls.items():
            for log in iter(atl):
                if index < log.num_lidar_frame:
                    input_dict = create_input_dict(log, subset, index)
                    break
                else:
                    index -= log.num_lidar_frame
        return input_dict


def create_argo_infos(data_path, save_path, subsets, workers=4):
    dataset = BaseArgoDataset(data_path, subsets)

    #print('---------------Start to generate data infos---------------')
    #for subset in subsets:
    #    filename = save_path / subset / 'argo_infos.pkl'
    #
    #    argo_infos = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    #    with open(filename, 'wb') as f:
    #        pickle.dump(argo_infos, f)
    #    print('ArgoAI info {} file is saved to {}'.format(subset, filename))

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.create_gt_parts(save_path)
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Generates a database of Parts')
    parser.add_argument('data_path', help='root path of the dataset')
    parser.add_argument('save_path', help='path for saving the parts')
    parser.add_argument('--subsets', nargs='+', default=['train1','train2','train3','train4'], help='List of database subsets')
    args = parser.parse_args()

    create_argo_infos(args.data_path, args.save_path, args.subsets)