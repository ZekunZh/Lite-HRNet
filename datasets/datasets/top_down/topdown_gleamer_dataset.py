# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List

import os

import numpy as np
from mmcv import Config
from .topdown_coco_dataset import TopDownCocoDataset
from ...builder import DATASETS


@DATASETS.register_module(name="TopDownGleamerDataset")
class TopDownGleamerDataset(TopDownCocoDataset):
    """GleamerDataset for top-down pos estimation"""

    def __init__(
            self,
            ann_file,
            img_prefix,
            data_cfg,
            pipeline: List[dict],
            dataset_info: str = None,
            test_mode: bool = False
    ):
        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/topdown_gleamer_dataset.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode
        )

    def _get_db(self):
        """Load dataset"""
        if (not self.test_mode) or self.use_gt_bbox:
            gt_db = self._load_coco_keypoint_annotations()
        else:
            gt_db = self._load_gleamer_images()
        return gt_db

    def _load_gleamer_images(self):
        """Use bboxes that cover whole images"""
        bbox_db = []
        num_joints = self.ann_info["num_joints"]
        bbox_id = 0
        for img_id in self.img_ids:
            image_file = os.path.join(self.img_prefix, self.id2name[img_id])
            img_ann = self.coco.loadImgs(img_id)[0]
            width = img_ann["widht"]
            height = img_ann["height"]
            box = [0, 0, width, height]
            center, scale = self._xywh2cs(*box[:4])
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.ones((num_joints, 3), dtype=np.float32)

            bbox_db.append({
                "image_file"       : image_file,
                "center"           : center,
                "scale"            : scale,
                "rotation"         : 0,
                "bbox"             : box[:4],
                "bbox_score"       : 100,
                "dataset"          : self.dataset_name,
                "joints_3d"        : joints_3d,
                "joints_3d_visible": joints_3d_visible,
                "bbox_id"          : bbox_id
            })
            bbox_id += 1
        return bbox_db
