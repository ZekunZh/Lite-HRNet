import time
from pathlib import Path

import torch
import torchvision

from mmcv.runner import EpochBasedRunner, master_only
from mmcv.runner.builder import RUNNERS

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

@RUNNERS.register_module()
class GleamerEpochBasedRunner(EpochBasedRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dir = str(Path(self.work_dir) / 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log_image(self, data_batch):

        input_images = data_batch["img"]    # (N, 3, H, W)
        # sum heatmaps of all keypoints into one image
        target_heatmaps = data_batch["target"]
        target_heatmaps = torch.sum(target_heatmaps, dim=1, keepdim=True) # (N, 1, H_hm, W_hm)

        # de-normalize input images to get original images
        inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        ])
        input_images = inverse_transform(input_images)
        input_images = input_images[:, :1, :, :] # (N, 1, H, W)

        # normalize target heatmap between [0, 1]
        batch_size, _, height, width = target_heatmaps.shape
        target_heatmaps = target_heatmaps.view(batch_size, -1)
        target_heatmaps -= target_heatmaps.min(1, keepdim=True)[0]
        target_heatmaps /= (target_heatmaps.max(1, keepdim=True)[0] + 1.e-10)
        target_heatmaps = target_heatmaps.view(batch_size, 1, height, width)  # (N, 1, H, W)

        img_grid = torchvision.utils.make_grid(input_images, nrow=2)
        heatmap_grid = torchvision.utils.make_grid(target_heatmaps, nrow=2)

        self.writer.add_image(f"input/epoch_{self.epoch}", img_grid)
        self.writer.add_image(f"target/epoch_{self.epoch}", heatmap_grid)

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook("before_train_epoch")
        time.sleep(2)
        for i, data_batch in enumerate(self.data_loader):
            # Plot input image & target heatmap for the first batch of each epoch
            if i == 0:
                self.log_image(data_batch)
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1



