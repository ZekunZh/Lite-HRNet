import time
from pathlib import Path
from typing import Union, List, Dict

import mmcv
import torch
import torchvision
import warnings

from mmcv.runner import EpochBasedRunner, master_only, get_host_info
from mmcv.runner.builder import RUNNERS
from torch.utils.data import DataLoader

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

    def run(
        self,
        data_loaders: Union[List["DataLoader"], Dict[str, List["DataLoader"]]],
        workflow,
        max_epochs=None,
        **kwargs
    ):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`], dict[str, list[:obj:`DataLoader`]):
                Dataloaders for training and validation.

                Example: [train_dataloader, val_dataloader], {"train": [loader1, loader2, ...]}

            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, (list, dict))
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        if isinstance(data_loaders, list):
            assert len(data_loaders) == len(workflow)
        elif isinstance(data_loaders, dict):
            for flow in workflow:
                assert flow[0] in data_loaders
                assert len(data_loaders[flow[0]]) > 0
            logger_str = "Dataloaders in dict: \n\t"
            for mode, loaders in data_loaders.items():
                assert isinstance(loaders, list)
                logger_str += f"{mode}: {len(loaders)}\n\t"
            self.logger.info(logger_str)


        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs
        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i]) if isinstance(data_loaders, list) else self._max_epochs * len(data_loaders[mode][0])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break

                    if isinstance(data_loaders, list):
                        epoch_runner(data_loaders[i], **kwargs)
                    elif isinstance(data_loaders, dict):
                        data_loader_idx = self.epoch % len(data_loaders[mode])
                        self.logger.info(f"Running {mode} epoch {self.epoch} with {data_loader_idx}th {mode} dataloader")
                        epoch_runner(data_loaders[mode][data_loader_idx], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')



