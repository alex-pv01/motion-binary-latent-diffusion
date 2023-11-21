# Adaptation from Motion Diffusion Model code mixed with Stable Diffusion Model code.

#import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate
from data_loaders.motionx.data.utils import mx_collate
from utils.utils import instantiate_from_config


def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name == 'motionx':
        from data_loaders.motionx.data.dataset import MotionX
        return MotionX
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name == "motionx":
        return mx_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    else:
        return all_collate


def get_dataset(name, 
                num_frames=None, 
                motions_path=None, 
                texts_path=None, 
                debug=True, 
                split='train', 
                hml_mode='train'):
    DATA = get_dataset_class(name)
    if name == "motionx":
        assert motions_path is not None and texts_path is not None, "motions_path and texts_path must be provided for MotionX dataset"
        dataset = DATA(motions_path, texts_path, debug, split=split)
    elif name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, 
                       batch_size,
                       num_frames=None,
                       motions_path=None,
                       texts_path=None, 
                       debug=True,
                       split='train', 
                       hml_mode='train',
                       ):
    dataset = get_dataset(name, num_frames, motions_path, texts_path, debug, split, hml_mode)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader


class DataModuleFromConfig(object):
    def __init__(self, 
                 batch_size, 
                 motions_path=None,
                 texts_path=None,
                 name=None, 
                 train=None, 
                 val=None, 
                 test=None, 
                 num_workers=None,
                 debug=True,
                 ):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.motions_path = motions_path
        self.texts_path = texts_path
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.train_dataloader = self._train_dataloader
        if val is not None:
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.test_dataloader = self._test_dataloader
        self.debug = debug

    # def prepare_data(self):
    #     for data_cfg in self.dataset_configs.values():
    #         instantiate_from_config(data_cfg)

    # def setup(self, stage=None):
    #     self.datasets = dict(
    #         (k, instantiate_from_config(self.dataset_configs[k])) 
    #         for k in self.dataset_configs
    #     )
    #     if self.wrap:
    #         for k in self.datasets:
    #             self.datasets[k] = WrappedDataset(self.datasets[k])
    
    def _train_dataloader(self):
        return get_dataset_loader(name=self.name,
                                  batch_size=self.batch_size,
                                  motions_path=self.motions_path,
                                  texts_path=self.texts_path,
                                  debug=self.debug,
                                  split='train',
                                  )
    def _val_dataloader(self):
        return get_dataset_loader(name=self.name,
                                  batch_size=self.batch_size//2,
                                  motions_path=self.motions_path,
                                  texts_path=self.texts_path,
                                  debug=self.debug,
                                  split='val',
                                  )
    def _test_dataloader(self):
        return get_dataset_loader(name=self.name,
                                  batch_size=self.batch_size//2,
                                  motions_path=self.motions_path,
                                  texts_path=self.texts_path,
                                  debug=self.debug,
                                  split='test',
                                  )