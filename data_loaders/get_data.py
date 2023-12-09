# Adaptation from Motion Diffusion Model code mixed with Stable Diffusion Model code.

#import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data_loaders.tensors import collate as all_collate
from data_loaders.motionx.data.utils import mx_collate, t2m_collate


def get_dataset_class(name):
    if name == 'motionx':
        from data_loaders.motionx.data.dataset import MotionX
        return MotionX
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    # if hml_mode == 'gt':
    #     from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
    #     return t2m_eval_collate
    if name == "motionx":
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


class DataModule(object):
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
        self.train = train
        self.val = val
        self.test = test
        if train:
            self.train_dataloader = self._train_dataloader
        else: 
            self.train_dataloader = None
        if val:
            self.val_dataloader = self._val_dataloader
        else: 
            self.val_dataloader = None
        if test:
            self.test_dataloader = self._test_dataloader
        else:
            self.test_dataloader = None
        self.debug = debug
    
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