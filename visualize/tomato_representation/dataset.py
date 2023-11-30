# coding=utf-8
# Copyright 2022 The IDEA Authors (Shunlin Lu and Ling-Hao Chen). All rights reserved.
#
# For all the datasets, be sure to read and follow their license agreements,
# and cite them accordingly.
# If the unifier is used in your research, please consider to cite as:
#
# @article{humantomato,
#   title={HumanTOMATO: Text-aligned Whole-body Motion Generation},
#   author={Lu, Shunlin and Chen, Ling-Hao and Zeng, Ailing and Lin, Jing and Zhang, Ruimao and Zhang, Lei and Shum, Heung-Yeung},
#   journal={arxiv:2310.12978},
#   year={2023}
# }
#
# @InProceedings{Guo_2022_CVPR,
#     author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
#     title     = {Generating Diverse and Natural 3D Human Motions From Text},
#     booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#     month     = {June},
#     year      = {2022},
#     pages     = {5152-5161}
# }
#
# Licensed under the IDEA License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/IDEA-Research/HumanTOMATO/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. We provide a license to use the code, 
# please read the specific details carefully.

import os
from torch.utils import data
from tqdm import tqdm
import numpy as np
import torch


def findAllFile(base, debug=False):
    """
    Recursively find all files in the specified directory.

    Args:
        base (str): The base directory to start the search.

    Returns:
        list: A list of file paths found in the directory and its subdirectories.
    """
    print("Searching for files in {}...".format(base))
    file_path = []

    # Limiting the number of files for debugging purposes
    if debug:
        i = 1
        for root, ds, fs in os.walk(base, followlinks=True):
            for f in fs:
                fullname = os.path.join(root, f)
                file_path.append(fullname)
                i += 1
                if i > 100:
                    return file_path
                
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


def collate_tensors(batch):
    # Function for collating a batch of PyTorch tensors
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def mld_collate(batch):
    # Adapter function for collating batches in the MotionDatasetV2 class
    notnone_batches = [b for b in batch if b is not None]
    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "name": [b[1] for b in notnone_batches],
        "length":
        collate_tensors([torch.tensor(b[2]).float() for b in notnone_batches]),
    }

    return adapted_batch


class MotionDatasetV2(data.Dataset):
    # Custom dataset class for motion data
    def __init__(self, root_path, debug):

        # Lists to store motion data and corresponding lengths
        self.data = []
        self.lengths = []

        # Finding all files in the specified directory
        self.id_list = findAllFile(root_path, debug=debug)

        # Limiting the number of files for debugging purposes
        if debug:
            self.id_list = self.id_list[:100]

        # Loading motion data from files and populating data and lengths lists
        for name in tqdm(self.id_list):
            motion = np.load(name)
            self.lengths.append(motion.shape[0])
            self.data.append({'motion': motion, 'name': name})

    def __len__(self):
        # Returns the number of items in the dataset
        return len(self.id_list)

    def __getitem__(self, item):
        # Returns motion data, file name, and length for a given item

        motion = self.data[item]['motion']
        name = self.data[item]['name']
        length = self.lengths[item]

        return motion, name, length


class Text2MotionDataset(data.Dataset):
    # Custom dataset class for text-to-motion data
    def __init__(self, motions_path, texts_path, debug, type='train'):

        # Lists to store motion and text data and corresponding lengths
        self.data = []
        self.lengths = []

        # Finding all motions in motion_path
        if debug:
            self.motion_id_list = findAllFile(motions_path, debug=debug)
        else:
            if type == 'train':
                # Read the train list
                train_list = open("/home/apujol/mbld/datasets/MotionX/MotionX/datasets/train.txt").read().splitlines()
                self.motion_id_list = [os.path.join(motions_path, elem) for elem in train_list]
            elif type == 'val':
                # Read the val list
                val_list = open("/home/apujol/mbld/datasets/MotionX/MotionX/datasets/val.txt").read().splitlines()
                self.motion_id_list = [os.path.join(motions_path, elem) for elem in val_list]
            elif type == 'test':
                # Read the test list
                test_list = open("/home/apujol/mbld/datasets/MotionX/MotionX/datasets/test.txt").read().splitlines()
                self.motion_id_list = [os.path.join(motions_path, elem) for elem in test_list]
            elif type == 'all':
                # Read the test list
                all_list = open("/home/apujol/mbld/datasets/MotionX/MotionX/datasets/all.txt").read().splitlines()
                self.motion_id_list = [os.path.join(motions_path, elem) for elem in all_list]
            else:
                raise ValueError('Unsupported type name [{type}]')


        # Loading motion data from files and populating motions and lengths lists
        for motion_path in tqdm(self.motion_id_list):
            try: 
                # Get relative path of the motion file
                rel_path = os.path.relpath(motion_path, motions_path).split('.')[0]
                text_path = os.path.join(texts_path, rel_path + '.txt')
                motion = np.load(motion_path)
                text = open(text_path).read().splitlines()
                assert isinstance(text, list)
                name = motion_path.split('/')[-1].split('.')[0]
                # Check if is a HumanML3D motion 
                if rel_path.split('/')[0] == 'humanml':
                    # Split text on # and get the first element
                    for line in text:
                        line = line.split('#')[0]
                        assert isinstance(line, str)
                        tokens = w_tokenize(line)
                        self.lengths.append(motion.shape[0])
                        self.data.append({'motion': motion, 
                                        'motion_path': motion_path,
                                        'text': line,
                                        'text_path': text_path,
                                        'name': name})
                else:
                    assert len(text) == 1
                    text = text[0]
                    assert isinstance(text, str)
                    self.lengths.append(motion.shape[0])
                    self.data.append({'motion': motion, 
                                    'motion_path': motion_path,
                                    'text': text,
                                    'text_path': text_path,
                                    'name': name})
            except:
                print('Warning - Motion file {} not loaded.'.format(motion_path))
        

    def __len__(self):
        # Returns the number of items in the dataset
        return len(self.data)
    

    def __getitem__(self, item):
        # Returns motion data, file name, and length for a given item

        motion = self.data[item]['motion']
        motion_path = self.data[item]['motion_path']
        text = self.data[item]['text']
        text_path = self.data[item]['text_path']
        name = self.data[item]['name']
        length = self.lengths[item]

        return motion, motion_path, text, text_path, name, length


def t2m_collate(batch):
    # Adapter function for collating batches in the Text2MotionDataset class
    notnone_batches = [b for b in batch if b is not None]
    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "motion_path": [b[1] for b in notnone_batches],
        "text": [b[2] for b in notnone_batches],
        "text_path": [b[3] for b in notnone_batches],
        "name": [b[4] for b in notnone_batches],
        "length":
        collate_tensors([torch.tensor(b[5]).float() for b in notnone_batches]),
    }

    return adapted_batch