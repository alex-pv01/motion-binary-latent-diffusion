# Adaptation of the original code from Tomato representation mixed with Motion Diffusion Model code.
#
# Tomato license is:
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

from data_loaders.motionx.data.utils import findAllFile
from data_loaders.motionx.data.text_preprocess import TextPreprocess
from data_loaders.motionx.data.word_vectorizer import WordVectorizer


class MotionDataset(data.Dataset):
    # Custom dataset class for motion data
    def __init__(self, root_path, debug=100):

        # Lists to store motion data and corresponding lengths
        self.data = []
        self.lengths = []

        # Finding all files in the specified directory
        self.id_list = findAllFile(root_path, debug=debug)

        # Limiting the number of files for debugging purposes
        if debug:
            self.id_list = self.id_list[:debug]

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


class MotionX(data.Dataset):
    # Custom dataset class for text-to-motion data
    def __init__(self, motions_path, texts_path, debug=100, split='train'):

        # Lists to store motion and text data and corresponding lengths
        self.data = []
        self.lengths = []

        # Finding all motions in motion_path
        if debug:
            self.motion_id_list = findAllFile(motions_path, debug=debug)
        else:
            if split == 'train':
                # Read the train list
                train_list = open("/home/apujol/mbld/datasets/MotionX/MotionX/datasets/train.txt").read().splitlines()
                self.motion_id_list = [os.path.join(motions_path, elem + ".npy") for elem in train_list]
            elif split == 'val':
                # Read the val list
                val_list = open("/home/apujol/mbld/datasets/MotionX/MotionX/datasets/val.txt").read().splitlines()
                self.motion_id_list = [os.path.join(motions_path, elem + ".npy") for elem in val_list]
            elif split == 'test':
                # Read the test list
                test_list = open("/home/apujol/mbld/datasets/MotionX/MotionX/datasets/test.txt").read().splitlines()
                self.motion_id_list = [os.path.join(motions_path, elem + ".npy") for elem in test_list]
            elif split == 'all':
                # Read the test list
                all_list = open("/home/apujol/mbld/datasets/MotionX/MotionX/datasets/all.txt").read().splitlines()
                self.motion_id_list = [os.path.join(motions_path, elem + ".npy") for elem in all_list]
            else:
                raise ValueError('Unsupported split name [{split}]')


        # Loading motion data from files and populating motions and lengths lists
        for motion_path in tqdm(self.motion_id_list):
            try: 
                # Get relative path of the motion file
                rel_path = os.path.relpath(motion_path, motions_path).split('.')[0]
                text_path = os.path.join(texts_path, rel_path + '.txt')
                motion = np.load(motion_path)
                text = open(text_path).read()
                #assert isinstance(text, list)
                name = motion_path.split('/')[-1].split('.')[0]
                # Check if is a HumanML3D motion 
                if rel_path.split('/')[0] == 'humanml':
                    # Split text on # and get the first element
                    text = text.splitlines()
                    for line in text:
                        line = line.split('#')[0]
                        assert isinstance(line, str)
                        self.lengths.append(motion.shape[0])
                        self.data.append({'motion': motion, 
                                        'motion_path': motion_path,
                                        'text': line,
                                        'text_path': text_path,
                                        'name': name})
                else:
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


