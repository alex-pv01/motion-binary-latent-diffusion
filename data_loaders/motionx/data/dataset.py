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
import random

from data_loaders.motionx.data.utils import findAllFile
from data_loaders.motionx.data.text_tokenizer import TextTokenizer
from data_loaders.motionx.data.word_vectorizer import WordVectorizer

GLOVE_PATH = "/home/apujol/mbld/glove"

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

        # Minimum and maximum lengths of the motion data
        self.min_length = 10
        self.max_length = 500

        self.text_tokenizer = TextTokenizer()
        self.word_vectorizer = WordVectorizer(GLOVE_PATH, 'our_vab')

        self.max_text_length = 20

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
                # If the motion is too short or too long, go to exception
                if motion.shape[0] < self.min_length or motion.shape[0] > self.max_length:
                    print('Warning - Motion file {} has length {}.'.format(motion_path, motion.shape[0]))
                    raise Exception('Motion file {} has length {}.'.format(motion_path, motion.shape[0]))
                # If there are nan values go to exception
                if np.isnan(motion).any():
                    print('Warning - Motion file {} contains nan values.'.format(motion_path))
                    raise Exception('Motion file {} contains nan values.'.format(motion_path))
                text = open(text_path).read()
                #assert isinstance(text, list)
                name = motion_path.split('/')[-1].split('.')[0]
                # Check if is a HumanML3D motion 
                if rel_path.split('/')[0] == 'humanml':
                    text = text.splitlines()
                    for line in text:
                        line_split = line.split('#')
                        caption = line_split[0]
                        tokens = self.text_tokenizer.tokenize(caption)
                        assert isinstance(caption, str)
                        f_tag = 0.0 if np.isnan(float(line_split[2])) else float(line_split[2])
                        to_tag = 0.0 if np.isnan(float(line_split[3])) else float(line_split[3])
                        if f_tag == 0.0 and to_tag == 0.0:
                            self.lengths.append(motion.shape[0])
                            self.data.append({'motion': motion, 
                                            'motion_path': motion_path,
                                            'text': caption,
                                            'text_path': text_path,
                                            'tokens': tokens,
                                            'name': name})
                        else:
                            n_motion = motion[int(f_tag*20):int(to_tag*20)]
                            if n_motion.shape[0] < self.min_length or n_motion.shape[0] > self.max_length:
                                continue
                            else:
                                self.lengths.append(n_motion.shape[0])
                                self.data.append({'motion': n_motion, 
                                                'motion_path': motion_path,
                                                'text': caption,
                                                'text_path': text_path,
                                                'tokens': tokens,
                                                'name': name})
                else:
                    assert isinstance(text, str)
                    tokens = self.text_tokenizer.tokenize(text)
                    self.lengths.append(motion.shape[0])
                    self.data.append({'motion': motion, 
                                    'motion_path': motion_path,
                                    'text': text,
                                    'text_path': text_path,
                                    'tokens': tokens,
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
        tokens = self.data[item]['tokens']
        name = self.data[item]['name']
        m_length = self.lengths[item]

        if len(tokens) < self.max_text_length:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_length + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_length]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        # Vectorize text
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.word_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # print('motion.shape', motion.shape)
        if m_length < self.max_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_length - m_length, motion.shape[1], motion.shape[2]))
                                     ], axis=0)
            

        #return motion, motion_path, text, text_path, name, length, word_embeddings, pos_one_hots, sent_len, tokens
        return word_embeddings, pos_one_hots, text, sent_len, motion, m_length, '_'.join(tokens), name

