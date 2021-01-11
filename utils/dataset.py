from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import torchvision.transforms as transforms

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) 
                    if not file.startswith('.')]
        #logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.749, 0.556, 0.656], std=[0.166, 0.203, 0.153]),
            transforms.ToPILImage()
        ])
        if type(pil_img) == np.ndarray:
            w = pil_img.shape[0]
            h = pil_img.shape[1]  
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small'
            np.resize(pil_img.shape[0], newW)
            np.resize(pil_img.shape[1], newH) 
        else:   
            w, h = pil_img.size
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small'
            pil_img = pil_img.resize((newW, newH))
       
            
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 3:
            img_trans = img_trans / 255
            #img_trans = normalize(img_nd)
            #img_trans = np.array(img_trans)
            #img_trans = img_trans.transpose((2, 0, 1))
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = []
        img_file = []
        mask_file.append(self.masks_dir + idx + '.bmp')
        img_file.append(self.imgs_dir + idx + '.jpg')
        #mask_file = glob(self.masks_dir + idx + '.jpg')
        #img_file = glob(self.imgs_dir + idx + '*')
        #assert len(mask_file) == 1, \
            #f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        #assert len(img_file) == 1, \
            #f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        #assert img.size == mask.size, \
            #f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}


class UnlabeledDataset(Dataset):
    def __init__(self, imgs_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.733, 0.577, 0.673], std=[0.180, 0.220, 0.168]),
            transforms.ToPILImage()
            ])

        if type(pil_img) == np.ndarray:
            w = pil_img.shape[0]
            h = pil_img.shape[1]
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small'
            np.resize(pil_img.shape[0], newW)
            np.resize(pil_img.shape[1], newH)
        else:
            w, h = pil_img.size
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small'
            pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 3:
            img_trans = img_trans / 255
            #img_trans = normalize(img_nd)
            #img_trans = np.array(img_trans)
            #img_trans = img_trans.transpose((2, 0, 1))
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = []
        img_file.append(self.imgs_dir + idx + '.jpg')
        #assert len(img_file) == 1, \
            #f'Either no image or multiple images found for the ID {idx}: {img_file}'
        img = Image.open(img_file[0])
        img = self.preprocess(img, self.scale)

        return {'image': torch.from_numpy(img)}