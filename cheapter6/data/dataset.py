# coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
#
# root = 'data/train/'
# dataset = os.listdir(root)
# imgs = [os.path.join(root,img) for img in dataset]
# imgs1 = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
#
# root = 'data/test1/'
# dataset = os.listdir(root)
# imgs = [os.path.join(root,img) for img in dataset]
# imgs2 = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))

class DogCat(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg 
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        np.random.seed(2019)
        ims  = np.random.permutation(imgs)

        if self.test:
            # self.imgs = imgs
            self.imgs = imgs[0:100]
        elif train:
            # self.imgs = imgs[:int(0.7 * imgs_num)]
            self.imgs = imgs[:1000]
        else:
            # self.imgs = imgs[int(0.7 * imgs_num):]
            self.imgs = imgs[1000:1100]

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(150),
                    T.CenterCrop(150),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.RandomSizedCrop(150),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
