import random
from torchvision.transforms import functional as F


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target[:, :4]
            # print('boxes', bbox)

            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            # print('newbbox', bbox)
            target[:, :4] = bbox
            # print('target', target)
        return image, target

class Resize(object):
    def __init__(self, ispad:bool=True, size:int=300):
        self.ispad = ispad
        self.size = size

    def __call__(self, image, target):
        if self.ispad:
            ratio = self.size/max(image.size)
            img1 = F.resize(image, int(min(image.size)*ratio))
            image = F.pad(img1, [0, 0, self.size - img1.size[0], self.size - img1.size[1]])
            target[:, :4] = target[:, :4]*ratio
        else:
            w_ratio, h_ratio = self.size/image.size[1], self.size/image.size[0]
            image = F.resize(image, (self.size, self.size))
            target[:, [0, 2]] = target[:, [0, 2]]*h_ratio
            target[:, [1, 3]] = target[:, [1, 3]]*w_ratio
        return image, target




