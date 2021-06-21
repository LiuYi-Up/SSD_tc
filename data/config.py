# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

GD_CLASSES = ['监护袖章(红only)', 'offground', 'ground', 'safebelt'] 

# SSD300 CONFIGS

gdgrid = {
    'num_classes': 5,  # 分类个数+背景
    'lr_steps': (80000, 100000, 120000),  # 学习速率改变周期
    'max_iter': 1000,  # ？？？
    'feature_maps': [38, 19, 10, 5, 3, 1], # 特征图层尺寸
    'min_dim': 300,  # 输入尺寸300*300
    'steps': [8, 16, 32, 64, 100, 300],  # ？？？
    'min_sizes': [21, 45, 99, 153, 207, 261],  # 
    'max_sizes': [45, 99, 153, 207, 261, 315],  # 
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]], # 边框比例，比例1不含在这里
    'variance': [0.1, 0.2],  # ？？？
    'clip': True,  # ？？？
    'name': 'GDGRID'
}