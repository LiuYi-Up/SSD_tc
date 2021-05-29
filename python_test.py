import torch

coco = [1, 2, 3, 4]
voc = [9, 8, 7, 6]
num = 1
b = coco + voc
a = (coco, voc)[num == 21]
print(b)
      