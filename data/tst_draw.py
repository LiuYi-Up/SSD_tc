import torch
import gdgrid
import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from datetime import datetime

np.random.seed(1)

def random_draw(dataset):
    now = datetime.now().time().strftime("%H_%M")
    num2label = {1:'badge', 2:'offground', 3:'ground', 4:'safebelt'}
    nums = np.random.randint(0, 2000, 9)
    
    fig = plt.figure(figsize=(15,15))
    for idx, num in enumerate(nums):
        ax = fig.add_subplot(3,3,idx+1)
        image, target = dataset[num]

        # 如果是 tensor 对通道顺序进行调整
        if torch.is_tensor(image):
            image = image.numpy()
            image = np.transpose(image, (1, 2, 0))
        
        # draw image
        ax.imshow(image)
        
        # draw bbox
        labels = target[:, 4]
        boxes = target[:, :4]
        for j in range(len(labels)):
            xmin,ymin,xmax,ymax = boxes[j]
            rect = patches.Rectangle((xmin, ymin),xmax-xmin,ymax-ymin,linewidth=1,edgecolor='r',facecolor='none')
            ax.text(xmin, ymin, num2label[labels[j]], c='red', 
                    bbox=dict(boxstyle='Square', fc='yellow', ec='k', lw=1, alpha=0.5))
            ax.add_patch(rect)
    
    plt.savefig("/home/qingren/Project/Ly_pro/SSD_v1/doc/figure" + now  + ".jpg")

if __name__ == "__main__":
    compose_transforms = transforms.Compose([transforms.Resize(ispad=False), 
                                            transforms.ToTensor(), 
                                            transforms.RandomHorizontalFlip()])

    dw_dataset = gdgrid.DwDataset("/home/qingren/Project/Tianchi_dw/Dataset",
                                compose_transforms,
                                'train.txt')
    random_draw(dw_dataset)



                  

