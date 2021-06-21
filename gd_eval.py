"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
"""

from json.decoder import JSONDecodeError
import os
import json

import torch
from tqdm import tqdm
import numpy as np

from data import *
from data.transforms import *
from data.gdgrid import *
from utils.augmentations import SSDAugmentation
from utils.util_init import *
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os

from train_utils import get_coco_api_from_dataset, CocoEvaluator


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def main(parser_data):
    device = torch.device(parser_data.use_cuda if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = {
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # read class_indict
    label_json_path = './dianwang_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}

    dataset_root = parser_data.data_path
    """
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))
    """

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # load validation data set
    val_dataset = DwDataset(dataset_root, data_transform["val"], "val.txt")
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=nw,
                                                     pin_memory=True,
                                                     collate_fn=val_dataset.collate_fn)

    # val_dataset = DwDataset(args.dataset_root,
    #                         compose_transforms,
    #                         "val.txt")
    # val_dataset_loader = data.DataLoader(valset, args.batch_size,
    #                             #   num_workers=args.num_workers,
    #                               shuffle=False, 
    #                               collate_fn=valset.collate_fn,
    #                               pin_memory=False)
    # create model num_classes equal background + 20 classes
    # 注意，这里的norm_layer要和训练脚本中保持一致
    num_classes = len(category_index) + 1                      # +1 for background
    model = build_ssd('test', 300, num_classes)            # initialize SSD

    # 载入你自己训练好的模型权重

    model.load_state_dict(torch.load(args.trained_model))
    print('Finished loading model!')
    
    # print(model)

    model.to(device)

    # evaluate on the test dataset
    coco = get_coco_api_from_dataset(val_dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    cpu_device = torch.device("cpu")

    model.eval()
    with torch.no_grad():
        for image, targets in tqdm(val_dataset_loader, desc="validation..."):
            if args.use_cuda and torch.cuda.is_available():
                image = Variable(image.cuda())
                # targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            else:
                image = Variable(image)
                # targets = [Variable(ann, volatile=True) for ann in targets]
            # 将图片传入指定设备device
            # image = list(img.to(device) for img in image)

            # inference
            print('+++++++++++', image.size())
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            
            
            for i in range(len(outputs)):
                jsontext = {'image_id': targets[i]['image_id'].item(), 'objs':[]}
                for j in range(len(outputs[i]["labels"])):
                    bbox, label, score = outputs[i]["boxes"][j], outputs[i]["labels"][j], outputs[i]["scores"][j]
                    obj_dict = {"label":label.numpy().tolist(), "bbox":bbox.numpy().tolist(), "score":score.numpy().tolist()}
                    jsontext["objs"].append(obj_dict)
                jsondata = json.dumps(jsontext, indent=4, separators=(',', ': '))
                with open("./pred_results/img" + str(targets[i]['image_id'].item()) + ".json",'w') as f:
                    f.write(jsondata)
            
            coco_evaluator.update(res)
    

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_eval = coco_evaluator.coco_eval["bbox"]
    # calculate COCO info for all classes
    coco_stats, print_coco = summarize(coco_eval)

    # calculate voc info for every classes(IoU=0.5)
    voc_map_info_list = []
    for i in range(len(category_index)):
        stats, _ = summarize(coco_eval, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))

    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)

    # 将验证结果保存至txt文件中
    with open("record_mAP.txt", "w") as f:
        record_lines = ["COCO results:",
                        print_coco,
                        "",
                        "mAP(IoU=0.5) for each category:",
                        print_voc]
        f.write("\n".join(record_lines))

def reinfer(img):
    boxes = img["boxes"]
    labels = img["labels"]
    scores = img["scores"]

    keep = {}
    
    bage_idxs = torch.where(labels==1)[0]
    offground_idxs = torch.where(labels==2)[0]
    ground_idxs = torch.where(labels==3)[0]
    safebelt_idxs = torch.where(labels==4)[0]

    # offground ground 互斥
    for offground_idx in offground_idxs:
        for ground_idx in ground_idxs:
            offground_box = boxes[offground_idx.item()]
            ground_box = boxes[ground_idx.item()]
            

        
        # count iou
        # if iou 
        pass

def get_iou(box1, score1, box2, score2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    
            


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 使用设备类型
    parser.add_argument('--use_cuda', default='cuda', help='device')

    # # 检测目标类别数
    # parser.add_argument('--num-classes', type=int, default='20', help='number of classes')

    # 数据集的根目录(VOCdevkit)
    parser.add_argument('--data-path', default='/home/qingren/Project/Tianchi_dw/Dataset', help='dataset root')

    # 训练好的权重文件
    parser.add_argument('--trained_model', default='checkpoints/model-70.pkl', type=str, help='training weights')

    # batch size
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='batch size when validation.')

    args = parser.parse_args()

    main(args)
