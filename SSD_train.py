# from torch._C import per_tensor_affine
from data import *
from data.transforms import *
from data.gdgrid import *
from utils.augmentations import SSDAugmentation
from utils.util_init import *
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from utils.options import parse_args_function


args = parse_args_function()

"""# Load Dataset"""

compose_transforms = Compose([Resize(ispad=False),
                                      ToTensor(),
                                      RandomHorizontalFlip()])
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if args.train:
    # trainset = Dataset(root=root, load_set='train', transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    trainset = DwDataset(args.dataset_root,
                            compose_transforms,
                            "train.txt")
    trainloader = data.DataLoader(trainset, args.batch_size,
                                #   num_workers=args.num_workers,
                                  shuffle=True, collate_fn=trainset.collate_fn,
                                  pin_memory=False)
    print('Train files loaded')

if args.val:
    # valset = Dataset(root=root, load_set='val', transform=transform)
    # valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    valset = DwDataset(args.dataset_root,
                            compose_transforms,
                            "val.txt")
    valloader = data.DataLoader(valset, args.batch_size,
                                #   num_workers=args.num_workers,
                                  shuffle=False, collate_fn=valset.collate_fn,
                                  pin_memory=False)
    print('Validation files loaded')

if args.test:
    # testset = Dataset(root=root, load_set='test', transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    testset = DwDataset(args.dataset_root,
                            compose_transforms,
                            "test.txt")
    testloader = data.DataLoader(testset, args.batch_size,
                                #   num_workers=args.num_workers,
                                  shuffle=False, collate_fn=testset.collate_fn,
                                  pin_memory=False)
    print('Test files loaded')


cfg = gdgrid

"""# Model"""

use_cuda = False
if args.cuda:
    use_cuda = True

ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
net = ssd_net

if use_cuda and torch.cuda.is_available():
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

"""# Load Snapshot"""

if args.pretrained_model != '':
    ssd_net.load_state_dict(torch.load(args.pretrained_model))
    losses = np.load(args.pretrained_model[:-4] + '-losses.npy').tolist()
    start = len(losses)
else:
    print('Loading base network...')
    vgg_weights = torch.load(args.basenet)
    ssd_net.vgg.load_state_dict(vgg_weights)
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)
    losses = []
    start = 0

if args.cuda:
    net = net.cuda()

"""# Optimizer"""

# criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay)
criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
scheduler.last_epoch = start

"""# Train"""

if args.train:
    print('Begin training the network...')
    for epoch in range(start, args.num_iterations):  # loop over the dataset multiple times
    
        net.train()
        running_loss = 0.0
        train_loss = 0.0
        for i, tr_data in enumerate(trainloader):
            # get the inputs
            images, targets = tr_data

            # wrap them in Variable
            if use_cuda and torch.cuda.is_available():
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]
            
            t0 = time.time()
    
            # forward + backward + optimize
            out = net(images)
            # print('-------------',(len(out[0]), len(out[1]), len(out[2])))
            # a = out[1].cpu().detach().numpy()
            print('---------------------', out.size(1))
            

            # zero the parameter gradients
            optimizer.zero_grad()

            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            
            t1 = time.time()
            # print statistics
            running_loss += loss.data
            train_loss += loss.data
            if (i+1) % args.log_batch == 0:    # print every log_iter mini-batches
                print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / args.log_batch))
                running_loss = 0.0
                
        if args.val and (epoch+1) % args.val_epoch == 0:
            net.val()
            val_loss = 0.0
            for v, val_data in enumerate(valloader):
                # get the inputs
                images, targets = tr_data
    
                # wrap them in Variable
                if use_cuda and torch.cuda.is_available():
                    images = images.float().cuda(device=args.num_workers[0])
                    targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
                else:
                    images = Variable(images)
                    targets = [Variable(ann, volatile=True) for ann in targets]
                
                t0 = time.time()
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                out = net(images)

                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                val_loss += loss.data
            print('val error: %.5f' % (val_loss / (v+1)))
        losses.append((train_loss / (i+1)).cpu().numpy())
        
        if (epoch+1) % args.snapshot_epoch == 0:
            torch.save(ssd_net.state_dict(), args.output_file+str(epoch+1)+'.pkl')
            np.save(args.output_file+str(epoch+1)+'-losses.npy', np.array(losses))

        # Decay Learning Rate
        scheduler.step()
    
    print('Finished Training')

"""# Test"""

if args.test:
    print('Begin testing the network...')
    
    running_loss = 0.0
    for i, ts_data in enumerate(testloader):
        # get the inputs
        images, targets = tr_data
    
        # wrap them in Variable
        if use_cuda and torch.cuda.is_available():
            images = images.float().cuda(device=args.num_workers[0])
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        
        t0 = time.time()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        out = net(images)

        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        running_loss += loss.data
    print('test error: %.5f' % (running_loss / (i+1)))
