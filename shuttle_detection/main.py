#coding:utf8
import torch
from models import FAN, FAN_light
from data.dataset import shuttle
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from skimage import measure
from utils import get_args, translate, visualize, heatmap, inference_video
import numpy as np 
import cv2
import os
from collections import OrderedDict


def test(epoch_num, save = False):
    global net, args, device
    net.eval()
    if epoch_num >= 0:
        state_dict = net.state_dict()
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if len(k) > 7 and k[:7] == 'module.':
                namekey = k[7:] # remove `module.`
            else:
                namekey = k
            new_state_dict[namekey] = v
        if args.model_mode == 'detaction':
            torch.save(new_state_dict, 
                os.path.join(args.result_path, 'epoch_%d.pkl'%(epoch_num))) 
        elif args.model_mode == 'tracking':
            torch.save(new_state_dict, 
                os.path.join(args.result_path, 'track_epoch_%d.pkl'%(epoch_num))) 
    
    if save:
        test_data =shuttle(args.data_path, args.neighbor, args.model_mode, train = False)
        test_dataloader = DataLoader(test_data, 1, shuffle=False)
        
        for num, (data, gt_heatmap) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            with torch.no_grad():
                data = data.to(device)
                result_heatmap = net(data)[0].cpu().data[0].numpy()
                #result_heatmap = gt_heatmap.data[0].numpy()

            points = heatmap.heatmap2points(result_heatmap[0].copy())
            #print(points)

            img = translate.torch2img(data.cpu().data[0, 3 * args.neighbor : 3 * args.neighbor + 3])
            img = visualize.label_points(img, points)
            result_heatmap = translate.np2heatmap(result_heatmap[0])
            result_heatmap = cv2.addWeighted(img, 0.6, result_heatmap, 0.4, 0)
            img = np.concatenate((img, result_heatmap), 0)
            cv2.imwrite(os.path.join(args.result_path, 'epoch%d_%d.jpg'%(epoch_num, num)), img)

    net.train()

    
def train():
    global net, args, device

    # step2: data 
    train_data = shuttle(args.data_path, args.neighbor, args.model_mode)
    train_dataloader = DataLoader(train_data, args.batch_size,
                        shuffle=True,num_workers=8)
    
    # step3: criterion and optimizer
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = args.LR)

    # train
    for epoch in range(1, args.epoch + 1):
        epoch_loss = 0
        print("EPOCH %d"%(epoch))

        for step, (data, heatmap) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):

            # train model 
            data = data.to(device)
            heatmap = heatmap.to(device)

            output = net(data)[0]
            loss = loss_func(output, heatmap)

            epoch_loss += loss.item()
            optimizer.zero_grad()  
            loss.backward()          
            optimizer.step() 

        print("loss is %.8f"%(epoch_loss))
        if epoch % args.checkpoint == 0:
            test(epoch, False)
            


if __name__ == '__main__':
    args = get_args()

    device = torch.device("cuda")
    
    net = FAN_light(mode = args.model_mode)
    if args.preload != None:
        state_dict = torch.load(args.preload)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if len(k) > 7 and k[:7] == 'module.':
                namekey = k[7:] # remove `module.`
            else:
                namekey = k
            new_state_dict[namekey] = v
        net.load_state_dict(new_state_dict)
    
    net = nn.DataParallel(net, device_ids=[0])
    net = net.to(device)

    if (args.mode == "train"):
        train()
    elif (args.mode == "test"):
        test(-1, True)
    elif (args.mode == "inference_video"):
        inference_video.inference_video(args.data_path, args.neighbor, net, device, args.result_path)
    else:
        print ('Please input the mode!')

