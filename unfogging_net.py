import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import DIP


def unfog_image(image_path, unfog_net):

	data_foggy = Image.open(image_path)
	data_foggy = (np.asarray(data_foggy) / 255.0)

	data_foggy = torch.from_numpy(data_foggy).float()
	data_foggy = data_foggy.permute(2, 0, 1)
	data_foggy = data_foggy.cuda().unsqueeze(0)

	
	clean_image = unfog_net(data_foggy)

	Image.fromarray((clean_image.cpu().detach().squeeze().permute((1, 2, 0)).numpy()*255).astype(np.uint8)).save(f"results/{image_path}")
	

if __name__ == '__main__':
	unfog_times = []
	unfog_net = net.unfog_net().cuda()
	unfog_net.load_state_dict(torch.load('snapshots/net.pth'))

	dip_times = []
	DIP_model = torch.load("DIP_Model_Best.pth")
	to_tensor = transforms.ToTensor()

	test_list = glob.glob("test_images/*")	
	start_time = time.time()
	for image in test_list:
		unfog_image(image, unfog_net)
	unfog_times.append(time.time()-start_time)
	
	start_time = time.time()
	for image in test_list:
		DIP_model(to_tensor(Image.open(image)).unsqueeze(0).cuda())
	dip_times.append(time.time()-start_time)

	print(np.mean(unfog_times))
	print(np.mean(dip_times))