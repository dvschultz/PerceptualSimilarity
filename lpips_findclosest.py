import argparse
import lpips
import os
import pandas as pd 
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p','--path', type=str, default='', help='path to guidance file')
parser.add_argument('-f','--folder', type=str, default='', help='path to folder')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('-n','--net', type=str, default='alex', help='alex or vgg')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('--save_csv', action='store_true', help='save to CSV')
parser.add_argument('-o','--out', type=str, default='./out/', help='path to save files')
opt = parser.parse_args()

os.makedirs(opt.out, exist_ok=True)

data = []

## Initializing the model
loss_fn = lpips.LPIPS(net=opt.net,version=opt.version)

if(opt.use_gpu):
	loss_fn.cuda()

smallest_file = opt.path
smallest_dist = 3.0

img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.folder,smallest_file)))

files = os.listdir(opt.folder)

while (len(files) > 1):
	for (ff,file) in enumerate(files[:-1]):
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.folder,file))) # RGB image from [-1,1]

		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()

		# Compute distance
		dist01 = loss_fn.forward(img0,img1)
		# print('%s: %.3f'%(file,dist01))

		if((dist01!= 0.0) and (dist01 < smallest_dist)):
			smallest_dist = dist01.item()
			smallest_file = file

print(f'closest file: {smallest_file}: {smallest_dist}')

	








