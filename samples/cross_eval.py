import argparse
import commentjson as json
import numpy as np
import os
import sys
import torch
import time
import h5py
from datetime import datetime
import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant('cuda_ad_rgb')


def save_gt(data):
    with h5py.File(data, 'r') as hdf:
        keys = list(hdf.keys())
        view = hdf[keys[0]][:]
        location = hdf[keys[1]][:]
        color = hdf[keys[2]][:]
        light = hdf[keys[3]][:]

    numdir = color.shape[0]
    ynum = color.shape[1]
    xnum = color.shape[2]
    with torch.no_grad():
        # for index in range(0, numdir, gap):
        for index in range(0, 10):
            curcolor = color[index]
            curcolor = mi.Bitmap(curcolor).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32)
            filename = savedir + 'color_' +str(index)+ '_gt.exr'
            curcolor.write(filename)

def evaluate(data, model, method, savedir, reparam=False, jacobian_file='leather_04_jacobian.npy'):
    with h5py.File(data, 'r') as hdf:
        keys = list(hdf.keys())
        view = hdf[keys[0]][:]
        location = hdf[keys[1]][:]
        color = hdf[keys[2]][:]
        light = hdf[keys[3]][:]

    if reparam:
        jacobian = np.load('../BTFdata/'+jacobian_file)

    numdir = color.shape[0]
    ynum = color.shape[1]
    xnum = color.shape[2]

    # evaluation
    with torch.no_grad():
        #for index in range(0, numdir, gap):
        for index in range(0, 10):
            curlocation = location[index].reshape(-1, 2)
            curlight = light[index].reshape(-1, dir_dim)
            curview = view[index].reshape(-1, dir_dim)
            curinput = np.concatenate((curlocation, curlight, curview), axis=-1)
            curinput = torch.tensor(curinput, device=device, dtype=torch.float32)
            curoutput = model(curinput)
            curoutput = yuv_to_rgb(curoutput)
            curoutput = torch.exp(curoutput) - 1

            curoutput = curoutput.reshape(ynum, xnum, n_channels).clamp(0.0).detach().cpu().numpy()
            if reparam:
                curoutput = curoutput / jacobian[index][:,:,None]
            curoutput = np.nan_to_num(curoutput)
            curoutput = mi.Bitmap(curoutput).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32)
            filename = savedir + 'color_' +str(index)+ '_' + method + '_pred'
            if reparam:
                filename += '_reparam'
            filename = filename + '.exr'
            curoutput.write(filename)


def yuv_to_rgb(x):

	x_r = x[:,0] + 1.13983*x[:,2]
	x_g = x[:,0] - 0.39465*x[:,1] - 0.58060*x[:,2]
	x_b = x[:,0] + 2.03211*x[:,1]
	
	x = torch.stack((x_r, x_g, x_b), dim=-1)

	return x

def get_args():
	parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

	parser.add_argument("--data", default="BTFdata/leather_04_anglenum5.hdf5", help="Image to match")
	parser.add_argument("--config", default="data/config_hash_naive.json", help="JSON config for tiny-cuda-nn")
	parser.add_argument("--n_steps", type=int, default=50000, help="Number of training steps")
	parser.add_argument("--batch_size", type=int, default=15, help="power of 2 batch size") # 10 is good, 19 diverges
	parser.add_argument("--interval", type=int, default=100, help="interval for printing and saving results")
	parser.add_argument("--prefix", default="leather_synthetic", help="prefix for saving results")
	parser.add_argument("--method", default="naive", help="method for computing hash")
	parser.add_argument("--gap", type=int, default=50, help="gap for saving results")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
    print("=======================================================================")
    print("This script evaluates the tiny-cuda-nn results using unseen directions.")
    print("=======================================================================")

    print(f"Using PyTorch version {torch.__version__} with CUDA {torch.version.cuda}")

    device = torch.device("cuda")
    args = get_args()
    prefix = args.prefix
    method = args.method
    gap = args.gap
    dir_dim = 2
    n_channels = 3

    # directories for saving results
    today = datetime.now()
    todaystr = today.strftime('%Y%m%d')
    savedir =  todaystr + "/evaluate/"
    print("savedir: ", savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # load naive model
    filename = '../20240102/leather_04_real_factor3/naive_hash/result/leather_04_real_factor3_naive_hash.pth'
    model = torch.load(filename).to(device)

    data = '../BTFdata/real_leather_04_factor3_retro.hdf5'

    # save gt retro data
    save_gt(data)

    # eval naive on retro data
    evaluate(data, model, method, savedir)

    # load reparam
    filename = '../20240102/leather_04_real_factor3/reparam_hash/result/leather_04_real_factor3_reparam_hash.pth'
    model = torch.load(filename).to(device)

    # eval reparam on retro data
    data = '../BTFdata/real_leather_04_reparam_simple_factor3_retro.hdf5'
    reparam = True
    jacobian_file='leather_04_jacobian.npy'
    evaluate(data, model, method, savedir, reparam, jacobian_file)
    