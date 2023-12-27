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


    # evaluation
	location = location.reshape(numdir, ynum, xnum, 2)
	light = light.reshape(numdir, ynum, xnum, dir_dim)
	view = view.reshape(numdir, ynum, xnum, dir_dim)
	with torch.no_grad():
		for index in range(0, numdir, gap):
			curlocation = location[index].reshape(-1, 2)
			curlight = light[index].reshape(-1, dir_dim)
			curview = view[index].reshape(-1, dir_dim)
			curinput = np.concatenate((curlocation, curlight, curview), axis=-1)
			curinput = torch.tensor(curinput, device=device, dtype=torch.float32)
			curoutput = model(curinput)

			curoutput_aux = curoutput[:, n_channels:]
			curoutput = curoutput[:, :n_channels]
			curoutput = yuv_to_rgb(curoutput)
			curoutput = torch.exp(curoutput) - 1

			curoutput = curoutput.reshape(ynum, xnum, n_channels).clamp(0.0).detach().cpu().numpy()
			curoutput = mi.Bitmap(curoutput).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32)
			filename = image_dir + 'color_' +str(index)+ '_' + method + '_pred.exr'
			curoutput.write(filename)