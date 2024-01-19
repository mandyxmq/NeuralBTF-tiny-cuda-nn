#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# @file   mlp_learning_an_image_pytorch.py
# @author Thomas Müller, NVIDIA
# @brief  Replicates the behavior of the CUDA mlp_learning_an_image.cu sample
#         using tiny-cuda-nn's PyTorch extension. Runs ~2x slower than native.

import argparse
import commentjson as json
import numpy as np
import os
import sys
import torch
import time
import h5py
from datetime import datetime
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset
import mitsuba as mi
import numbers

mi.set_variant('cuda_ad_rgb')

try:
	import tinycudann as tcnn
except ImportError:
	print("This sample requires the tiny-cuda-nn extension for PyTorch.")
	print("You can install it by running:")
	print("============================================================")
	print("tiny-cuda-nn$ cd bindings/torch")
	print("tiny-cuda-nn/bindings/torch$ python setup.py install")
	print("============================================================")
	sys.exit()

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
sys.path.insert(0, SCRIPTS_DIR)


class BTFdataset(Dataset):
	def __init__(self, input, output, jacobian):
		super().__init__()
		self.input = input
		self.output = output
		self.jacobian = jacobian

	def __len__(self):
		return self.input.shape[0]

	def __getitem__(self, idx): 

		return self.input[idx], self.output[idx], self.jacobian[idx], idx
			
def yuv_to_rgb(x):

	x_r = x[:,0] + 1.13983*x[:,2]
	x_g = x[:,0] - 0.39465*x[:,1] - 0.58060*x[:,2]
	x_b = x[:,0] + 2.03211*x[:,1]
	
	x = torch.stack((x_r, x_g, x_b), dim=-1)

	return x

		
def query_image(image, xs):
	with torch.no_grad():
		# Bilinearly filtered lookup from the image. Not super fast,
		# but less than ~20% of the overall runtime of this example.
		shape = image.shape[0:2]

		xs = xs * torch.tensor([shape[1], shape[0]], device=xs.device).float()

		indices = xs.long()
		lerp_weights = xs - indices.float()

		x0 = indices[:, 0].clamp(min=0, max=shape[1]-1)
		y0 = indices[:, 1].clamp(min=0, max=shape[0]-1)
		x1 = (x0 + 1).clamp(max=shape[1]-1)
		y1 = (y0 + 1).clamp(max=shape[0]-1)

		return (
			image[y0, x0] * (1.0 - lerp_weights[:,0:1]) * (1.0 - lerp_weights[:,1:2]) +
			image[y0, x1] * lerp_weights[:,0:1] * (1.0 - lerp_weights[:,1:2]) +
			image[y1, x0] * (1.0 - lerp_weights[:,0:1]) * lerp_weights[:,1:2] +
			image[y1, x1] * lerp_weights[:,0:1] * lerp_weights[:,1:2]
		)
	

def query_images(images, xs):
	num_img = images.shape[0]
	shape = images.shape[1:3]
	n_pixels = shape[0] * shape[1]
	result = torch.zeros([xs.shape[0], 3], device=xs.device, dtype=torch.float32)

	with torch.no_grad():
		xs = xs * torch.tensor([shape[1], shape[0]], device=xs.device).float()

		for i in range(num_img):
			image = images[i]
			# Bilinearly filtered lookup from the image. Not super fast,
			# but less than ~20% of the overall runtime of this example.			

			indices = xs[i*n_pixels:(i+1)*n_pixels].long()
			lerp_weights = xs[i*n_pixels:(i+1)*n_pixels] - indices.float()

			x0 = indices[:, 0].clamp(min=0, max=shape[1]-1)
			y0 = indices[:, 1].clamp(min=0, max=shape[0]-1)
			x1 = (x0 + 1).clamp(max=shape[1]-1)
			y1 = (y0 + 1).clamp(max=shape[0]-1)

			result[i*n_pixels:(i+1)*n_pixels, :] = (
				image[y0, x0] * (1.0 - lerp_weights[:,0:1]) * (1.0 - lerp_weights[:,1:2]) +
				image[y0, x1] * lerp_weights[:,0:1] * (1.0 - lerp_weights[:,1:2]) +
				image[y1, x0] * (1.0 - lerp_weights[:,0:1]) * lerp_weights[:,1:2] +
				image[y1, x1] * lerp_weights[:,0:1] * lerp_weights[:,1:2]
			)

	return result


def query_jacobian(jacobians, xs):
	num_img = jacobians.shape[0]
	shape = jacobians.shape[1:3]
	n_pixels = shape[0] * shape[1]
	result = torch.zeros([xs.shape[0], 1], device=xs.device, dtype=torch.float32)

	with torch.no_grad():
		xs = xs * torch.tensor([shape[1], shape[0]], device=xs.device).float()

		for i in range(num_img):
			jac = jacobians[i]
			# Bilinearly filtered lookup from the image. Not super fast,
			# but less than ~20% of the overall runtime of this example.			

			indices = xs[i*n_pixels:(i+1)*n_pixels].long()
			lerp_weights = xs[i*n_pixels:(i+1)*n_pixels] - indices.float()

			x0 = indices[:, 0].clamp(min=0, max=shape[1]-1)
			y0 = indices[:, 1].clamp(min=0, max=shape[0]-1)
			x1 = (x0 + 1).clamp(max=shape[1]-1)
			y1 = (y0 + 1).clamp(max=shape[0]-1)

			result[i*n_pixels:(i+1)*n_pixels, :] = (
				jac[y0, x0] * (1.0 - lerp_weights[:,0:1]) * (1.0 - lerp_weights[:,1:2]) +
				jac[y0, x1] * lerp_weights[:,0:1] * (1.0 - lerp_weights[:,1:2]) +
				jac[y1, x0] * (1.0 - lerp_weights[:,0:1]) * lerp_weights[:,1:2] +
				jac[y1, x1] * lerp_weights[:,0:1] * lerp_weights[:,1:2]
			)

	return result

		
def nolow(x):
	x = x.clamp(-0.1)
	return torch.log1p(x)

def get_args():
	parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

	parser.add_argument("--data", default="BTFdata/real_sari_01_factor3.hdf5", help="data to match")
	parser.add_argument("--data2", default="", help="retro data")
	parser.add_argument("--test_data", default="BTFdata/real_sari_01_factor3.hdf5", help="data to test")
	parser.add_argument("--config", default="data/config_hash_naive.json", help="JSON config for tiny-cuda-nn")
	parser.add_argument("--n_steps", type=int, default=20000, help="Number of training steps")
	parser.add_argument("--batch_size", type=int, default=4, help="power of 2 batch size") # 10 is good, 19 diverges
	parser.add_argument("--interval", type=int, default=100, help="interval for printing and saving results")
	parser.add_argument("--prefix", default="sari_01", help="prefix for saving results")
	parser.add_argument("--method", default="naive", help="method for computing hash")
	parser.add_argument("--gap", type=int, default=50, help="gap for saving results")
	parser.add_argument("--gap2", type=int, default=20, help="gap for saving results")
	parser.add_argument("--gt", type=int, default=1, help="save ground truth")
	parser.add_argument("--n_levels", type=int, default=12, help="number of levels")
	parser.add_argument("--n_features_per_level", type=int, default=2, help="number of features per level")
	parser.add_argument("--log2_hashmap_size", type=int, default=15, help="log2 hash map size")
	parser.add_argument("--base_resolution", type=int, default=8, help="base resolution")
	parser.add_argument("--xrange", type=int, default=150, help="xrange")
	parser.add_argument("--yrange", type=int, default=100, help="yrange")
	parser.add_argument("--val_gap", type=int, default=50, help="gap for validation")
	parser.add_argument("--numlayer", type=int, default=4, help="number of layers")
	parser.add_argument("--loss", type=int, default=1, help="training loss, 0 is log, 1 is pure l2, 2 is relative l2")
	parser.add_argument("--train_len", type=int, default=2000, help="training length")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	print("================================================================")
	print("This script replicates the behavior of the native CUDA example  ")
	print("mlp_learning_an_image.cu using tiny-cuda-nn's PyTorch extension.")
	print("================================================================")

	print(f"Using PyTorch version {torch.__version__} with CUDA {torch.version.cuda}")

	randseed = 1025
	random.seed(randseed)
	np.random.seed(randseed)
	torch.manual_seed(randseed)
	torch.cuda.manual_seed_all(randseed)

	device = torch.device("cuda")
	args = get_args()
	prefix = args.prefix
	method = args.method
	gap = args.gap
	gap2 = args.gap2
	gt = args.gt
	workers = 8
	n_levels = args.n_levels
	n_features_per_level = args.n_features_per_level
	log2_hashmap_size = args.log2_hashmap_size
	base_resolution = args.base_resolution
	xrange = args.xrange
	yrange = args.yrange
	val_gap = args.val_gap
	numlayer = args.numlayer
	data2 = args.data2
	train_len = args.train_len

	# directories for saving results
	today = datetime.now()
	todaystr = today.strftime('%Y%m%d')
	savedir = todaystr + "/"
	if not os.path.exists(savedir):
		os.makedirs(savedir)

	configuration = str(n_levels) + '_' + str(n_features_per_level) + '_' + str(log2_hashmap_size) + '_' + str(base_resolution)
	savedir = savedir + prefix + '/layer'+str(numlayer)+'/' + configuration + '/' + method + '/'

	result_dir = savedir + "result/"
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	image_dir = savedir + "image/"
	if not os.path.exists(image_dir):
		os.makedirs(image_dir)

	with open(args.config) as config_file:
		config = json.load(config_file)

	n_channels = 3
	n_aux = 0
	dir_dim = 2

	model = tcnn.NetworkWithInputEncoding(n_input_dims = 2 + 2 * dir_dim, n_output_dims = n_channels + n_aux, encoding_config=config["encoding"], network_config=config["network"]).to(device)
	print(model)

	#===================================================================================================
	# The following is equivalent to the above, but slower. Only use "naked" tcnn.Encoding and
	# tcnn.Network when you don't want to combine them. Otherwise, use tcnn.NetworkWithInputEncoding.
	#===================================================================================================
	# encoding = tcnn.Encoding(n_input_dims=2, encoding_config=config["encoding"])
	# network = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=n_channels, network_config=config["network"])
	# model = torch.nn.Sequential(encoding, network)
	
	optimizer_config = config["optimizer"]
	lr = optimizer_config['learning_rate']
	print("learning_rate", lr)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	# read in training data
	with h5py.File(args.data, 'r') as hdf:
		keys = list(hdf.keys())
		view = hdf[keys[0]][:]
		color = hdf[keys[2]][:]
		light = hdf[keys[3]][:]
		jacobian_np = hdf[keys[4]][:]

	view = view[:, 0:yrange, 0:xrange, :]
	color = color[:, 0:yrange, 0:xrange, :]
	light = light[:, 0:yrange, 0:xrange, :]
	jacobian_np = jacobian_np[:, 0:yrange, 0:xrange, :]


	# extra training data
	if data2 != "":
		with h5py.File(args.data2, 'r') as hdf:
			keys = list(hdf.keys())
			view2 = hdf[keys[0]][:]
			color2 = hdf[keys[2]][:]
			light2 = hdf[keys[3]][:]
			jacobian_np2 = hdf[keys[4]][:]

		view2 = view2[:, 0:yrange, 0:xrange, :]
		color2 = color2[:, 0:yrange, 0:xrange, :]
		light2 = light2[:, 0:yrange, 0:xrange, :]
		jacobian_np2 = jacobian_np2[:, 0:yrange, 0:xrange, :]

		# combine data
		view = np.concatenate((view, view2), axis=0)
		color = np.concatenate((color, color2), axis=0)
		light = np.concatenate((light, light2), axis=0)
		jacobian_np = np.concatenate((jacobian_np, jacobian_np2), axis=0)


	# take train_len data
	view = view[0:train_len]
	color = color[0:train_len]
	light = light[0:train_len]
	jacobian_np = jacobian_np[0:train_len]

	numdir = color.shape[0]
	resolution = color.shape[1:3]
	n_pixels = resolution[0] * resolution[1]

	print("numdir", numdir)
	print("resolution", resolution)
	print("n_pixels", n_pixels)


	# read in test data
	with h5py.File(args.test_data, 'r') as hdf:
		keys = list(hdf.keys())
		test_view = hdf[keys[0]][:]
		test_location = hdf[keys[1]][:]
		test_color = hdf[keys[2]][:]
		test_light = hdf[keys[3]][:]
		test_jacobian = hdf[keys[4]][:]

	test_view = test_view[:, 0:yrange, 0:xrange, :]
	test_color = test_color[:, 0:yrange, 0:xrange, :]
	test_light = test_light[:, 0:yrange, 0:xrange, :]
	test_location = test_location[:, 0:yrange, 0:xrange, :]
	test_jacobian = test_jacobian[:, 0:yrange, 0:xrange, :]

	numdir2 = test_color.shape[0]	

	half_dim1 =  0.5 / resolution[0]
	half_dim2 =  0.5 / resolution[1]
	dim1 = torch.linspace(half_dim1, 1-half_dim1, resolution[0], device=device)
	dim2 = torch.linspace(half_dim2, 1-half_dim2, resolution[1], device=device)
	dim1v, dim2v = torch.meshgrid([dim1, dim2])  # is different from np.meshgrid
	dim12 = torch.stack((dim2v.flatten(), dim1v.flatten())).T

	img_shape = (resolution[0], resolution[1], n_channels)

	if gt:
		# save training gt
		for i in range(0, numdir, gap):
			curcolor = color[i].reshape(img_shape)
			print("i", i, curcolor.shape)
			curcolor = torch.tensor(curcolor, device=device, dtype=torch.float32)
			curcolor = query_image(curcolor, dim12).detach().cpu().numpy()		
			curcolor = np.reshape(curcolor, img_shape)
			curcolor = mi.Bitmap(curcolor).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32)
			filename = image_dir + 'color_' +str(i)+ '_' + method + '_gt.exr'
			curcolor.write(filename)

		# save test gt
		for i in range(0, numdir2, gap2):
			curcolor = test_color[i].reshape(img_shape)
			print("i", i, curcolor.shape)
			curcolor = torch.tensor(curcolor, device=device, dtype=torch.float32)
			curcolor = query_image(curcolor, dim12).detach().cpu().numpy()		
			curcolor = np.reshape(curcolor, img_shape)
			curcolor = mi.Bitmap(curcolor).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32)
			filename = image_dir + 'color_' +str(i)+ '_' + method + '_val_gt.exr'
			curcolor.write(filename)

	prev_time = time.perf_counter()

	batch_size = args.batch_size
	batch_ele = batch_size * n_pixels

	interval = args.interval

	print(f"Beginning optimization with {args.n_steps} training steps.")

	input = np.concatenate((light, view), axis=-1)
	input = torch.tensor(input, device=device, dtype=torch.float32)
	print("input", input.shape)
	color = torch.tensor(color, device=device, dtype=torch.float32)

	jacobian = torch.tensor(jacobian_np, device=device, dtype=torch.float32)

	total_num = input.shape[0]

	# test data
	test_location = test_location.reshape(-1, 2)
	test_light = test_light.reshape(-1, 2)
	test_view = test_view.reshape(-1, 2)

	test_input = np.concatenate((test_location, test_light, test_view), axis=-1)
	test_input = torch.tensor(test_input, device=device, dtype=torch.float32)
	print("test_input", test_input.shape)

	test_color = torch.tensor(test_color.reshape(-1, 3), device=device, dtype=torch.float32)
	test_jacobian = torch.tensor(test_jacobian.reshape(-1, 1), device=device, dtype=torch.float32)


	dataset = BTFdataset(input, color, jacobian)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)


	losses = []
	test_losses = []
	iter = 0
	go = True
	while go:
		
		for (input_data, output_data, jacobian_data, idx) in dataloader:

			model.train()

			batch = torch.rand([batch_ele, 2], device=device, dtype=torch.float32)
			curinput = torch.cat((batch, input_data.reshape(-1, 4)), dim=-1)
			output = model(curinput)

			# log + relative
			if args.loss == 0:
				#curjacobian = query_jacobian(jacobian_data, batch)
				curjacobian = jacobian_data.reshape(-1, 1)
				targets = query_images(output_data, batch)
				targets = nolow(targets * curjacobian).to(output.dtype)
				relative_l2_error = (output - targets)**2 / (output.detach()**2 + 0.01)
				loss = relative_l2_error.mean()

			# log
			elif args.loss == 1:
				curjacobian = query_jacobian(jacobian_data, batch)
				targets = query_images(output_data, batch) * curjacobian
				targets = nolow(targets)
				l2_error = (output - targets.to(output.dtype))**2
				loss = l2_error.mean()

			# pure l2 loss
			elif args.loss == 2:
				
				curjacobian = query_jacobian(jacobian_data, batch)
				targets = query_images(output_data, batch) * curjacobian
				l2_error = (output - targets.to(output.dtype))**2
				loss = l2_error.mean()

			# relative l2 loss, no log
			else:
				curjacobian = query_jacobian(jacobian_data, batch)
				targets = query_images(output_data, batch) * curjacobian
				relative_l2_error = (output - targets)**2 / (output.detach()**2 + 0.01)
				loss = relative_l2_error.mean()

			losses.append(loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if iter % interval == 0:
				#print("iter: ", iter, "interval ", interval)
				loss_val = loss.item()
				torch.cuda.synchronize()
				elapsed_time = time.perf_counter() - prev_time
				print(f"Step#{iter}: loss={loss_val} time={int(elapsed_time*1000000)}[µs]")

				filename = result_dir + prefix + '_'+method
				curloss = np.array(losses)

				np.save(filename + '_loss.npy', curloss)
				plt.plot(curloss)
				plt.yscale('log')
				plt.xlabel('steps')
				plt.title('loss, step ' + str(iter))
				plt.savefig(filename+'_loss_lr'+str(lr)+'.png')
				plt.close()

				# Ignore the time spent saving the image
				prev_time = time.perf_counter()

				if iter > 0 and interval < 1000:
					interval *= 10

			# validation
			if iter % val_gap == 0:
				model.eval()
				with torch.no_grad():
					output = model(test_input)

					# if args.loss == 0:
					# 	# log
					# 	output = torch.exp(output) - 1
					# 	output = output / test_jacobian
					# 	output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
					# 	output = nolow(output)
					# 	l2_error = (output - nolow(test_color.to(output.dtype)))**2
					# 	loss = l2_error.mean()
					# else:
					# 	# pure l2 loss
					output = output / test_jacobian
					output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
					l2_error = (output - test_color.to(output.dtype))**2
					loss = l2_error.mean()


					# # print("loss", loss.item())
					# name, param = list(model.named_parameters())[0]
					# param = param.detach().cpu().numpy()
					# np.save(result_dir + prefix + '_' + method + '_param_iter'+str(iter)+'.npy', param)

					# # test_color = nolow(test_color.to(output.dtype))
					# # relative_l2_error = (output - test_color)**2 / (output.detach()**2 + 0.01)
					# # loss = relative_l2_error.mean()


					# # pure l2 loss
					# output = output / test_jacobian
					# output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
					# l2_error = (output - test_color.to(output.dtype))**2
					# loss = l2_error.mean()

					test_losses.append(loss.item())

					# save test loss
					filename = result_dir + prefix + '_' + method
					curloss = np.array(test_losses)

					np.save(filename + '_val_loss.npy', curloss)
					plt.plot(curloss)
					plt.yscale('log')
					plt.xlabel('steps')
					plt.title('loss, step ' + str(iter))
					plt.savefig(filename+'_val_loss_lr'+str(lr)+'.png')
					plt.close()


			iter += 1

			if iter >= args.n_steps:
				go = False
				break


	# save model
	filename = result_dir + prefix + '_' + method + '.pth'
	torch.save(model, filename)


	light = light.reshape(numdir, resolution[0], resolution[1], dir_dim)
	view = view.reshape(numdir, resolution[0], resolution[1], dir_dim)
	test_light = test_light.reshape(numdir2, resolution[0], resolution[1], dir_dim)
	test_view = test_view.reshape(numdir2, resolution[0], resolution[1], dir_dim)
	test_jacobian = test_jacobian.reshape(numdir2, resolution[0], resolution[1], 1)

	with torch.no_grad():
		for index in range(0, numdir, gap):
			curlocation = dim12.detach().cpu().numpy()
			curlight = light[index].reshape(-1, dir_dim)
			curview = view[index].reshape(-1, dir_dim)
			curinput = np.concatenate((curlocation, curlight, curview), axis=-1)
			curinput = torch.tensor(curinput, device=device, dtype=torch.float32)
			curoutput = model(curinput)

			if args.loss == 0 or args.loss == 1:
				curoutput = torch.exp(curoutput) - 1

			curoutput = curoutput / jacobian[index].reshape(-1, 1)
			curoutput = torch.nan_to_num(curoutput, nan=0.0, posinf=0.0, neginf=0.0)

			curoutput = curoutput.reshape(img_shape).clamp(0.0).detach().cpu().numpy()

			curoutput = mi.Bitmap(curoutput).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32)
			filename = image_dir + 'color_' +str(index)+ '_' + method + '_pred.exr'
			curoutput.write(filename)

		for index in range(0, numdir2, gap2):
			curlocation = dim12.detach().cpu().numpy()
			curlight = test_light[index].reshape(-1, dir_dim)
			curview = test_view[index].reshape(-1, dir_dim)
			curinput = np.concatenate((curlocation, curlight, curview), axis=-1)
			curinput = torch.tensor(curinput, device=device, dtype=torch.float32)
			curoutput = model(curinput)

			if args.loss == 0 or args.loss == 1:
				curoutput = torch.exp(curoutput) - 1

			curoutput = curoutput / test_jacobian[index].reshape(-1, 1)
			curoutput = torch.nan_to_num(curoutput, nan=0.0, posinf=0.0, neginf=0.0)

			curoutput = curoutput.reshape(img_shape).clamp(0.0).detach().cpu().numpy()

			curoutput = mi.Bitmap(curoutput).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32)
			filename = image_dir + 'color_' +str(index)+ '_' + method + '_val_pred.exr'
			curoutput.write(filename)

	print("done.")

	tcnn.free_temporary_memory()
