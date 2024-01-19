import numpy as np
import os
import sys
import argparse
import mitsuba as mi
import drjit as dr
import h5py
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import gc

sys.path.append('/home/xia/Github/NeuralBTF-tiny-cuda-nn/samples/')
from brdf import *


mi.set_variant('cuda_ad_rgb')

def downsample(img, factor):
    rows, cols, channels = img.shape
    new_rows = rows // factor
    new_cols = cols // factor
    allimg = np.zeros((new_rows, new_cols, channels, factor**2))
    for i in range(factor):
        for j in range(factor):
            tmp = img[i::factor, j::factor, :]
            allimg[:,:,:,i*factor+j] = tmp[:new_rows, :new_cols, :]
    newimg = np.mean(allimg, axis=3)
    newimg = newimg.reshape(new_rows, new_cols, channels)
    return newimg


def rot(axis, angle):
    '''Compute a rotation matrix from an axis-angle representation'''

    # Lifted from http://en.wikipedia.org/wiki/Rotation_matrix
    ca = np.cos(angle)
    sa = np.sin(angle)
    C = 1 - ca

    # Unpack
    x, y, z = axis

    # Precompute a few multiplications
    xs = x*sa
    ys = y*sa
    zs = z*sa
    xC = x*C
    yC = y*C
    zC = z*C
    xyC = x*yC
    yzC = y*zC
    zxC = z*xC

    return np.array([
        [x*xC + ca, xyC - zs, zxC + ys],
        [xyC + zs, y*yC + ca, yzC - xs],
        [zxC - ys, yzC + xs, z*zC + ca]
    ])


def rot_compact(v):
    '''
    Compute a rotation matrix from a compact axis-angle representation.

    In contrast to ``rot()``, this function determines the rotation angle (in
    radians) from the 2-norm of the 3D vector 'v'. The axis is given by the
    normalized direction.
    '''
    v = np.asarray(v)
    angle = np.linalg.norm(v)
    if angle > 0.0:
        return rot(v / angle, angle)
    else:
        return np.eye(3)
    

def image_to_3d(fov, aspect, cam_res, pos_proj, cam_rot, cam_matrix, cam_pos, zval):
    c = 1.0 / np.tan(0.5 * fov)

    a = (pos_proj[1] / cam_res[1] * 2 - 1) / c
    b = (pos_proj[0] / cam_res[0] * 2 - 1) / (c / aspect)

    R_prime = rot_compact(cam_rot) @ cam_matrix.T
    P_prime = rot_compact(cam_rot) @ cam_pos

    num = a.shape[0]
    mat = np.zeros((2, 2, num))
    mat[0, 0, :] = R_prime[0, 0] - a * R_prime[2, 0]
    mat[0, 1, :] = R_prime[0, 1] - a * R_prime[2, 1]
    mat[1, 0, :] = R_prime[1, 0] - b * R_prime[2, 0]
    mat[1, 1, :] = R_prime[1, 1] - b * R_prime[2, 1]

    vec = np.zeros((2, num))
    vec[0, :] = a * R_prime[2, 2] * zval - a * P_prime[2] + P_prime[0] - R_prime[0, 2] * zval
    vec[1, :] = b * R_prime[2, 2] * zval - b * P_prime[2] + P_prime[1] - R_prime[1, 2] * zval

    pos = np.zeros((2, num))
    for i in range(num):
        pos[:, i] = np.linalg.solve(mat[:, :, i], vec[:, i])

    return pos



if __name__ == '__main__':

    description = '''
    NeuMIP_input_real.py: generate training data for NeuMIP, based on real capture.
    '''

    class DefaultHelpParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)

    os.environ["COLUMNS"] = "80"
    parser = DefaultHelpParser(description=description)

    parser.add_argument('--data', default='leather_04', type=str,
                        help="data name, \
                        default is leather_04")
    parser.add_argument('--factor', default=3, type=int,
                        help="downsample factor, \
                        default is 3")
    parser.add_argument('--xstart', default=1100, type=int,
                        help="xstart, \
                        default is 1100")
    parser.add_argument('--ystart', default=600, type=int,
                        help="ystart, \
                        default is 600")
    parser.add_argument('--xnum', default=1800, type=int,
                        help="xnum, \
                        default is 1800")
    parser.add_argument('--ynum', default=1200, type=int,
                        help="ynum, \
                        default is 1200")
    parser.add_argument('--zval', default=0, type=float,
                        help="zval, \
                        default is 0")
    parser.add_argument('--mode', default='', type=str,
                        help="mode, \
                        default is empty, could be _retro")
    parser.add_argument('--same', default=1, type=int,
                        help="same, \
                        default is 1")
    parser.add_argument('--name', default='full', type=str,
                        help="name, \
                        default is full")
    
    args = parser.parse_args()
    data = args.data
    xstart = args.xstart
    ystart = args.ystart
    xnum = args.xnum
    ynum = args.ynum
    factor = args.factor
    zval = args.zval
    mode = args.mode
    same = args.same
    name = args.name

    totalnum = xnum * ynum
    xnum_final = xnum // factor
    ynum_final = ynum // factor
    finalnum = totalnum // factor // factor

    # read json file, images and downsample
    rootdir = '/home/xia/Github/BTF/data/'

    # save data
    folder = 'real_' + data + '_' + name
    new_file_path = 'data/' + folder + '_naive_factor' + str(factor) + mode
    if same == 1:
        new_file_path = new_file_path + '_samedirection.hdf5'
    else:
        new_file_path = new_file_path + '.hdf5'

    print("reading data from", new_file_path)

    dataset_dtype = np.float32

    # Dataset names
    dataset_names = [
        'ground_camera_dir',
        'ground_camera_target_loc',
        'ground_color',
        'ground_light',
        'jacobian'
    ]

    # read hdf5 file
    print("new_file_path", new_file_path)
    with h5py.File(new_file_path, 'r') as hdf:
        print("Keys: %s" % hdf.keys())
        keys = list(hdf.keys())
        print("a_group_key", keys[0])
        view = hdf[keys[0]][:]
        location = hdf[keys[1]][:]
        color = hdf[keys[2]][:]
        light = hdf[keys[3]][:]

    hdf.close()

    numdir = view.shape[0]
    print("view.shape", view.shape)
    print("location.shape", location.shape)
    print("color.shape", color.shape)
    print("light.shape", light.shape) 

    tmp = np.reshape(light, (numdir*ynum_final*xnum_final, 2))
    tmpz = np.clip(np.sqrt(1 - tmp[:, 0]**2 - tmp[:, 1]**2), 0, 1)
    wi = mi.Vector3f(tmp[:, 0], tmp[:, 1], tmpz)

    tmp = np.reshape(view, (numdir*ynum_final*xnum_final, 2))
    tmpz = np.clip(np.sqrt(1 - tmp[:, 0]**2 - tmp[:, 1]**2), 0, 1)
    wo = mi.Vector3f(tmp[:, 0], tmp[:, 1], tmpz)
    wm = dr.normalize(wi + wo)
    mapdir = rootdir + data + '_rectified/final/'
    
    # reparam using standard GGX
    roughnessx_filename = mapdir+'alphax_'+str(factor)+'x.exr'
    #roughnessx_filename = mapdir+'roughnessx_'+str(factor)+'x.exr'
    roughnessx_bmp = mi.Bitmap(roughnessx_filename)
    alphax = np.array(roughnessx_bmp)[:ynum,:xnum,0:1]
    alphax = mi.Float(alphax.flatten()) 

    roughnessy_filename = mapdir+'alphay_'+str(factor)+'x.exr'
    #roughnessy_filename = mapdir+'roughnessy_'+str(factor)+'x.exr'
    roughnessy_bmp = mi.Bitmap(roughnessy_filename)
    alphay = np.array(roughnessy_bmp)[:ynum,:xnum,0:1]
    alphay = mi.Float(alphay.flatten())

    alphax = dr.tile(alphax, numdir)
    alphay = dr.tile(alphay, numdir)
    # alphax = 0.32766736
    # alphay = 0.01955817

    # apply jacobian to color
    vndfval = vndf_simple(wi, wm, alphax, alphay)
    dwodwh = dr.abs(4 * dr.dot(wo, wm))
    #jacobian = dwodwh / vndfval
    jacobian = 1 / vndfval

    del alphax, alphay, vndfval, dwodwh

    color = np.reshape(color, (numdir*ynum_final*xnum_final, 3))

    jacobian = np.array(jacobian)
    jacobian = np.reshape(jacobian, (numdir, ynum_final, xnum_final, 1))

    color = np.reshape(color, (numdir, ynum_final, xnum_final, 3))

    new_file_path = 'data/'+folder+'_reparam_new_factor'+str(factor)+mode
    if same == 1:
        new_file_path = new_file_path + '_samedirection.hdf5'
    else:
        new_file_path = new_file_path + '.hdf5'
        
    print("view.shape", view.shape)
    print("location.shape", location.shape)
    print("color.shape", color.shape)
    print("light.shape", light.shape)
    dataall = [view, location, color, light, jacobian]
    dataset_shapes = [(numdir, ynum_final, xnum_final, 2), (numdir, ynum_final, xnum_final, 2), \
                      (numdir, ynum_final, xnum_final, 3), (numdir, ynum_final, xnum_final, 2), \
                        (numdir, ynum_final, xnum_final, 1)]
    
    # Create a new HDF5 file
    with h5py.File(new_file_path, 'w') as new_file:
        for index in range(len(dataset_names)):
            name = dataset_names[index]
            dataset_shape = dataset_shapes[index]
            data = dataall[index].astype(dataset_dtype)
            new_file.create_dataset(name, data=data)

    del view, location, color, light, jacobian
    gc.collect()

    new_file.close()
    print("finish writing reparam data")