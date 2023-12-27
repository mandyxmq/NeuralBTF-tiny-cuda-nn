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
import ujson

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

def readfull(jsondir, fulldir, factor, xstart, ystart, xnum, ynum):    
    dataset_fname = f'{jsondir}/dataset.json'
    print(f'Loading dataset JSON file "{dataset_fname}" ..')
    print("fulldir", fulldir)

    with open(dataset_fname, 'r') as f:
        dataset = ujson.load(f)

    cam_mat = np.asarray(dataset[0]['outgoing_matrix_measured'])

    # full directions
    thetai_degree = []
    phii_degree = []
    thetao_degree = []
    phio_degree = []

    angle_limit = 70

    numdir = 0
    filter = 'full'
    imgall = []
    for item in dataset:
        if filter is not None and filter not in item['fname']:
            continue

        fname = f'{fulldir}/r_{item["fname"]}'             

        if not os.path.exists(fname):
            continue

        if item['theta_o'] > angle_limit:
            continue

        thetai_degree += [item['theta_i']]
        phii_degree += [item['phi_i']]
        thetao_degree += [item['theta_o_measured']]
        phio_degree += [item['phi_o_measured']]

        # read in image and downsample
        print("fname", fname)
        img = np.array(mi.Bitmap(fname),dtype = np.float32)
        # select the desired region
        img = img[ystart:(ystart+ynum), xstart:(xstart+xnum), :]
        # downsample
        img = downsample(img, factor)
        imgall.append(img)

        numdir += 1

    print("numdir", numdir)
        
    thetais = np.array(thetai_degree)*np.pi/180
    phiis = np.array(phii_degree)*np.pi/180
    thetaos = np.array(thetao_degree)*np.pi/180
    phios = np.array(phio_degree)*np.pi/180

    xi = np.sin(thetais)*np.cos(phiis)
    yi = np.sin(thetais)*np.sin(phiis)
    zi = np.cos(thetais)
    xo = np.sin(thetaos)*np.cos(phios)
    yo = np.sin(thetaos)*np.sin(phios)
    zo = np.cos(thetaos)

    return  numdir, xi, yi, zi, xo, yo, zo, \
            thetais, phiis, thetaos, phios, imgall, cam_mat


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


def invertVNDF(wi, wm, m_mu_x, m_mu_y, m_alpha_u, m_alpha_v, m_phi):
    wi_trans = mi.Vector3f(0, 0, 0)
    wi_trans.x = m_alpha_u * dr.cos(m_phi) * wi.x + m_alpha_u * dr.sin(m_phi) * wi.y
    wi_trans.y = -m_alpha_v * dr.sin(m_phi) * wi.x + m_alpha_v * dr.cos(m_phi) * wi.y
    wi_trans.z = - m_mu_x * wi.x - m_mu_y * wi.y + wi.z
    wi_trans = dr.normalize(wi_trans)

    m_trans = mi.Vector3f(0, 0, 0)
    m_trans.x = dr.cos(m_phi) / m_alpha_u * wm.x + dr.sin(m_phi) / m_alpha_u * wm.y + \
                (m_mu_x * dr.cos(m_phi) + m_mu_y * dr.sin(m_phi)) / m_alpha_u * wm.z
    m_trans.y = -dr.sin(m_phi) / m_alpha_v * wm.x + dr.cos(m_phi) / m_alpha_v * wm.y + \
                (-m_mu_x * dr.sin(m_phi) + m_mu_y * dr.cos(m_phi)) / m_alpha_v * wm.z
    m_trans.z = wm.z

    norm = dr.norm(m_trans)
    m_trans = dr.normalize(m_trans)

    c = (2.0 * dr.dot(wi_trans, m_trans) * m_trans - wi_trans)
    phi = dr.atan2(c.y, c.x)
    u1 = (phi / dr.pi) * 0.5 + 0.5
    u2 = (1.0 - c.z) / (1.0 + wi_trans.z)

    return u1, u2


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
    
    
    args = parser.parse_args()
    data = args.data
    xstart = args.xstart
    ystart = args.ystart
    xnum = args.xnum
    ynum = args.ynum
    factor = args.factor
    zval = args.zval

    totalnum = xnum * ynum
    xnum_final = xnum // factor
    ynum_final = ynum // factor
    finalnum = totalnum // factor // factor

    # read json file, images and downsample
    rootdir = '/home/xia/Github/BTF/data/'
    jsondir = rootdir + data
    fulldir = rootdir + data + '_rectified/full'

    numdir, xi, yi, zi, xo, yo, zo, \
    thetais, phiis, thetaos, phios, imgall, cam_mat\
    = readfull(jsondir, fulldir, factor, xstart, ystart, xnum, ynum)
    imgall = np.array(imgall)
    print("imgall.shape", imgall.shape)

    # position data
    xunit = 1 / xnum_final
    yunit = 1 / ynum_final
    # position
    xvec = np.linspace(xunit/2, 1-xunit/2, xnum_final)
    yvec = np.linspace(yunit/2, 1-yunit/2, ynum_final)
    xx, yy = np.meshgrid(xvec, yvec)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    location = np.stack((xx, yy), axis=1) * 2 - 1
    location = np.reshape(location, (ynum_final, xnum_final, 2))
    location = np.repeat(location[np.newaxis, :, :, :], numdir, axis=0)
    print("location.shape", location.shape)

    light = np.stack((xi, yi, zi), axis=-1)
    light = np.repeat(light[:, np.newaxis], finalnum, axis=1)
    light = light.reshape((numdir, ynum_final, xnum_final, 3))
    print("light.shape", light.shape)    

    direction = np.stack((xo, yo, zo), axis=-1)
    direction = np.repeat(direction[:, np.newaxis], finalnum, axis=1)
    direction = direction.reshape((numdir, ynum_final, xnum_final, 3))    
    print("direction.shape", direction.shape)

    # # take into account the spatially varying directions
    # M3fileName = rootdir + data + '_rectified/retro/M3.binary'
    # M3 = np.fromfile(M3fileName, dtype="float32")
    # M3 = np.reshape(M3, (3, 3))
    
    # calib = rootdir + data + '_rectified/geo_calib.json'
    # print(f'Loading calibration JSON file "{calib}" ..')
    # with open(calib, 'r') as f:
    #     params = ujson.load(f)

    # cam_pos = np.asarray(params['cam_trans'])
    # axis_trans = np.asarray(params['axis_trans'])   # origin of rotation
    # dist = np.linalg.norm(cam_pos - axis_trans)
    # posz = zval - axis_trans[2]

    # mapdir = rootdir + data + "_rectified/results/chunk/"
    # posall_small = np.zeros((ynum, xnum, 3))
    # for i in range(10):
    #     pos = np.fromfile(mapdir + 'pos_xys_chunk'+str(i)+'.binary', dtype="float32")
    #     pos = np.reshape(pos, (ynum//10, xnum, 2))
    #     posall_small[i * ynum//10: (i+1)*ynum//10, :, 0:2] = pos
    # posall_small[:, :, 2] = posz

    # posall_small = downsample(posall_small, factor)

    # view = np.stack((xo, yo, zo), axis=-1)
    # view = np.repeat(view[:, np.newaxis], finalnum, axis=1)
    # view = view.reshape((numdir, ynum_final, xnum_final, 3))

    # posall = np.zeros((numdir, ynum_final, xnum_final, 3))
    # posall[:, :, :, 0] = np.tile(posall_small[:,:,0][np.newaxis, :, :], (numdir, 1, 1))
    # posall[:, :, :, 1] = np.tile(posall_small[:,:,1][np.newaxis, :, :], (numdir, 1, 1))
    # posall[:, :, :, 2] = posz
    # print("posall.shape", posall.shape)

    # direction = view - posall
    # direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    # print("direction.shape", direction.shape)

    view = direction[:, :, :, 0:2]

    # save data
    folder = 'real_' + data
    new_file_path = '../BTFdata/'+folder+'_samedirection.hdf5'
    dataset_shapes = [(numdir, ynum_final, xnum_final, 2), (numdir, ynum_final, xnum_final, 2), (numdir, ynum_final, xnum_final, 3), (numdir, ynum_final, xnum_final, 2)]
    dataset_dtype = np.float32

    dataall = [view, location, imgall, light[:, :, :, 0:2]]
    print("len(dataall)", len(dataall))

    # Dataset names
    dataset_names = [
        'ground_camera_dir',
        'ground_camera_target_loc',
        'ground_color',
        'ground_light'
    ]

    # Create a new HDF5 file
    with h5py.File(new_file_path, 'w') as new_file:
        # Iterate over dataset names and create each dataset
        for index in range(len(dataset_names)):
            name = dataset_names[index]
            dataset_shape = dataset_shapes[index]
            # Initialize the dataset with random values
            data = dataall[index].astype(dataset_dtype)
            new_file.create_dataset(name, data=data)

    # close file
    new_file.close()
    print("finish writing naive data")