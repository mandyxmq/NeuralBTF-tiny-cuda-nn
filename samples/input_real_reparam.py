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
import h5py

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



def vndf(wi, wm, m_mu_x, m_mu_y, m_alpha_u, m_alpha_v, m_phi):

    m_trans = mi.Vector3f(0, 0, 0)
    m_trans.x = dr.cos(m_phi) / m_alpha_u * wm.x + dr.sin(m_phi) / m_alpha_u * wm.y + \
                (m_mu_x * dr.cos(m_phi) + m_mu_y * dr.sin(m_phi)) / m_alpha_u * wm.z
    m_trans.y = -dr.sin(m_phi) / m_alpha_v * wm.x + dr.cos(m_phi) / m_alpha_v * wm.y + \
                (-m_mu_x * dr.sin(m_phi) + m_mu_y * dr.cos(m_phi)) / m_alpha_v * wm.z
    m_trans.z = wm.z

    norm = dr.norm(m_trans)
    m_trans = dr.normalize(m_trans)

    wi_trans = mi.Vector3f(0, 0, 0)
    wi_trans.x = m_alpha_u * dr.cos(m_phi) * wi.x + m_alpha_u * dr.sin(m_phi) * wi.y
    wi_trans.y = -m_alpha_v * dr.sin(m_phi) * wi.x + m_alpha_v * dr.cos(m_phi) * wi.y
    wi_trans.z = - m_mu_x * wi.x - m_mu_y * wi.y + wi.z
    wi_trans = dr.normalize(wi_trans)

    vndfval = dr.select(m_trans.z > 0, 1 / dr.pi, 0)
    vndfval = dr.select(dr.dot(m_trans, wi_trans)>1e-20, dr.dot(m_trans, wi_trans) * vndfval / ((1 + wi_trans.z)/2), 0)
    vndfval = vndfval * dr.rcp(m_alpha_u * m_alpha_v) / (norm * norm * norm)
    vndfval = dr.select(wm.z < 1e-20, 0, vndfval)

    return vndfval


def computeM(mux, muy, alphax, alphay, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    shear = np.array([[1,0,0],[0,1,0],[mux,muy,1]])
    rot = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    scale = np.array([[1/alphax,0,0],[0,1/alphay,0],[0,0,1]])    
    M = shear @ rot @ scale
    return M

def invertVNDF_np(wi, wm, M):
    wistd = np.linalg.inv(M) @ wi
    # normalize wistd
    wistd = wistd / np.linalg.norm(wistd, axis=0)
    wmstd = M.transpose() @ wm
    # normalize wmstd
    wmstd = wmstd / np.linalg.norm(wmstd, axis=0)

    c = (2.0 * np.sum(wistd * wmstd, axis=0) * wmstd - wistd)

    phi = np.arctan2(c[1,:], c[0,:])
    u1 = (phi / np.pi) * 0.5 + 0.5
    u2 = (1.0 - c[2,:]) / (1.0 + wistd[2,:])

    return u1, u2

def sampleVNDF_np(wi, M, u):
    wistd = np.linalg.inv(M) @ wi    
    # normalize wistd
    wistd = wistd / np.linalg.norm(wistd, axis=0)

    phi = (2 * u[:, 0] - 1 ) * np.pi
    z = (1 - u[:, 1])*(1 + wistd[2,:])-wistd[2,:]
    sinTheta = np.sqrt(np.clip(1 - z * z, 0, 1))
    x = sinTheta * np.cos(phi)
    y = sinTheta * np.sin(phi)
    c = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1).transpose()

    wmstd = c + wistd
    wm = np.linalg.inv(M).transpose() @ wmstd

    # normalize wm
    wm = wm / np.linalg.norm(wm, axis=0)

    return wm, c, wistd


def invertVNDF(wi, wm, m_mu_x, m_mu_y, m_alpha_u, m_alpha_v, m_phi):
    # M^-1 * wi
    wi_trans = mi.Vector3f(0, 0, 0)
    wi_trans.x = m_alpha_u * dr.cos(m_phi) * wi.x + m_alpha_u * dr.sin(m_phi) * wi.y
    wi_trans.y = -m_alpha_v * dr.sin(m_phi) * wi.x + m_alpha_v * dr.cos(m_phi) * wi.y
    wi_trans.z = - m_mu_x * wi.x - m_mu_y * wi.y + wi.z
    wi_trans = dr.normalize(wi_trans)

    # M^T * wm
    m_trans = mi.Vector3f(0, 0, 0)
    m_trans.x = dr.cos(m_phi) / m_alpha_u * wm.x + dr.sin(m_phi) / m_alpha_u * wm.y + \
                (m_mu_x * dr.cos(m_phi) + m_mu_y * dr.sin(m_phi)) / m_alpha_u * wm.z
    m_trans.y = -dr.sin(m_phi) / m_alpha_v * wm.x + dr.cos(m_phi) / m_alpha_v * wm.y + \
                (-m_mu_x * dr.sin(m_phi) + m_mu_y * dr.cos(m_phi)) / m_alpha_v * wm.z
    m_trans.z = wm.z
    m_trans = dr.normalize(m_trans)

    c = (2.0 * dr.dot(wi_trans, m_trans) * m_trans - wi_trans)
    phi = dr.atan2(c.y, c.x)
    u1 = (phi / dr.pi) * 0.5 + 0.5
    u2 = (1.0 - c.z) / (1.0 + wi_trans.z)

    return u1, u2


def sampleVNDF(wi, u1, u2, m_mu_x, m_mu_y, m_alpha_u, m_alpha_v, m_phi):
    wi_trans = mi.Vector3f(0, 0, 0)
    wi_trans.x = m_alpha_u * dr.cos(m_phi) * wi.x + m_alpha_u * dr.sin(m_phi) * wi.y
    wi_trans.y = -m_alpha_v * dr.sin(m_phi) * wi.x + m_alpha_v * dr.cos(m_phi) * wi.y
    wi_trans.z = - m_mu_x * wi.x - m_mu_y * wi.y + wi.z
    wi_trans = dr.normalize(wi_trans)

    phi = (2 * u1 - 1 ) * dr.pi
    z = (1 - u2)*(1 + wi_trans.z)-wi_trans.z
    sinTheta = dr.sqrt(dr.clamp(1 - z * z, 0, 1))
    x = sinTheta * dr.cos(phi)
    y = sinTheta * dr.sin(phi)
    c = mi.Vector3f(x, y, z)

    wm_std = c + wi_trans
    wm = mi.Vector3f(0, 0, 0)
    wm.x = m_alpha_u * dr.cos(m_phi) * wm_std.x - m_alpha_v * dr.sin(m_phi) * wm_std.y \
            - m_mu_x * wm_std.z
    wm.y = m_alpha_u * dr.sin(m_phi) * wm_std.x + m_alpha_v * dr.cos(m_phi) * wm_std.y \
            - m_mu_y * wm_std.z
    wm.z = wm_std.z
    wm = dr.normalize(wm)

    return wm

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
                        default is train")
    
    
    args = parser.parse_args()
    data = args.data
    xstart = args.xstart
    ystart = args.ystart
    xnum = args.xnum
    ynum = args.ynum
    factor = args.factor
    zval = args.zval
    mode = args.mode

    totalnum = xnum * ynum
    xnum_final = xnum // factor
    ynum_final = ynum // factor
    finalnum = totalnum // factor // factor

    # read json file, images and downsample
    rootdir = '/home/xia/Github/BTF/data/'

    numdir = 320

    # save data
    folder = 'real_' + data
    new_file_path = '../BTFdata/'+folder+'_samedirection'+'_'+mode+'.hdf5'
    dataset_dtype = np.float32


    # Dataset names
    dataset_names = [
        'ground_camera_dir',
        'ground_camera_target_loc',
        'ground_color',
        'ground_light'
    ]

    # read hdf5 file
    print("new_file_path", new_file_path)
    with h5py.File(new_file_path, 'r') as hdf:
        # Now you can access datasets, attributes, etc. within the file
        # For example, list all groups
        print("Keys: %s" % hdf.keys())
        keys = list(hdf.keys())
        print("a_group_key", keys[0])

        # Suppose we want to read a dataset named 'data_set_name' within the group
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
    dataset_shapes = [(numdir, ynum_final, xnum_final, 2), (numdir, ynum_final, xnum_final, 2), (numdir, ynum_final, xnum_final, 3), (numdir, ynum_final, xnum_final, 2)]
    
    # store reparam directions
    mapdir = rootdir + data + "_rectified/results/chunk/"
    mux_filename = mapdir+"mux.exr"
    mux_bmp = mi.Bitmap(mux_filename)
    mux = mi.Float(np.tile(np.array(mux_bmp)[:,:,0].flatten(), numdir))

    muy_filename = mapdir+"muy.exr"
    muy_bmp = mi.Bitmap(muy_filename)
    muy = mi.Float(np.tile(np.array(muy_bmp)[:,:,0].flatten(), numdir))

    alphax_filename = mapdir+"alphax.exr"
    alphax_bmp = mi.Bitmap(alphax_filename)
    alphax = mi.Float(np.tile(np.array(alphax_bmp)[:,:,0].flatten(), numdir))

    alphay_filename = mapdir+"alphay.exr"
    alphay_bmp = mi.Bitmap(alphay_filename)
    alphay = mi.Float(np.tile(np.array(alphay_bmp)[:,:,0].flatten(), numdir))

    phi_filename = mapdir+"phi.exr"
    phi_bmp = mi.Bitmap(phi_filename)
    phimap = mi.Float(np.tile(np.array(phi_bmp)[:,:,0].flatten(), numdir))

    tmp = np.reshape(light, (numdir*ynum_final*xnum_final, 2))
    tmpz = np.clip(np.sqrt(1 - tmp[:, 0]**2 - tmp[:, 1]**2), 0, 1)
    wi = mi.Vector3f(tmp[:, 0], tmp[:, 1], tmpz)

    tmp = np.reshape(view, (numdir*ynum_final*xnum_final, 2))
    tmpz = np.clip(np.sqrt(1 - tmp[:, 0]**2 - tmp[:, 1]**2), 0, 1)
    wo = mi.Vector3f(tmp[:, 0], tmp[:, 1], tmpz)
    wm = dr.normalize(wi + wo)

    u1, u2 = invertVNDF(wi, wm, mux, muy, alphax, alphay, phimap)
    u1 = u1.numpy()
    u2 = u2.numpy()

    # # check if I can recover the original directions, why I cannot recover?
    # wm_test = sampleVNDF(wi, u1, u2, mux, muy, alphax, alphay, phimap)
    # wm = np.array(wm) 
    # wm_test = np.array(wm_test)

    # print("wm[0:10, :]", wm[0:10, :])
    # print("wm_test[0:10, :]", wm_test[0:10, :])

    # apply jacobian to color
    vndfval = vndf(wi, wm, muy, muy, alphax, alphay, phimap)
    dwodwh = dr.abs(4 * dr.dot(wo, wm))
    jacobian = dwodwh / vndfval

    color = np.reshape(color, (numdir*ynum_final*xnum_final, 3))

    color = color * jacobian.numpy()[:, np.newaxis]
    mask = np.isnan(color[:, 0]) | (color[:,0]<0) | (u1 < 0) | (u1 > 1) | (u2 < 0) | (u2 > 1)
    color[mask, :] = 0
    color = np.reshape(color, (numdir, ynum_final, xnum_final, 3))
    
    u1 = u1 * 2 - 1
    u2 = u2 * 2 - 1
    view = np.stack((u1.reshape((numdir, ynum_final, xnum_final)), u2.reshape((numdir, ynum_final, xnum_final))), axis=-1)    

    new_file_path = '../BTFdata/'+folder+'_anglenum5_reparam_samedirection'+'_'+mode+'.hdf5'
    print("view.shape", view.shape)
    print("location.shape", location.shape)
    print("color.shape", color.shape)
    print("light.shape", light.shape)
    dataall = [view, location, color, light]
    # Create a new HDF5 file
    with h5py.File(new_file_path, 'w') as new_file:
        # Iterate over dataset names and create each dataset
        for index in range(len(dataset_names)):
            name = dataset_names[index]
            dataset_shape = dataset_shapes[index]
            # Initialize the dataset with random values
            data = dataall[index].astype(dataset_dtype)
            new_file.create_dataset(name, data=data)

    new_file.close()
    print("finish writing reparam data")