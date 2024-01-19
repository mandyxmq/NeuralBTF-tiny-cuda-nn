import numpy as np
import os
import sys
import argparse
import mitsuba as mi
import drjit as dr
import h5py
import matplotlib.pyplot as plt
from datetime import datetime
import ujson
from scipy.spatial.transform import Rotation

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


# vectors is vector3, from_direction is a matrix of directions (defined by normal map), to_direction is vector3, 
# assume from_direction and to_directions are normalized
def rotate_vectors(vectors, from_direction, to_directions):

    # compute rotation axis
    rotation_axes = np.cross(from_direction, to_directions)

    print("from_direction.shape", from_direction.shape)
    print("to_directions.shape", to_directions.shape)
    print("rotation_axes.shape", rotation_axes.shape)

    rotation_magnitudes = np.linalg.norm(rotation_axes, axis=-1, keepdims=True)

    # Avoid division by zero for parallel vectors
    rotation_axes = np.where(rotation_magnitudes > 0, rotation_axes / rotation_magnitudes, rotation_axes)
    
    # Compute the angles for rotation
    angles = np.arccos(np.clip(np.einsum('ji,i->j', from_direction, to_directions), -1.0, 1.0))
    print("angles.shape", angles.shape)
    print("angles[:10]", angles[:10])
    
    rotation_vectors = rotation_axes * angles[:,np.newaxis]

    # Apply the rotations
    rotations = Rotation.from_rotvec(rotation_vectors.reshape(-1, 3))
    rotated_vectors = rotations.apply(vectors)

    print("rotated_vectors.shape", rotated_vectors.shape)

    return rotated_vectors


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
    parser.add_argument('--same', default=1, type=int,
                        help="same direction, \
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
    new_file_path = 'data/' + folder + '_naive_factor' + str(factor)
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


    # position data
    xunit = 1 / xnum_final
    yunit = 1 / ynum_final
    # position
    xvec = np.linspace(xunit/2, 1-xunit/2, xnum_final)
    yvec = np.linspace(yunit/2, 1-yunit/2, ynum_final)
    xx, yy = np.meshgrid(xvec, yvec)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    location = np.stack((xx, yy), axis=1)
    location = np.reshape(location, (ynum_final, xnum_final, 2))
    location = np.repeat(location[np.newaxis, :, :, :], numdir, axis=0).astype(np.float32)

    tmp = np.reshape(light, (numdir*ynum_final*xnum_final, 2))
    tmpz = np.clip(np.sqrt(1 - tmp[:, 0]**2 - tmp[:, 1]**2), 0, 1)
    wi = mi.Vector3f(tmp[:, 0], tmp[:, 1], tmpz)

    tmp = np.reshape(view, (numdir*ynum_final*xnum_final, 2))
    tmpz = np.clip(np.sqrt(1 - tmp[:, 0]**2 - tmp[:, 1]**2), 0, 1)
    wo = mi.Vector3f(tmp[:, 0], tmp[:, 1], tmpz)
    wm = dr.normalize(wi + wo)

    del wo

    half = wm.numpy()[:, 0:2].reshape((numdir, ynum_final, xnum_final, 2)).astype(np.float32)
    print("half.shape", half.shape)

    # compute difference vector, which is to rotate the light direction to the frame where half vector is the normal
    wm = wm.numpy().reshape((numdir, ynum_final, xnum_final, 3)).astype(np.float32)
    print("wm.shape", wm.shape)
    wm = wm[:,0,0,:]
    wi = wi.numpy().reshape((numdir, ynum_final, xnum_final, 3)).astype(np.float32)
    print("wi.shape", wi.shape)
    wi = wi[:,0,0,:]
    wi_rotated = rotate_vectors(wi, wm, np.array([0, 0, 1]))

    print("wi_rotated.shape", wi_rotated.shape)

    for i in range(5):
        wi_check = wi[i, :]
        wi_rotated_check = wi_rotated[i, :]
        wm_check = wm[i, :]

        theta = np.arccos(wi_check[2])
        print("theta", theta)

        theta_ori = np.arccos(np.dot(wi_check, wm_check))
        print("theta_ori", theta_ori)


    # # check if the rotation is correct
    # # check the theta angle of wi_rotated is the same of the angle between wi and wm
    # theta = np.arccos(np.clip(np.einsum('ji,i->j', wi_rotated, wm), -1.0, 1.0))
    # print("theta", theta)
    # print("theta.shape", theta.shape)
    # print("theta.max()", theta.max())
    # print("theta.min()", theta.min())

    # # check the phi angle of wi_rotated is the same of the angle between wi and wm
    # theta_ori = np.arccos(np.clip(np.einsum('ji,i->j', wi, wm), -1.0, 1.0))
    # print("theta_ori", theta_ori)
    # print("theta_ori.shape", theta_ori.shape)
    # print("theta_ori.max()", theta_ori.max())
    # print("theta_ori.min()", theta_ori.min())

    exit()



    diffx = np.repeat(wi_rotated[:, 0], finalnum).reshape((numdir, ynum_final, xnum_final)).astype(np.float32)
    diffy = np.repeat(wi_rotated[:, 1], finalnum).reshape((numdir, ynum_final, xnum_final)).astype(np.float32)
    diff = np.stack((diffx, diffy), axis=-1)
    print("diff.shape", diff.shape)

    del wm, wi, wi_rotated, diffx, diffy, tmp, tmpz, xx, yy, xvec, yvec

    # save data
    folder = 'real_' + data
    new_file_path = 'data/'+folder+'_'+name+'_halfvector_factor'+str(factor)
    if same == 1:
        new_file_path = new_file_path + '_samedirection.hdf5'
    else:
        new_file_path = new_file_path + '.hdf5'
    jacobian = np.ones((numdir, ynum_final, xnum_final, 1)).astype(np.float32)
    dataset_shapes = [(numdir, ynum_final, xnum_final, 2), (numdir, ynum_final, xnum_final, 2), \
                      (numdir, ynum_final, xnum_final, 3), (numdir, ynum_final, xnum_final, 2),
                      (numdir, ynum_final, xnum_final, 1)]
    dataset_dtype = np.float32

    dataall = [half, location, color, diff, jacobian]
    print("len(dataall)", len(dataall))

    # Dataset names
    dataset_names = [
        'ground_camera_dir',
        'ground_camera_target_loc',
        'ground_color',
        'ground_light',
        'jacobian'
    ]

    # Create a new HDF5 file
    with h5py.File(new_file_path, 'w') as new_file:
        for index in range(len(dataset_names)):
            name = dataset_names[index]
            dataset_shape = dataset_shapes[index]
            data = dataall[index].astype(dataset_dtype)
            new_file.create_dataset(name, data=data)

    del view, location, color, light, jacobian
    gc.collect()

    # close file
    new_file.close()
    print("finish writing naive data")