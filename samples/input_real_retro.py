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


def read_retro(fulldir, factor, xstart, ystart, xnum, ynum):    
    dataset_fname = f'{fulldir}/dataset.json'
    print(f'Loading dataset JSON file "{dataset_fname}" ..')

    with open(dataset_fname, 'r') as f:
        dataset = ujson.load(f)

    cam_mat = np.asarray(dataset[0]['outgoing_matrix_measured'])

    # full directions
    thetai_degree = []
    phii_degree = []
    thetao_degree = []
    phio_degree = []

    angle_limit = 70

    index = 0
    numdir = 0
    filter = 'retro'
    imgall = []
    for item in dataset:
        if filter is not None and filter not in item['fname']:
            continue

        fname = f'{fulldir}/r_{item["fname"]}'             

        if not os.path.exists(fname):
            continue

        if item['theta_r'] > angle_limit:
            continue

        index += 1
        if index % 16 !=0:   # only takes 1/16 of the data
            continue
        

        thetai_degree += [item['theta_i']]
        phii_degree += [item['phi_i']]
        thetao_degree += [item['theta_o_measured']]
        phio_degree += [item['phi_o_measured']]

        # read in image and downsample
        print("fname", fname, "index", index, "numdir", numdir)
        img = np.array(mi.Bitmap(fname),dtype = np.float32)
        # select the desired region
        img = img[ystart:(ystart+ynum), xstart:(xstart+xnum), :]
        # downsample
        img = downsample(img, factor)
        imgall.append(img)

        numdir += 1

    f.close()

    print("numdir", numdir)
    print("index", index)
        
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





if __name__ == '__main__':

    description = '''
    BTF_evaluate.py: evaluate the tiny-cuda-nn results using unseen directions
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
    parser.add_argument('--factor', default=2, type=int,
                        help="downsample factor, \
                        default is 2")
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
    

    today = datetime.now()
    savedir = today.strftime('%Y%m%d') + '/evaluate/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    totalnum = xnum * ynum
    xnum_final = xnum // factor
    ynum_final = ynum // factor
    finalnum = totalnum // factor // factor

    # read json file, images and downsample
    rootdir = '/home/xia/Github/BTF/data/'
    fulldir = rootdir + data + '_rectified/full'
    print("fulldir", fulldir)

    numdir, xi, yi, zi, xo, yo, zo, \
    thetais, phiis, thetaos, phios, imgall, cam_mat\
    = read_retro(fulldir, factor, xstart, ystart, xnum, ynum)
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

    light = np.stack((xi, yi), axis=-1)
    light = np.repeat(light[:, np.newaxis], finalnum, axis=1)
    light = light.reshape((numdir, ynum_final, xnum_final, 2))
    print("light.shape", light.shape)    

    view = np.stack((xo, yo), axis=-1)
    view = np.repeat(view[:, np.newaxis], finalnum, axis=1)
    view = view.reshape((numdir, ynum_final, xnum_final, 2))    
    print("view.shape", view.shape)

    # save data
    folder = 'real_' + data
    new_file_path = '../BTFdata/'+folder+'_samedirection_retro.hdf5'
    dataset_shapes = [(numdir, ynum_final, xnum_final, 2), (numdir, ynum_final, xnum_final, 2), (numdir, ynum_final, xnum_final, 3), (numdir, ynum_final, xnum_final, 2)]
    dataset_dtype = np.float32

    dataall = [view, location, imgall, light]
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