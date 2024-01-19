import numpy as np
import os
import sys
import mitsuba as mi
import drjit as dr
import h5py
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import ujson

mi.set_variant('cuda_ad_rgb')

def downsample_ndf(ndf, factor, numdir, xnum, ynum):    
    new_rows = ynum // factor
    new_cols = xnum // factor
    allndf = np.zeros((numdir, new_rows, new_cols, 3, factor**2))
    for i in range(factor):
        for j in range(factor):
            tmp = ndf[:, i::factor, j::factor, :]
            allndf[:,:,:,:,i*factor+j] = tmp[:,:new_rows, :new_cols, :]
    newndf = np.mean(allndf, axis=-1)

    return newndf

def getgrid(dataset, filter, angle_limit=70):
    target_phis_degree = []
    target_thetas_degree = []
    numdir = 0
    thetas = []
    phis = []
    for item in dataset:
        if filter is not None and filter not in item['fname']:
            continue
        thetas += [item['theta_r']]
        phis += [item['phi_r']]

        if item['theta_r'] > angle_limit:
            continue

        target_thetas_degree += [item['theta_r']]
        target_phis_degree += [item['phi_r']]

        numdir += 1

    target_phis = np.array(target_phis_degree)*np.pi/180
    target_thetas = np.array(target_thetas_degree)*np.pi/180
    

    xgrid = np.sin(target_thetas)*np.cos(target_phis)
    ygrid = np.sin(target_thetas)*np.sin(target_phis)
    zgrid = np.cos(target_thetas)

    return xgrid, ygrid, zgrid

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

    parser.add_argument('--data', default='sari_05', type=str,
                        help="data name, \
                        default is leather_04")
    parser.add_argument('--factor', default=2, type=int,
                        help="downsample factor, \
                        default is 2")
    parser.add_argument('--xnum', default=1200, type=int,
                        help="xnum, \
                        default is 1200")
    parser.add_argument('--ynum', default=800, type=int,
                        help="ynum, \
                        default is 800")
    parser.add_argument('--same', default=1, type=int,
                        help="same direction, \
                        default is 1")
    
    args = parser.parse_args()
    data = args.data
    xnum = args.xnum
    ynum = args.ynum
    factor = args.factor
    same = args.same

    totalnum = xnum*ynum
    xnum_final = xnum // factor
    ynum_final = ynum // factor
    finalnum = totalnum // factor // factor
    
    # read in ndf
    rootdir = '/home/xia/Github/BTF/data/'
    rect = rootdir + data + '_rectified'
    dataset_fname = f'{rect}/retro/dataset.json'

    # plot average ndf and look at the plot to determine the number of lobes!!
    print(f'Loading dataset JSON file "{dataset_fname}" ..')
    with open(dataset_fname, 'r') as f:
        dataset = ujson.load(f)

    filter = 'retro'
    xgrid, ygrid, zgrid = getgrid(dataset, filter, angle_limit=70)    


    phinum = 60
    thetanum = 15
    numdir = phinum * (thetanum-1) + 1
    fileName = rect+'/final/ndf_R_xnum'+str(xnum)+'_ynum'+str(ynum)+'.binary'
    ndfmat_R = np.fromfile(fileName, dtype="float32")
    ndfmat_R = np.reshape(ndfmat_R, (totalnum, numdir)).transpose()

    fileName = rect+'/final/ndf_G_xnum'+str(xnum)+'_ynum'+str(ynum)+'.binary'
    ndfmat_G = np.fromfile(fileName, dtype="float32")
    ndfmat_G = np.reshape(ndfmat_G, (totalnum, numdir)).transpose()

    fileName = rect+'/final/ndf_B_xnum'+str(xnum)+'_ynum'+str(ynum)+'.binary'
    ndfmat_B = np.fromfile(fileName, dtype="float32")
    ndfmat_B = np.reshape(ndfmat_B, (totalnum, numdir)).transpose()

    # downsample ndf
    ndfmat = np.stack((ndfmat_R, ndfmat_G, ndfmat_B), axis=-1)

    ndfmat = ndfmat.reshape(numdir, ynum, xnum, 3)
    ndfmat = downsample_ndf(ndfmat, factor, numdir, xnum, ynum)
    print("ndfmat.shape", ndfmat.shape)

    # create dataset
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
    print("location.shape", location.shape)

    # direction
    direction = np.stack((xgrid, ygrid), axis=1)
    # repeat and make the shape numdir x 2 to numdir x ynum x xnum x 2
    directionx = np.repeat(direction[:, 0][:,np.newaxis], finalnum, axis=1).reshape(numdir, ynum_final, xnum_final)
    directiony = np.repeat(direction[:, 1][:,np.newaxis], finalnum, axis=1).reshape(numdir, ynum_final, xnum_final)
    direction = np.stack((directionx, directiony), axis=-1)
    print("direction.shape", direction.shape)

    # create dataset and save
    folder = 'real_' + data
    new_file_path = 'data/'+folder+'_ndf_factor'+str(factor)
    if same == 1:
        new_file_path = new_file_path + '_samedirection.hdf5'
    else:
        new_file_path = new_file_path + '.hdf5'
    dataset_shapes = [(numdir, ynum_final, xnum_final, 2), (numdir, ynum_final, xnum_final, 2), \
                      (numdir, ynum_final, xnum_final, 3)]
    dataset_dtype = np.float32

    dataall = [direction, location,  ndfmat]
    print("len(dataall)", len(dataall))

    # Dataset names
    dataset_names = [
        'ground_dir',
        'ground_loc',
        'ground_color'
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
