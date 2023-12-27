import os
import sys



filename_list = ["real_leather_04", "real_sari_01", "real_wood", "real_light_grey_satin"]
prefix_list = ["leather_real", "sari_real", "wood_real", "light_grey_satin_real"]
method_list = ["naive", "reparam_simple"]


#for i in range(len(filename_list)):
for i in range(1):
    filename = filename_list[i]
    prefix = prefix_list[i]

    for j in range(len(method_list)):
        method = method_list[j]
        if j == 1:
            filename_cur = filename + "_reparam"
        else:
            filename_cur = filename
        filename_cur = filename_cur + '_samedirection'
        command = f"python samples/BTF_pytorch.py --data /home/xia/Github/NeuMIP/data/datasets/{filename_cur}.hdf5 --prefix {prefix} --method {method}_dense_samedirection --n_steps 1"
        print(command)
        os.system(command)