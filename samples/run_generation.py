import os
import sys

data_list = ["leather_04", "sari_01", "wood", "light_grey_satin"]
xstart_list = [1100, 1100, 1100, 1500]
xnum_list = [1800, 1800, 1800, 1500]
mode = '_retro'


# retro data generation
for i in range(0, len(data_list)):
# for i in range(1, 2):
    data = data_list[i]
    xstart = xstart_list[i]
    xnum = xnum_list[i]
    # command = f"python input_real_retro.py --data " + data + " --xstart " + str(xstart) + " --xnum " + str(xnum)
    # print(command)
    # os.system(command)
    command = f"python input_real_reparam_simple.py --data " + data + " --xstart " + str(xstart) + " --xnum " + str(xnum) 
    os.system(command)