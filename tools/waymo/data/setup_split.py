import numpy as np
import sys
import os
import shutil
import re

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------

def mkdir_if_missing(directory, delete_if_exist=False):
    """
    Recursively make a directory structure even if missing.

    if delete_if_exist=True then we will delete it first
    which can be useful when better control over initialization is needed.
    """

    if delete_if_exist and os.path.exists(directory): shutil.rmtree(directory)

    # check if not exist, then make
    if not os.path.exists(directory):
        os.makedirs(directory)


split = 'kitti'

# base paths
base_data = os.path.join(os.getcwd())
raw_data = './data'
kitti_raw = dict()
kitti_raw['cal'] = os.path.join(raw_data, 'waymo2kitti', 'training','replace', 'calib')
kitti_raw['ims'] = os.path.join(raw_data, 'waymo2kitti', 'training','replace', 'image_0')
kitti_raw['lab'] = os.path.join(raw_data, 'waymo2kitti', 'training','replace', 'label_0')
kitti_raw['dep'] = os.path.join(raw_data, 'waymo2kitti', 'training','replace', 'projected_points_0')

# kitti_raw['pre'] = os.path.join(base_data, 'kitti', 'training', 'prev_2')

kitti_tra = dict()
kitti_tra['cal'] = os.path.join(base_data, split, 'calib')
kitti_tra['ims'] = os.path.join(base_data, split, 'image')
kitti_tra['lab'] = os.path.join(base_data, split, 'label')
kitti_tra['dep'] = os.path.join(base_data, split, 'depth')

kitti_raw_val = dict()
kitti_raw_val['cal'] = os.path.join(raw_data, 'waymo2kitti', 'validation','replace', 'calib')
kitti_raw_val['ims'] = os.path.join(raw_data, 'waymo2kitti', 'validation','replace', 'image_0')
kitti_raw_val['lab'] = os.path.join(raw_data, 'waymo2kitti', 'validation','replace', 'label_0')
kitti_raw_val['dep'] = os.path.join(raw_data, 'waymo2kitti', 'validation','replace', 'projected_points_0')
kitti_val = dict()

tra_file = os.path.join(base_data,  'train_org.txt')
val_file = os.path.join(base_data,  'val_org.txt')

tra_file_new = os.path.join(base_data, 'train.txt')
val_file_new = os.path.join(base_data, 'val.txt')

# mkdirs
mkdir_if_missing(kitti_tra['cal'])
mkdir_if_missing(kitti_tra['ims'])
mkdir_if_missing(kitti_tra['lab'])
mkdir_if_missing(kitti_tra['dep'])

print('Linking train')
text_file = open(tra_file, 'r')
text_lines = text_file.readlines()
text_file.close()
text_file_new = open(tra_file_new, 'w')

imind = 0

for line in text_lines:

    parsed = line.strip().split(' ')#re.search('(\d+)', line)

    if parsed is not None:

        seg, id = parsed
        new_id = '{:06d}'.format(imind)

        if not os.path.exists(os.path.join(kitti_tra['cal'], str(new_id) + '.txt')):
            os.symlink(os.path.join(kitti_raw['cal'].replace('replace', seg), id + '.txt'), os.path.join(kitti_tra['cal'], str(new_id) + '.txt'))

        if not os.path.exists(os.path.join(kitti_tra['ims'], str(new_id) + '.png')):
            os.symlink(os.path.join(kitti_raw['ims'].replace('replace', seg), id + '.png'), os.path.join(kitti_tra['ims'], str(new_id) + '.png'))

        if not os.path.exists(os.path.join(kitti_tra['dep'], str(new_id) + '.npy')):
            os.symlink(os.path.join(kitti_raw['dep'].replace('replace', seg), id + '.npy'), os.path.join(kitti_tra['dep'], str(new_id) + '.npy'))

        if not os.path.exists(os.path.join(kitti_tra['lab'], str(new_id) + '.txt')):
            os.symlink(os.path.join(kitti_raw['lab'].replace('replace', seg), id + '.txt'), os.path.join(kitti_tra['lab'], str(new_id) + '.txt'))

        text_file_new.write(new_id + '\n')

        imind += 1

text_file.close()
text_file_new.close()

print('Linking val')
text_file = open(val_file, 'r')
text_lines = text_file.readlines()
text_file.close()
text_file_new = open(val_file_new, 'w')

# imind = 0

for line in text_lines:

    parsed = line.strip().split(' ')#re.search('(\d+)', line)

    if parsed is not None:

        seg, id = parsed
        new_id = '{:06d}'.format(imind)

        if not os.path.exists(os.path.join(kitti_tra['cal'], str(new_id) + '.txt')):
            os.symlink(os.path.join(kitti_raw_val['cal'].replace('replace', seg), id + '.txt'),
                       os.path.join(kitti_tra['cal'], str(new_id) + '.txt'))

        if not os.path.exists(os.path.join(kitti_tra['ims'], str(new_id) + '.png')):
            os.symlink(os.path.join(kitti_raw_val['ims'].replace('replace', seg), id + '.png'),
                       os.path.join(kitti_tra['ims'], str(new_id) + '.png'))

        if not os.path.exists(os.path.join(kitti_tra['dep'], str(new_id) + '.npy')):
            os.symlink(os.path.join(kitti_raw_val['dep'].replace('replace', seg), id + '.npy'),
                       os.path.join(kitti_tra['dep'], str(new_id) + '.npy'))

        if not os.path.exists(os.path.join(kitti_tra['lab'], str(new_id) + '.txt')):
            os.symlink(os.path.join(kitti_raw_val['lab'].replace('replace', seg), id + '.txt'),
                       os.path.join(kitti_tra['lab'], str(new_id) + '.txt'))

        text_file_new.write(new_id + '\n')

        imind += 1

text_file.close()
text_file_new.close()
print('Done')

