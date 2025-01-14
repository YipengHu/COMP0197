
# https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

import os
import requests
import tarfile
import shutil
import random

from PIL import Image
import numpy as np
import h5py


DATA_PATH = './data'

## download
filenames = ['images.tar.gz', 'annotations.tar.gz']
url_base = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/'

if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.makedirs(DATA_PATH)

print('Downloading and extracting data...')
for temp_file in filenames:
    url = url_base + temp_file
    print(url + ' ...')
    r = requests.get(url,allow_redirects=True)
    _ = open(temp_file,'wb').write(r.content)
    with tarfile.open(temp_file) as tar_obj:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar_obj)
        tar_obj.close()
    os.remove(temp_file)


## spliting and converting
img_dir = 'images'
seg_dir = 'annotations/trimaps'
#----- options -----
im_size = (64,64)
ratio_val = 0.1
ratio_test = 0.2
#-------------------
img_h5s, seg_h5s = [], []
for s in ["train", "val", "test"]:
    img_h5s.append(h5py.File(os.path.join(DATA_PATH,"images_{:s}.h5".format(s)), "w"))
    seg_h5s.append(h5py.File(os.path.join(DATA_PATH,"labels_{:s}.h5".format(s)), "w"))

img_filenames = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
num_data = len(img_filenames)
num_val = int(num_data * ratio_val)
num_test = int(num_data * ratio_test)
num_train = num_data - num_val - num_test

print("Extracting data into %d-%d-%d for train-val-test (%0.2f-%0.2f-%0.2f)..." % (num_train,num_val,num_test, 1-ratio_val-ratio_test,ratio_val,ratio_test))

random.seed(90)
random.shuffle(img_filenames)

# write all images/labels to h5 file
for idx, im_file in enumerate(img_filenames):

    if idx < num_train:  # train
        ids = 0
    elif idx < (num_train + num_val):  # val
        ids = 1
    else:  # test
        ids = 2

    with Image.open(os.path.join(img_dir,im_file)) as img:
        img = np.array(img.convert('RGB').resize(im_size).getdata(),dtype='uint8').reshape(im_size[0],im_size[1],3)
        img_h5s[ids].create_dataset("{:06d}".format(idx), data=img)
    with Image.open(os.path.join(seg_dir,im_file.split('.')[0]+'.png')) as seg:
        seg = np.array(seg.resize(im_size).getdata(),dtype='uint8').reshape(im_size[0],im_size[1])
        seg_h5s[ids].create_dataset("{:06d}".format(idx), data=seg)

for ids in range(len(img_h5s)):
    img_h5s[ids].flush()
    img_h5s[ids].close()
    seg_h5s[ids].flush()
    seg_h5s[ids].close()

shutil.rmtree(img_dir)
shutil.rmtree(seg_dir.split('/')[0]) #remove entire annatations folder

print('Data saved in %s.' % os.path.abspath(DATA_PATH))
