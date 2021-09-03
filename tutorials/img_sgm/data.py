
# https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

import os
import requests
import tarfile

from PIL import Image
import numpy as np
import h5py


DATA_PATH = './data'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)


filenames = ['images.tar.gz', 'annotations.tar.gz']

print('Downloading and extracting data...')
for temp_file in filenames:
    url = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/' + temp_file
    print('from ' + url + ' ...')
    r = requests.get(url,allow_redirects=True)
    _ = open(temp_file,'wb').write(r.content)
    with tarfile.open(temp_file) as tar_obj:
        tar_obj.extractall()
        tar_obj.close()
    os.remove(temp_file)


print('Converting data...')
image_dir = 'images'
seg_dir = 'annotations/trimaps'
im_size = (200,200)
# write all images/labels to h5 file
img_h5 = h5py.File(os.path.join(DATA_PATH,"images.h5"), "w")
seg_h5 = h5py.File(os.path.join(DATA_PATH,"labels.h5"), "w")
for idx, im_file in enumerate([f for f in os.listdir(image_dir) if f.endswith('.jpg')]):
    with Image.open(os.path.join(image_dir,im_file)) as img:
        img = np.array(img.resize(im_size).getdata(),dtype='uint8').reshape(im_size[0],im_size[1],3)
        img_h5.create_dataset("/{:06d}/".format(idx), data=img)
    with Image.open(os.path.join(seg_dir,im_file.split('.')[0]+'.png')) as seg:
        seg = np.array(seg.resize(im_size).getdata(),dtype='uint8').reshape(im_size[0],im_size[1])
        seg_h5.create_dataset("/{:06d}/".format(idx), data=seg)
img_h5.flush()
seg_h5.flush()
img_h5.close()
seg_h5.close()

print('Data saved in %s.' % os.path.abspath(DATA_PATH))
