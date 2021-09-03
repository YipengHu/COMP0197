import random

import h5py


class H5ImageLoader():
    def __init__(self, img_file, batch_size, seg_file=None):
        self.img_h5 = h5py.File(img_file,'r')
        self.dataset_list = list(self.img_h5.keys())
        if seg_file is not None:
            self.seg_h5 = h5py.File(seg_file,'r')
            if set(self.dataset_list) > set(self.seg_h5.keys()):
                raise("Images are not consistent with segmentation.")
        else:
            self.seg_h5 = None
        
        self.num_images = len(self.img_h5)
        self.batch_size = batch_size
        self.num_batches = int(self.num_images/self.batch_size) # skip the remainders        
        self.img_ids = [i for i in range(self.num_images)]        
        self.image_size = self.img_h5[self.dataset_list[0]][()].shape 

    
    def __iter__(self):
        self.batch_idx = 0
        random.shuffle(self.img_ids)
        return self


    def __next__(self):
        self.batch_idx += 1
        batch_img_ids = self.img_ids[self.batch_idx*self.batch_size:(self.batch_idx+1)*self.batch_size]
        datasets = [self.dataset_list[idx] for idx in batch_img_ids]

        if self.batch_idx>=self.num_batches:
            raise StopIteration
        
        images = [self.img_h5[ds][()] for ds in datasets]
        labels = None if (self.seg_h5 is None) else [self.seg_h5[ds][()]==1 for ds in datasets] # foreground only

        return images, labels
