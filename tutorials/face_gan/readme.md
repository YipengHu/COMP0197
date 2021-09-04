# Generative adversarial networks

The [Celeb-A Faces dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is used. 

The Celeb-A Faces dataset can be downloaded at the [linked site](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), or in [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg). The dataset will be used is "img_align_celeba.zip". Once downloaded, create a directory named "data" and extract the zip file into that directory. The dataroot is by default set to `dataroot="data"`. The resulting directory structure should be:

```
face_gan/data/img_align_celeba
    -> 000001.jpg
    -> 000002.jpg
    -> 000003.jpg
    -> 000004.jpg
           ...
```

<img src="../../docs/media/celeba_gans.gif" alt="alt text"/>
