# COMP0197: Applied Deep Learning
[UCL Module](https://www.ucl.ac.uk/module-catalogue/modules/applied-deep-learning-COMP0197) | [CS](https://www.ucl.ac.uk/computer-science/) | [UCL Moodle Page]()
>Term 2, Academic Year 2025-26 

## 1. Development environment
The module tutorials (see bellow) and coursework use Python, NumPy and an option between TensorFlow and PyTorch. The [Development environment](docs/dev.md) document contains details of the supported development environment, though it is not mandatory.  


## 2. Tutorials
### Quick start
To run the tutorial examples, follow the instruction below.

First, set up the micromamba environments:
``` bash
micromamba create --name comp0197_pt python=3.12 -y
micromamba activate comp0197_pt
pip install pytorch torchvision
```

``` bash
micromamba create --name comp0197_tf python=3.12 -y
micromamba activate comp0197_tf
pip install tensorflow pillow
```


>Additional libraries and/or data required for individual tutorials are specified in the _readme_ file in each tutorial directory. 

>Scripts with "_tf" and "_pt" postfix are using TensorFlow 2 and PyTorch, respectively.

>All visual examples will be saved in files, without requiring graphics.

Then, change directory `cd` to each individual tutorial folders and run individual training scripts, e.g.:
``` bash
micromamba activate comp0197_pt
python train_pt.py   
```
or 
``` bash
micromamba activate comp0197_tf
python train_tf.py  
```

### Convolutional neural networks
[Image classification](tutorials/img_cls)  
[Image segmentation](tutorials/img_sgm)

### Recurrent neural networks
[Text classification](tutorials/txt_cls)  
[Character generation](tutorials/char_gen)

### Variational autoencoder
[MNIST generation](tutorials/mnist_vae)

### Generative adversarial networks
[Face image simulation](tutorials/face_gan)


## 3. Reading list
A collection of books and research papers is provided in the [Reading List](docs/reading.md).
