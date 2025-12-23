# COMP0197: Applied Deep Learning
[UCL Module](https://www.ucl.ac.uk/module-catalogue/modules/applied-deep-learning-COMP0197) | [CS](https://www.ucl.ac.uk/computer-science/) | [UCL Moodle Page]()
>Term 2, Academic Year 2025-26 

## 1. Development environment
The module tutorials (see bellow) and coursework use Python, NumPy and an option between TensorFlow and PyTorch. The [Development environment](docs/dev.md) document contains details of the supported development environment, though it is not mandatory.  


## 2. Tutorials
### Quick start
Create a micromamba environment `comp0197` and install PyTorch and/or TensorFlow:
``` bash
micromamba create --name comp0197 python=3.12 -y
micromamba activate comp0197
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-cpu 
```

Go to a tutorial folder and run the training script:
``` bash
cd path/to/tutorial
python train_pt.py
# or
python train_tf.py
```

Notes:
- Each tutorial folder has its own `readme` with any extra dependencies or data.
- `*_pt` scripts use PyTorch, `*_tf` scripts use TensorFlow 2.
- Visual outputs are saved to files, no GUI required.


### Convolutional neural networks
[Image classification](tutorials/img_cls)  
[Image segmentation](tutorials/img_sgm)

### Recurrent neural networks
[Text classification](tutorials/txt_cls)  
[Character generation](tutorials/char_gen)

### Transformers
[Language modelling](tutorials/gen_tform)

### Variational autoencoder
[MNIST generation](tutorials/mnist_vae)

### Generative adversarial networks
[Face image simulation](tutorials/face_gan)


## 3. Reading list
A collection of books and research papers is provided in the [Reading List](docs/reading.md).
