# Introduction to Deep Learning Module Repository
UCL Computer Science Module COMP0090


**Module Lead** Yipeng Hu <yipeng.hu@ucl.ac.uk>

|**Other Contacts**   | Email                       | Role           |
|---------------------|-----------------------------|----------------|
|Dr Andre Altmann     | <a.altmann@ucl.ac.uk>       | Academic Staff |
|Dr XX                | <axx.xxxxx@ucl.ac.uk>       | Teaching Assistant |
|XXXX XXX             | <bxx.xxxxx@ucl.ac.uk>       | Teaching Assistant |


## 1. Development environment

### Python and conda environment
The tutorials require a few dependencies, numpy, matplotlib, in addition to one of the two deep learning libraries. Individual tutorials may also require other libraries which will be specified in the readme.md in individual tutorial folders (see links below). Conda is recommended to manage the required dependencies. 

It is not mandatory, in tutorials or assessed coursework, to use any specific development, package or environment management tools. However, technical support in this module is available for the setups detailed in [Supported Development Environment](docs/env.md). 

### Deep learning libraries
Module tutorials are implemented in both [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/). 

Learning materials for TensorFlow for Medical Imaging are recommended in [Learning TensorFlow for Medical Imaging](docs/tensorflow.md).

Learning materials for PyTorch for Medical Imaging are recommended in [Learning PyTorch for Medical Imaging](docs/pytorch.md).

### General programming and software development
This is outside of the module scope, but one might find useful materials and links in the [UCL Module MPHY0030 Programming Foundations for Medical Image Analysis](https://weisslab.cs.ucl.ac.uk/WEISSTeaching/mphy0030), including: 
- [Supported development environment for Python](https://weisslab.cs.ucl.ac.uk/WEISSTeaching/mphy0030/-/blob/master/docs/dev_env_python.md)
- [Work with Git](https://weisslab.cs.ucl.ac.uk/WEISSTeaching/mphy0030/-/blob/master/docs/dev_env_git.md)
- [A Roadmap to Learning Python and Machine Learning by Yourself](https://weisslab.cs.ucl.ac.uk/WEISSTeaching/mphy0030/-/blob/master/docs/diy_python_ml.md)


## 2. Tutorials
### Quick start
To run the tutorial examples, follow the instruction below.

First, set up the environment:
``` bash
conda create --name mphy0041 numpy matplotlib h5py tensorflow pytorch
conda activate mphy0041
```
>Additional libraries may be required for individual tutorials. Please see the _readme.md_ file in individual tutorial folders. 

Then, go to each individual tutorial and run the `data.py` script to download tutorial data, after replacing `tutorial_subfolder_name`:
``` bash
cd tutorial_subfolder_name  # e.g. `cd segmentation`
python data.py
```

Run one of the training scripts:
``` bash
python train_pt.py  # train using PyTorch
```
or 
``` bash
python train_tf2.py  # train using TensorFlow2
```

Visualise example data and (predicted) labels:
``` bash
python visualise.py
```

### Convolutional neural networks
[Image classification](tutorials/cnn)

### Recurrent neural networks
[](tutorials/rnn)

### Autoencoder
[Unsupervised variational autoencoder](tutorials/vae)

### Deep generative adversarial networks
[Ultrasound image simulation](tutorials/gan)


## 3. Formative assessment
A list of tasks are detailed in the [Formative Assessment](docs/formative.md). Complete them for individual tutorials.

## 4. Reading list
A collection of books and research papers, relevant to deep learning for medical image analysis, is provided in the [Reading List](docs/reading.md).