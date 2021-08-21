# COMP0090: Introduction to Deep Learning
UCL Module | [Department of Computer Science](https://www.ucl.ac.uk/computer-science/) | [UCL Moodle Page](https://moodle.ucl.ac.uk/course/view.php?id=1444)
>Term 1 (Autumn), Academic Year 2021-22 


**Module Lead**  
Yipeng Hu <yipeng.hu@ucl.ac.uk>

|**Other Contacts**   | Email                       | Role           |
|---------------------|-----------------------------|----------------|
|Dr Andre Altmann     | <a.altmann@ucl.ac.uk>       | Academic Staff |
|Dr XX                | <axx.xxxxx@ucl.ac.uk>       | Teaching Assistant |
|XXXX XXX             | <bxx.xxxxx@ucl.ac.uk>       | Teaching Assistant |


## 1. Development environment

### Python
The tutorials require a few dependencies, numpy, matplotlib, in addition to one of the two deep learning libraries. Individual tutorials may also require other libraries which will be specified in the readme.md in individual tutorial folders (see links below). 

### Deep learning libraries
Module tutorials are implemented in Python with both [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/). 

### Technical support
Conda is recommended to manage the required dependencies and libraries. It is not mandatory, in tutorials or assessed coursework, to use any specific development, package or environment management tools. However, technical support will be available with the tested development environment - see the [supported development environment for Python].

- For Python programming and numerical computing: 
    - Basic Python programming is required in this module. Relevant tutorials are readily available, e.g. tutorial links in the [supported development environment for Python].
    - Other tools may be useful but not supported: [Jupyter Notebook](https://jupyter.org/), [Anaconda](https://www.anaconda.com/products/individual) and any other IDEs or code editors.

- For TensorFlow and PyTorch:
    - TA-led tutorials will be provided as a refresher at the beginners level.
    - Other tutorials are readily available, e.g. the respective official documentation [TensorFlow tutorials](https://www.tensorflow.org/tutorials) and [PyTorch tutorials](https://pytorch.org/tutorials/). 
    - TA support are also available with the above-specified development environment.

- For GPU acceleration:
    - [Google Colab](https://colab.research.google.com/) provides freely available computing resource, though restrictions apply.
    - UCL Department of Computer Science hosts a [high performance computing cluster](https://hpc.cs.ucl.ac.uk/), with independent technical support.
    - Other GPU supply is to be confirmed.

[supported development environment for Python]: https://weisslab.cs.ucl.ac.uk/YipengHu/mphy0030/-/blob/main/docs/dev_env_python.md


## 2. Tutorials
### Quick start
To run the tutorial examples, follow the instruction below.

First, set up the environment:
``` bash
conda create --name comp0090 numpy matplotlib h5py tensorflow pytorch torchvision
conda activate comp0090
```
>Additional libraries may be required for individual tutorials. Please see the _readme.md_ file in individual tutorial folders. 

Then, go to each individual tutorial directory and run individual script, e.g.
``` bash
python script_pt.py   
```
or 
``` bash
python script_tf.py  
```

>Scripts with "_tf" and "_pt" postfix are using TensorFlow 2 and PyTorch, respectively.


### Convolutional neural networks
[Image classification](tutorials/image_classification)
[Image segmentation](tutorials/image_segmentation)

### Recurrent neural networks
[Text classification](tutorials/text_classification)

### Autoencoder
[Unsupervised variational autoencoder](tutorials/mnist_vae)

### Generative adversarial networks
[Image simulation](tutorials/image_gan)


## 3. Reading list
A collection of books and research papers is provided in the [Reading List](docs/reading.md).
