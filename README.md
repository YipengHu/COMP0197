# COMP0090: Introduction to Deep Learning
UCL Module | [Department of Computer Science](https://www.ucl.ac.uk/computer-science/) | [UCL Moodle Page](https://moodle.ucl.ac.uk/course/view.php?id=1444)
>Term 1 (Autumn), Academic Year 2021-22 


**Module Lead**  
Yipeng Hu <yipeng.hu@ucl.ac.uk>

  
|Tutors & TAs     | Email                       |
|-----------------|-----------------------------|  
|Dr Andre Altmann | a.altmann@ucl.ac.uk         |  
|Dr Ziyi Shen     | ---                         |  
|Shaheer Saeed    | shaheer.saeed.17@ucl.ac.uk  |  
|Kate Li          | yiwen.li@st-annes.ox.ac.uk  |  
|Sophie Martin    | s.martin.20@ucl.ac.uk       |  
|Liam Chalcroft   | liam.chalcroft.20@ucl.ac.uk |  
|Mark Pinnock     | mark.pinnock.18@ucl.ac.uk   |  


## 1. Development environment

### Python
The tutorials require a few dependencies, e.g. NumPy, in addition to one of the two deep learning libraries. Individual tutorials may also require other libraries which will be specified in the readme.md in individual tutorial folders (see links below). 

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
conda create --name comp0090 numpy tensorflow pytorch torchvision
conda activate comp0090
```
>Additional libraries required for individual tutorials are specified in the _readme_ file in each tutorial directory. 

>Scripts with "_tf" and "_pt" postfix are using TensorFlow 2 and PyTorch, respectively.

>All visual examples will be saved in jpg files, without graphics.

Then, change directory `cd` to each individual tutorial folder and run individual training scripts, e.g.:
``` bash
python train_pt.py   
```
or 
``` bash
python train_tf.py  
```

After training, run individual test script if available, e.g.:
``` bash
python test_pt.py   
```
or 
``` bash
python test_tf.py  
```



### Convolutional neural networks
[Image classification](tutorials/img_cls)  
[Image segmentation](tutorials/img_sgm)

### Recurrent neural networks
[Text classification](tutorials/txt_cls)

### Variational autoencoder
[MNIST generation](tutorials/mnist_vae)

### Generative adversarial networks
[Face image simulation](tutorials/face_gan)


## 3. Reading list
A collection of books and research papers is provided in the [Reading List](docs/reading.md).
