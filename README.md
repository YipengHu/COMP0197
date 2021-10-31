# COMP0090: Introduction to Deep Learning
[UCL Module](https://www.ucl.ac.uk/module-catalogue/modules/introduction-to-deep-learning/COMP0090) | [CS](https://www.ucl.ac.uk/computer-science/) | [UCL Moodle Page](https://moodle.ucl.ac.uk/course/view.php?id=1444)
>Term 1 (Autumn), Academic Year 2021-22 


**Module Lead**  
Yipeng Hu <yipeng.hu@ucl.ac.uk>

  
|Tutors & TAs     | Email                       |
|-----------------|-----------------------------|  
|Dr Andre Altmann | a.altmann@ucl.ac.uk         |  
|Dr Ziyi Shen     | ---                         |  
|Ahmed Shahin     | ahmed.shahin.19@ucl.ac.uk   |  
|Shaheer Saeed    | shaheer.saeed.17@ucl.ac.uk  |  
|Kate Yiwen Li    | yiwen.li@st-annes.ox.ac.uk  |  
|Sophie Martin    | s.martin.20@ucl.ac.uk       |  
|Liam Chalcroft   | liam.chalcroft.20@ucl.ac.uk |  
|Mark Pinnock     | mark.pinnock.18@ucl.ac.uk   |  
|Iani Gayo        | iani.gayo.20@ucl.ac.uk      |  
|Qi Li            | qi.li.21@ucl.ac.uk          |  

	
## 1. Development environment
The module tutorials (see bellow) and coursework use Python, NumPy and an option between TensorFlow and PyTorch. The [Development environment](docs/dev.md) document contains details of the supported development environment, though it is not mandatory.  


## 2. Tutorials
### Quick start
To run the tutorial examples, follow the instruction below.

First, set up the environment:
``` bash
conda create --name comp0090 tensorflow pytorch torchvision
conda activate comp0090
```
>Additional libraries required for individual tutorials are specified in the _readme_ file in each tutorial directory. 

>Scripts with "_tf" and "_pt" postfix are using TensorFlow 2 and PyTorch, respectively.

>All visual examples will be saved in files, without requiring graphics.

Then, change directory `cd` to each individual tutorial folder and run individual training scripts, e.g.:
``` bash
python train_pt.py   
```
or 
``` bash
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
