# train script
# adapted from: https://www.tensorflow.org/tutorials/images/segmentation


import os

import tensorflow as tf

from loader import H5ImageLoader
from network_tf import ResUNet
import utils_tf as utils


os.environ["CUDA_VISIBLE_DEVICES"]="0"
DATA_PATH = './data'


## settings
minibatch_size = 32
network_size = 16
learning_rate = 1e-4
num_epochs = 500
freq_info = 1
freq_save = 100
save_path = "results_tf"

if not os.path.exists(save_path):
    os.makedirs(save_path)


## data loader
loader_train = H5ImageLoader(DATA_PATH+'/images_train.h5', minibatch_size, DATA_PATH+'/labels_train.h5')
loader_val = H5ImageLoader(DATA_PATH+'/images_val.h5', 20, DATA_PATH+'/labels_val.h5')


## network
seg_net = ResUNet(init_ch=network_size)
seg_net = seg_net.build(input_shape=loader_train.image_size)
# seg_net.summary()


## train
optimizer = tf.optimizers.Adam(learning_rate)

@tf.function
def train_step(images, labels):  # train step
    with tf.GradientTape() as tape:
        images, labels = utils.pre_process(images, labels)
        # Q: add data augmentation
        predicts = seg_net(images, training=True)
        loss = tf.reduce_mean(utils.dice_loss(predicts, labels))
    gradients = tape.gradient(loss, seg_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, seg_net.trainable_variables))
    return loss

@tf.function
def val_step(images, labels):  # validation step
    images, labels = utils.pre_process(images, labels)
    predicts = seg_net(images, training=False)
    losses = utils.dice_loss(predicts, labels)
    dsc_scores = utils.dice_binary(predicts, labels)
    return losses, dsc_scores

for epoch in range(num_epochs):

    for frames, masks in loader_train: 
        loss_train = train_step(frames, masks)

    if (epoch+1) % freq_info == 0:
        tf.print('Epoch {}: loss={:0.5f}'.format(epoch,loss_train))

    if (epoch+1) % freq_save == 0:
        losses_all, dsc_scores_all = [], []
        for frames_val, masks_val in loader_val:
            losses, dsc_scores = val_step(frames_val, masks_val)
            losses_all += [losses]
            dsc_scores_all += [dsc_scores]
        tf.print('Epoch {}: val-loss={:0.5f}, val-DSC={:0.5f}'.format(
            epoch,
            tf.reduce_mean(tf.concat(losses_all,axis=0)),
            tf.reduce_mean(tf.concat(dsc_scores_all,axis=0))
            ))
        tf.saved_model.save(seg_net, os.path.join(save_path, 'epoch{:d}'.format(epoch)))
        tf.print('Model saved.')
