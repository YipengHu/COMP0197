# train script
# adapted from: https://www.tensorflow.org/tutorials/images/cnn


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from PIL import Image


## cifar-10 dataset
(train_images, train_labels), (_, _) = datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# example images
num_images = 20
im = Image.fromarray(tf.concat([train_images[i,...] for i in range(num_images)],1).numpy())
im.save("train_tf_images.jpg")
print('train_tf_images.jpg saved.')
print('Ground truth labels:' + ' '.join('%5s' % class_names[train_labels[j,0]] for j in range(num_images)))

# normalize to [0,1]
train_images = train_images / 255.0


## cnn
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()


## compile with loss and optimiser
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


## train 
history = model.fit(train_images, train_labels, epochs=10)
print('Training done.')

# save trained model
model.save('saved_model_tf')
print('Model saved.')
