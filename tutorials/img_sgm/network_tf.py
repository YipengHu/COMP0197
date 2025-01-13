#utility functions
import tensorflow as tf
from tensorflow.keras import layers


class ResUNet(tf.keras.Model):

    def __init__(self, init_ch=32, num_levels=3, out_ch=1):
        super().__init__()
        self.first_layer = Conv2dLayer(init_ch)        
        self.encoder = [ResnetBlock(2**i*init_ch, type='down') for i in range(num_levels)]
        self.encoder += [ResnetBlock(2**num_levels*init_ch, type='none')]
        self.decoder = [ResnetBlock(2**i*init_ch, type='up') for i in range(num_levels,0,-1)] 
        self.decoder += [ResnetBlock(init_ch, type='none')]
        self.out_layer = Conv2dLayer(out_ch, is_output=True)

    def call(self, inputs):
        x = self.first_layer(inputs)
        skips = []
        for down in self.encoder[:-1]:
            x = down(x)
            skips += [x]
        x = self.encoder[-1](x)
        for up, skip in zip(self.decoder[:-1], reversed(skips)):
            x = self._resize_to(x,skip) + skip
            x = up(x)
        x = self._resize_to(x,inputs)
        x = self.decoder[-1](x)
        return self.out_layer(x)

    def _resize_to(self, x, y):
        if x.shape[1:3]==y.shape[1:3]:
            return x
        elif  any([abs(x.shape[i]-y.shape[i])>1 for i in [1,2]]):
            raise Warning('padding/cropping more than 1.')
        else:
            return tf.image.resize_with_crop_or_pad(x, y.shape[1], y.shape[2])


class Conv2dLayer(layers.Layer):

    def __init__(self, ch, is_output=False):
        super().__init__()
        self.is_output = is_output
        use_bias = True if is_output else False
        activation = 'sigmoid' if is_output else 'relu'
        self.conv2d = layers.Conv2D(
            filters=ch,
            kernel_size=(3,3),
            padding='same', 
            use_bias=use_bias, 
            activation=activation)
        self.batch_norm = layers.BatchNormalization()

    def call(self, input):
        x = self.conv2d(input)
        y = x if self.is_output else self.batch_norm(x)
        return y


class ResnetBlock(layers.Layer):

    def __init__(self, ch, type, bn=True):
        super().__init__()
        self.type = type
        self.bn = bn
        self.conv2d = Conv2dLayer(ch)
        self.conv2d_down = layers.Conv2D(
                filters=ch*2,
                kernel_size=(3,3),
                strides=(1,1), 
                padding='same', 
                use_bias=False, 
                activation='relu')
        self.max_pool = layers.MaxPool2D(
                pool_size=(2,2),
                strides=(2,2), 
                padding='same')
        self.conv2d_up = layers.Conv2DTranspose(
                filters=int(ch/2),
                kernel_size=(3,3),
                strides=(2,2), 
                padding='same',
                output_padding=1,
                use_bias=False, 
                activation='relu')
        self.batch_norm =  layers.BatchNormalization()

    def call(self, input):
        x = self.conv2d(input)
        x = self.conv2d(x)
        x += input
        # sampling layer
        if self.type == "down":
            x = self.conv2d_down(x)
            x = self.max_pool(x)
            y = self.batch_norm(x) if self.bn else x
        elif self.type == "up":
            x = self.conv2d_up(x)
            y = self.batch_norm(x) if self.bn else x
        else:  # no resampling
            y = x        
        return y
