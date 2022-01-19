from keras import layers
from keras.layers import Layer
from keras.models import Model
from keras.models import load_model
import numpy as np

class MSConv2D(Layer):
    
    def __init__(self, nodes, kernels):
        super(MSConv2D, self).__init__()
        
        # Make 3 Conv2D Layers
        self.Conv1 = layers.Conv2D(nodes, (kernels[0],kernels[0]), activation="relu", padding="same")
        self.Conv2 = layers.Conv2D(nodes, (kernels[1],kernels[1]), activation="relu", padding="same")
        self.Conv3 = layers.Conv2D(nodes, (kernels[2],kernels[2]), activation="relu", padding="same")
    
    def call(self, input):
        '''
        Compute each Convolution separately then add together
        
        Maybe add Batch Normalization Layers after each MaxPool
        '''
        # Kernel 1
        x1 = self.Conv1(input)
        x1 = layers.MaxPooling2D((2, 2), padding="same")(x1)
        
        # Kernel 2
        x2 = self.Conv2(input)
        x2 = layers.MaxPooling2D((2, 2), padding="same")(x2)
        
        # Kernel 3
        x3 = self.Conv3(input)
        x3 = layers.MaxPooling2D((2, 2), padding="same")(x3)
        
        # Add together
        x_sum = layers.Add()([x1, x2, x3])
        
        return x_sum
    
class MSConv2DTranspose(Layer):
    
    def __init__(self, nodes, kernels):
        super(MSConv2DTranspose, self).__init__()
        
        # Conv2D Transposes
        self.Conv1Trans = layers.Conv2DTranspose(nodes, (kernels[0],kernels[0]), strides=2, activation="relu", padding="same")
        self.Conv2Trans = layers.Conv2DTranspose(nodes, (kernels[1],kernels[1]), strides=2, activation="relu", padding="same")
        self.Conv3Trans = layers.Conv2DTranspose(nodes, (kernels[2],kernels[2]), strides=2, activation="relu", padding="same")
        
    def call(self, inputs):
        '''
        Compute each Convolution Transpose separately then add together
        
        Maybe add Batch Normalization Layers after each MaxPool
        '''
        # Kernel 1
        x1 = self.Conv1Trans(inputs)
        
        # Kernel 2
        x2 = self.Conv2Trans(inputs)
        
        # Kernel 3
        x3 = self.Conv3Trans(inputs)
        
        return layers.Add()([x1, x2, x3])