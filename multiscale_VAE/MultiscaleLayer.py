from keras import layers
from keras.layers import Layer
from keras.models import Model
from keras.models import load_model

class MSConv2D(Layer):
    
    def __init__(self, filters=32, kernels=[1,3,5]):
        super(MSConv2D, self).__init__()
        
        # Make 3 Conv2D Layers
        self.Conv1 = layers.Conv2D(filters, (kernels[0],kernels[0]), activation="relu", padding="same")
        self.Conv2 = layers.Conv2D(filters, (kernels[1],kernels[1]), activation="relu", padding="same")
        self.Conv3 = layers.Conv2D(filters, (kernels[2],kernels[2]), activation="relu", padding="same")
    
    def call(self, inputs):
        '''
        Compute each Convolution separately then add together
        
        Maybe add Batch Normalization Layers after each MaxPool
        '''
        # Kernel 1
        x1 = self.Conv1(inputs)
        x1 = layers.MaxPooling2D((2, 2), padding="same")(x1)
        
        # Kernel 2
        x2 = self.Conv2(inputs)
        x2 = layers.MaxPooling2D((2, 2), padding="same")(x2)
        
        # Kernel 3
        x3 = self.Conv3(inputs)
        x3 = layers.MaxPooling2D((2, 2), padding="same")(x3)
        
        # Add together
        return layers.Add()([x1, x2, x3])
    
class MSConv2DTranspose(Layer):
    
    def __init__(self, filters=32, kernels=[1,3,5]):
        super(MSConv2DTranspose, self).__init__()
        
        # Conv2D Transposes
        self.Conv1 = layers.Conv2DTranspose(filters, (kernels[0],kernels[0]), strides=2, activation="relu", padding="same")
        self.Conv2 = layers.Conv2DTranspose(filters, (kernels[1],kernels[1]), strides=2, activation="relu", padding="same")
        self.Conv3 = layers.Conv2DTranspose(filters, (kernels[2],kernels[2]), strides=2, activation="relu", padding="same")
        
    def call(self, inputs, *args, **kwargs):
        '''
        Compute each Convolution Transpose separately then add together
        
        Maybe add Batch Normalization Layers after each MaxPool
        '''
        # Kernel 1
        x1 = self.Conv1(inputs)
        
        # Kernel 2
        x2 = self.Conv2(inputs)
        
        # Kernel 3
        x3 = self.Conv3(inputs)
        
        return layers.Add()([x1, x2, x3])