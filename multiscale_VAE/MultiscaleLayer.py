from keras import layers

def MSConv2D(initial, nodes, kernels):
    '''
    Creates layers for Multi-scale 2D Convolution w/ 3 different kernel sizes
    '''
    # Kernel 1
    x1 = layers.Conv2D(nodes, (kernels[0],kernels[0]), activation="relu", padding="same")(initial)
    x1 = layers.MaxPooling2D((2, 2), padding="same")(x1)
    
    # Kernel 2
    x2 = layers.Conv2D(nodes, (kernels[1],kernels[1]), activation="relu", padding="same")(initial)
    x2 = layers.MaxPooling2D((2, 2), padding="same")(x2)
    
    # Kernel 3
    x3 = layers.Conv2D(nodes, (kernels[2],kernels[2]), activation="relu", padding="same")(initial)
    x3 = layers.MaxPooling2D((2, 2), padding="same")(x3)
    
    # Sum together
    return layers.Add()([x1, x2, x3])

def MSConv2DTranspose(initial, nodes, kernels):
    '''
    Creates layers for Multi-scale 2D Deconvolution w/ 3 different kernel sizes 
    '''
    # Kernel 1
    x1 = layers.Conv2DTranspose(nodes, (kernels[0],kernels[0]), strides=2, activation="relu", padding="same")(initial)
    
    # Kernel 2
    x2 = layers.Conv2DTranspose(nodes, (kernels[1],kernels[1]), strides=2, activation="relu", padding="same")(initial)
    
    # Kernel 3
    x3 = layers.Conv2DTranspose(nodes, (kernels[2],kernels[2]), strides=2, activation="relu", padding="same")(initial)
    
    # Sum together
    return layers.Add()([x1, x2, x3])