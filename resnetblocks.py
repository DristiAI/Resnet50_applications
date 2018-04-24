import tensorflow as tf
import numpy as np

def resnet_block(X,F1,F2,F3,f=None,s=None,block='convolutional'):
    
    """
        implements convolutional block
        F1,F2,F3 are filters for convolutions
        
    """
    
    X_=X
    
    if(block='convolutional'):
        
        #first component
        X=tf.layers.conv2d(inputs=X,filters=F1,kernel_size=[1,1],strides=[s,s])
        X=tf.batch_normalization(inputs=X,axis=-1)
        X=tf.nn.relu(X)

        #second component
        X=tf.layers.conv2d(inputs=X,filters=F2,kernel_size=[f,f],padding='same')
        X=tf.batch_normalization(inputs=X,axis=-1)
        X=tf.nn.relu(X)

        #third component
        X=tf.layers.conv2d(inputs=X,filetrs=F3,kernel_size=[1,1],strides=[1,1],padding='valid')
        X=tf.batch_normalization(inputs=X,axis=-1)


        #the shortcut path
        X_shortcut=tf.layers.conv2d(inputs=X_,filters=F3,kernel_size=[1,1],strides=[s,s],padding='valid')
        X_shortcut=tf.layers.batch_normalization(X_shortcut,axis=-1)
        X=tf.nn.relu(X+X_shortcut)

        return X
    
    elif(block='identity'):
        
        #1st component
        X=tf.layers.conv2d(inputs=X,filters=F1,kernel_size=[1,1],strides=[1,1],padding='valid')
        X=tf.layers.batch_normalization(X,axis=-1)
        X=tf.nn.relu(X)
        
        #2nd component
        X=tf.layers.conv2d(inputs=X,filters=F2,kernel_size=[f,f],strides[1,1],padding='same',kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))
        X=tf.layers.batch_normlization(X,axis=-1)
        X=tf.nn.relu(X)
        
        #3rd component
        X=tf.layers.conv2d(inputs=X,filters=F3,kernel_size=[f,f],strides=[1,1],padding='valid',kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))

        #the shortcut path
        X=tf.nn.relu(X+X_shortcut)
        
        return X
        

    
    