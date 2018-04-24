import tensorflow as tf
import numpy as np
from resnetblocks import resnet_block

def Resnet(X,num_classes):
    """
    implements Resnet 50 architecture
    """
    
    X=tf.keras.layers.ZeroPadding2D(padding=(3,3))(X)
    X=tf.layers.conv2d(inputs=X,filters=64,kernel_size=(7,7),strides=(2,2))
    X=tf.layers.batch_normalisation(inputs=X,axis=-1)
    X=tf.nn.relu(X)
    X=tf.layers.max_pooling2d(inputs=X,pool_size=(3,3),strides=(2,2))
    
    """
    using resnets blocks from here
    """
    
    X=resnet_block(X,F1=64,F2=64,F3=256,f=3,s=1)
    X=resnet_block(X,F1=64,F2=64,F3=256,f=3,block='identity')
    X=resnet_block(X,F1=64,F2=64,F3=256,f=3,block='identity')
    
    X=resnet_block(X,F1=128,F2=128,F3=512,f=3,s=1)
    X=resnet_block(X,F1=128,F2=128,F3=512,f=3,block='identity')
    X=resnet_block(X,F1=128,F2=128,F3=512,f=3,block='identity')
    X=resnet_block(X,F1=128,F2=128,F3=512,f=3,block='identity')
    
    X=resnet_block(X,F1=256,F2=256,F3=1024,f=3,s=2)
    X=resnet_block(X,F1=256,F2=256,F3=1024,f=3,block='identity')
    X=resnet_block(X,F1=256,F2=256,F3=1024,f=3,block='identity')
    X=resnet_block(X,F1=256,F2=256,F3=1024,f=3,block='identity')
    X=resnet_block(X,F1=256,F2=256,F3=1024,f=3,block='identity')
    X=resnet_block(X,F1=256,F2=256,F3=1024,f=3,block='identity')
    
    X=resnet_block(X,F1=512,F2=512,F3=2048,f=3,s=2)
    X=resnet_block(X,F1=512,F2=512,F3=2048,f=3,block='identity')
    X=resnet_block(X,F1=512,F2=512,F3=2048,f=3,block='identity')
    
    X=tf.layers.average_pooling2d(inputs=X,pool_size=(2,2))
    X=tf.layers.flatten(inputs=X)
    X_out=tf.layers.dense(inputs=X,num_classes,activation=tf.nn.softmax)
    
    return X_out

def compute_loss(logits,Y)
    
    return tf.reduce_mean(tf.losses.softmax_cross_entropy(Y,logits))

def train_op(loss,learning_rate):
    
    optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    return optimizer


    
    
    