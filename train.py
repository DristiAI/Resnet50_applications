import tensoflow as tf
import numpy as np
from tensorflow.python.data import Dataset
from resnet import Resnet,compute_loss,train_op

"""
CONSTANTS
"""
BATCH_SIZE=32
EPOCHS=20


#loading the training and the test data
X_train,Y_train,X_test,Y_test,classes=load_dataset()
num_classes=len(classes)
x1,y1,c1=X_train.shape

# defining placeholder
X=tf.placeholder(tf.float32,[None,x1,y1,c1])
Y=tf.placeholder(tf.float32,[None,num_classes])

#making dataset iterator
ds=Dataset.from_tensor_slices((X,Y)).batch(BATCH_SIZE).repeat()
iterator=tf.make_initializable_iterator()
sess.run(iterator.initializer,feed_dict={X:X_train,Y:Y_train})
batch_x,batch_y=iterator.get_next()

Y_pred=Resnet(batch_x,num_classes)
loss_op=train_op(Y_pred,batch_y)
optimizer-train_op(loss,learning_rate):
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y_pred,1),tf.argmax(Y_pred,1)),tf.float32))
init=tf.global_variables_initializer()
sess.run(init)
num_batches=X_train.shape[0]/BATCH_SIZE
for epoch in num_epochs:
    for _ in num_batches:
        _,accuracy,loss,=sess.run([optimizer,accuracy,loss])
    if epoch%2==0:
        print('Accuracy after {1} epoch is {0} '.format(accuracy,epoch))
        
    