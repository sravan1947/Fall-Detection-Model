import pandas as pd
import numpy as np
from pandas import DataFrame as df
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
walking=pd.read_excel('E:/Engineering/python/vesaithoncode/walking.xlsx') #dataset from saved position 
fall_data=pd.read_excel('E:/Engineering/python/vesaithoncode/falling.xlsx') #dataset from saved position
print(len(fall_data))
from sklearn.utils import shuffle
fall_data = shuffle(fall_data)
y_data=fall_data['output']
x_data=fall_data.drop('output',axis=1)
#x_data.drop(['Ax','Ay','Az'],axis=1,inplace=True)
x_train=x_data[:15539]
x_test=x_data[15539:17539]
y_train=y_data[:15539]
y_test=y_data[15539:17539]
train_scaled=scalar.fit_transform(x_train)
test_scaled=scalar.transform(x_test)
def next_batch(x,y,sp,batch_size):
    if(sp+batch_size<len(x)):
        return(x[sp:sp+batch_size].reshape(-1,batch_size,6),y[sp:sp+batch_size].reshape(-1,batch_size,1))
    return(x[sp:].reshape(-1,len(x[sp:]),6),y[sp:].reshape(-1,len(x[sp:]),1))
num_of_inputs=6
num_of_outputs=1
neurons=300
l_rate=0.1
num_of_iteration=30000
batch_size=1
x=tf.placeholder(tf.float32,[None,batch_size,num_of_inputs])
y=tf.placeholder(tf.float32,[None,batch_size,1])
cell=tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units=neurons,activation=tf.nn.sigmoid,forget_bias=2.0),output_keep_prob=0.5)
cell=tf.contrib.rnn.OutputProjectionWrapper(cell,output_size=num_of_outputs)
outputs,states=tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
tv = tf.trainable_variables()
regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in tv ])
cross_entropy=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=outputs)) + regularization_cost
optimizer=tf.train.AdamOptimizer(learning_rate=l_rate)
train=optimizer.minimize(cross_entropy)
init=tf.global_variables_initializer()
saver=tf.train.Saver()
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    s=0
    for i in range(num_of_iteration):
        batch_x,batch_y=next_batch(train_scaled,np.array(y_train),s,batch_size)
        sess.run(train,feed_dict={x:batch_x,y:np.array(batch_y)})
        s+=batch_size
        if(s==15539):
            s=0
    saver.save(sess,'./fall risk prediction')
new_data=fall_data
l=[]
with tf.Session() as sess:
    saver.restore(sess,'./fall risk prediction')
    s=0
    while(s!=len(new_data)):
        batch_x,batch_y=next_batch(test_scaled,np.array(y_test),s,batch_size)
        y_pred=sess.run(outputs,feed_dict={x:batch_x})
        l.append(y_pred)
        s=s+1
k=np.array(l).reshape(len(new_data))
print(k)
pred=[]
for i in k:
    if i>0:
        pred.append(1)
    else:
        pred.append(0)
print(pred)
