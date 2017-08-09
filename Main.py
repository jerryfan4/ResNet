
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
from ResNet import *
from Cifar10 import *
import cPickle
import matplotlib.pyplot as plt
import numpy as np
import os


# ## Configuration

# In[2]:

num_units = 18
exp_id = 0
gpu_number = 0
epoch = 400
image_shape = [32, 32, 3]
train_batch_size = 128
test_batch_size = 125
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)
    
all_train_loss, all_train_error = [], []
all_test_loss, all_test_error = [], []

exp_dir = os.path.join('.', 'exp{}'.format(exp_id))
if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)
log_file_path = os.path.join(exp_dir, 'log_exp{}.txt'.format(exp_id))
log_file = open(log_file_path, 'w+')


# ## Log

# In[3]:

def log(line):
    log_file.write(line)
    log_file.write('\n')
    log_file.flush()
    print line


# ## Main function

# In[ ]:

def main(sess):
    dataset = Cifar10(train_batch_size, test_batch_size)
    model = ResNet(num_units, image_shape, train_batch_size, test_batch_size)
    train_op, train_loss, train_accuracy = model.build_train_op()
    test_loss, test_accuracy = model.build_test_op()
    
    global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='')
    log('Global variables:')
    for i, var in enumerate(global_variables):
        log('{0} {1} {2}'.format(i, var.name, var.get_shape()))
    
    all_initializer_op = tf.global_variables_initializer()
    sess.run(all_initializer_op)
    
    saver = tf.train.Saver(max_to_keep=5000)
    
    for i in range(epoch):  
        total_loss = 0.0
        total_accuracy = 0.0
        dataset.shuffle_dataset()
        for j in range(dataset.train_batch_count):
            batch_images, batch_labels = dataset.next_aug_train_batch(j)
            
            sess.run(train_op,
                     feed_dict = {model.train_image_placeholder: batch_images, 
                                  model.train_label_placeholder: batch_labels})
            curr_loss, curr_accuracy = sess.run([train_loss,train_accuracy],
                                                feed_dict = {model.train_image_placeholder: batch_images, 
                                                             model.train_label_placeholder: batch_labels})
            #sess.run(train_step_op)
            total_loss += curr_loss
            total_accuracy += curr_accuracy
        
        total_loss /= dataset.train_batch_count
        total_accuracy /= dataset.train_batch_count
        
        all_train_loss.append(total_loss)
        all_train_error.append(1.0 - total_accuracy)
        
        log('Training epoch {0}, step {1}, learning rate {2}'.
            format(i, sess.run(model.train_step), sess.run(model.learning_rate)))
        log('    train loss {0}, train error {1}'.format(total_loss, 1.0 - total_accuracy))
            

        total_loss = 0.0
        total_accuracy = 0.0
        for k in range(dataset.test_batch_count):
            batch_images, batch_labels = dataset.next_test_batch(k)
                
            curr_loss, curr_accuracy = sess.run([test_loss, test_accuracy],
                                                feed_dict = {model.test_image_placeholder: batch_images,
                                                             model.test_label_placeholder: batch_labels})
            total_loss += curr_loss
            total_accuracy += curr_accuracy
            
        total_loss /= dataset.test_batch_count
        total_accuracy /= dataset.test_batch_count
        
        all_test_loss.append(total_loss)
        all_test_error.append(1.0 - total_accuracy)

        log('    test loss {0}, test_error {1}'.format(total_loss, 1.0 - total_accuracy))
        
        save_model_file = os.path.join(exp_dir, 'ResNet-model')
        if i % 20 == 0:
            saver.save(sess, save_model_file, global_step=model.train_step)
        
    


# In[ ]:

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
graph = tf.Graph()
with graph.as_default():
    with tf.Session(config=config) as sess:
        main(sess)


# In[ ]:

plt.figure(1)
plt.title('Loss curve')
plt.xlabel('Training epochs')
plt.ylabel('Loss')
plt.gca().yaxis.grid(True)
plt.plot(all_train_loss, label='training')
plt.plot(all_test_loss, label='testing')
plt.legend(loc='upper right')
loss_file = os.path.join(exp_dir, 'loss.png')
plt.savefig(loss_file)


# In[ ]:

plt.figure(2)
plt.title('Error curve')
plt.xlabel('Training epochs')
plt.ylabel('Error')
plt.gca().yaxis.grid(True)
plt.plot(all_train_error, label='training')
plt.plot(all_test_error, label='testing')
plt.legend(loc='upper right')
error_file = os.path.join(exp_dir, 'error.png')
plt.savefig(error_file)


# In[ ]:

plt.close('all')
log_file.close()

