import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# height x width = shape
# we can tell the shape or not it's optional
#if some out of the shape thing comes it will throw an error
x = tf.placeholder('float',[None, 784]) #784 is the width
y = tf.placeholder('float')

def neural_network_model(data):
    # (input_data * weights) + biases
    # we use biases to avoid the NULL condition
    # If all the input are zero and if there is no bias, no neron will fire

    # Difference between random normal and truncated normal is that it eliminates tail values 
    hidden_1_layer = {'weights':tf.Variable(tf.truncated_normal([784,n_nodes_hl1],stddev=0.1)),
                      'biases':tf.constant(0.1,shape=([n_nodes_hl1]))}
    
    hidden_2_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1,n_nodes_hl2],stddev=0.1)),
                      'biases':tf.constant(0.1,shape=([n_nodes_hl2]))}
    
    hidden_3_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl2,n_nodes_hl3],stddev=0.1)),
                      'biases':tf.constant(0.1,shape=([n_nodes_hl3]))}                   

    output_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl3,n_classes],stddev=0.1)),
                      'biases':tf.constant(0.1,shape=([n_classes]))}

    # (input_data * weights) + biases
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
    #cost function calculates how far is the prediction is from the actual value

    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    # It has parameter of learning rate which has default value of 0.001 that is enough for us for now
    #Optimizer is used to minimize cost to increase accuracy

    hm_epochs = 10 #cycles feed forward + backprop

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer,cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs,'loss:',epoch_loss)
        
        # correct will tell us how many predictions we made that were perfect matches to their labels.
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:' ,accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)




