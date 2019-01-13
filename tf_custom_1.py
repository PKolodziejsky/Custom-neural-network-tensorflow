import tensorflow as tf
import os
import numpy as np
import sklearn

tf.enable_eager_execution()

#Import tensorflow training csv file
train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
train_dataset_file = tf.keras.utils.get_file(fname = os.path.basename(train_dataset_url),origin =train_dataset_url)

columns =['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
features = columns[:-1]
labels = columns[-1]

batch_size = 32

train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_file,
    batch_size,
    column_names=columns,
    label_name=labels,
    num_epochs=1)

def features_vector(features, labels):
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

train_dataset = train_dataset.map(features_vector)

#Class for creating custom dense layers
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[int(input_shape[-1]),
                                               self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)

model = tf.keras.Sequential([MyDenseLayer(10),
                            MyDenseLayer(10),
                            MyDenseLayer(3)])

def loss(model ,x,y):
    return tf.losses.sparse_softmax_cross_entropy(labels = y ,logits = model(x) )

def gradient(model , inputs , desired):
    with tf.GradientTape() as gt:
        loss_value  = loss(model , inputs , desired)
    return loss_value, gt.gradient(loss_value , model.trainable_variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

global_step = tf.Variable(0)

epochs_num = 200
for epoch in range(epochs_num):

    for x,y in train_dataset:

        loss_value ,gradients = gradient(model , x,y)
        optimizer.apply_gradients(zip(gradients,model.variables) , global_step)
        epoch_loss_avg = np.mean (loss_value)
        epoch_accuracy = sklearn.metrics.accuracy_score(tf.argmax(model(x) ,axis=1, output_type=tf.int32), y)

    if epoch % 2 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg,epoch_accuracy))

print("Final:" ,"Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg,epoch_accuracy))
