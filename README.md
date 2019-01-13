# Custom-neural-network-tensorflow
Tensorflow neural network model built with custom training methods.

# Tensorflow functions

tf.enable_eager_execution() - eager execution is an important programming environment that evaluates operations immediately,
without building graphs: operations return concrete values instead of constructing a computational graph to run later.

tf.contrib.data.make_csv_dataset(...) - Reads CSV files into a dataset. However it will be removed in future version of tf and
substituted by tf.data.experimental.make_csv_dataset(...).

tf.stack() - Packs the list of tensors in values into a tensor with rank one higher than each tensor in values,
by packing them along the axis dimension.

tf.matmul() - Multiplies matrix x by matrix y

tf.losses.sparse_softmax_cross_entropy() - Computes sparse softmax cross entropy between logits and labels.

tf.train.GradientDescentOptimizer() - Optimizer that implements the gradient descent algorithm.

# Loss and accuracy measures

Loss is a simple numpy.mean of loss_value.

Accuracy is measured by sklearn.metrics.accuracy_score of model(x) which is a model output and y which is a target.

# References

https://www.tensorflow.org/api_docs/
