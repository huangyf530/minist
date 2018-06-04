from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import
import tensorflow as tf
import numpy as np


tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """mode function for cnn"""
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same",
                             activity_regularizer=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=5, padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=10)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits=logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    #load training and eval data
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="D:\\AIHomeWork\\hw3\\tmp\\my_model")
    '''train_data = {}
    train_labels = []
    eval_data = {}
    eval_labels = []
    feature_data = []
    eval_feature_data = []
    count = 0
    with open("train.csv", 'r') as f:
        for (linenum, content) in enumerate(f):
            count += 1
            if linenum == 0:
                continue
            line = content.split(',')
            one_data = []
            for num in line[1:]:
                one_data.append(float(int(num) / 255))
            if count <= 30000:
                feature_data.append(one_data)
                train_labels.append(int(line[0]))
            if count > 30000:
                eval_feature_data.append(one_data)
                eval_labels.append(int(line[0]))
        print("一共有%d行" % count)
    train_data['x'] = np.array(feature_data)
    eval_data['x'] = np.array(eval_feature_data)
    np_eval_labels = np.array(eval_labels)
    np_train_labels = np.array(train_labels)
    tensors_to_log = {"probabilities" : "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    #Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_data,
        y=np_train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    #mnist_classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])

    #Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=eval_data,
        y=np_eval_labels,
        num_epochs=1,
        shuffle=False
    )
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)'''
    predict_data = []
    with open("test.csv", 'r') as f:
        for (linenum, content) in enumerate(f):
            if linenum == 0:
                continue
            line = content.split(',')
            one_data = []
            for number in line:
                one_data.append(float(int(number)/255))
            predict_data.append(one_data)
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': np.array(predict_data)}, num_epochs=1, shuffle=False)
    predictions = list(mnist_classifier.predict(input_fn=pred_input_fn))
    cnt = 0
    with open ('submission.csv', 'w') as f:
        f.write('ImageId,Label\n')
        for pre in predictions:
            cnt += 1
            f.write("%d,%d\n" % (cnt, pre['classes']))


if __name__ == "__main__":
    tf.app.run()