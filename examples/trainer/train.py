#!/usr/bin/python
import logging
import tensorflow as tf

from networks.minimalnetwork import MinimalNetwork

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 32, 'Number of steps to run trainer.')
flags.DEFINE_string(
    "input_dir", None, "Path to the input files.")
flags.DEFINE_string(
    "output_dir", None, "Path to the output files.")
FLAGS = flags.FLAGS


def main(unused_argv):
    network = MinimalNetwork()
    input_fn = network.build_input_fn(
        batch_size=FLAGS.batch_size, mode=tf.contrib.learn.ModeKeys.TRAIN,
        input_dir=FLAGS.input_dir)
    model_fn = network.build_model_fn()
    feature_engineering_fn = network.feature_engineering_fn

    estimator = tf.contrib.learn.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.output_dir,
        feature_engineering_fn=feature_engineering_fn)

    estimator.fit(input_fn=input_fn, max_steps=FLAGS.max_steps)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    tf.app.run()
