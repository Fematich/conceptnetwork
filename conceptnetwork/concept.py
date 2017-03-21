#!/usr/bin/python

import logging
import re
import abc


class Concept(object):
    """Base class for Concepts"""

    def __init__(self, target=False, checkpoint_path=None):
        __metaclass__ = abc.ABCMeta
        if target and checkpoint_path:
            raise ValueError("Concept cannot be restored from checkpoint if it's a target")
        self.target = target
        self.checkpoint_path = checkpoint_path

    def __repr__(self):
        return self.__class__.__name__ + self.version.replace('.', '')

    def _short_repr(self):
        abrev = ''.join([s[:2] for s in re.split(
            "([A-Z][^A-Z]*)", self.__class__.__name__) if s])
        return abrev + self.version.replace('.', '')

    @abc.abstractmethod
    def _get_test_input(self):
        """This function will return a test input for encoding,
        this function is only required for testing"""
        raise NotImplementedError(
            'The base class needs to implement "get_test_input"')

    @abc.abstractmethod
    def preprocess(self, raw_input):
        """This function is responsible for encoding an raw_input object to a dict
        of feature names and a corresponding TensorFlow Example protobuffers"""
        raise NotImplementedError(
            'The base class needs to implement "encode"')

    @abc.abstractmethod
    def featdef(self):
        """This function defines thet interface between encode and model:
        Returns a dictionary of feature names and tf FeatureEncodingTypes"""
        raise NotImplementedError(
            'The base class needs to implement "featdef"')

    def feature_engineering(self, features):
        """This function does preprocessing/feature engineering for a single
        concept"""
        return features

    @abc.abstractmethod
    def inference(self, features):
        """This function takes a dictionary of tensors(features) and specifies
        the Tensorflow operations to transform these into a single tensor"""
        raise NotImplementedError(
            'The base class needs to implement "model"')

    @classmethod
    def get_children(cls):
        """Helper function to get the list of all children in the global
        namespace"""
        import sys
        import inspect
        subclasses = []
        callers_module = sys._getframe(1).f_globals['__name__']
        classes = inspect.getmembers(
            sys.modules[callers_module], inspect.isclass)
        for name, obj in classes:
            if (obj is not cls) and (cls in inspect.getmro(obj)):
                subclasses.append((name, obj))
        return subclasses

    @classmethod
    def _test(cls):
        """test is resposible for testing an individual Concept class"""
        import tensorflow as tf
        temp_file = '../data/test_concept.tfrecords'
        num_examples = 32
        self = cls()
        logging.info('\n' + '*' * 50 + '\n' + '*' * 50)
        logging.info('Test Concept : %s', self)
        features = self.preprocess(self._get_test_input())
        example = tf.train.Example(
            features=tf.train.Features(feature=features))
        serialized_example = example.SerializeToString()
        writer = tf.python_io.TFRecordWriter(temp_file)
        for _ in range(num_examples):
            writer.write(example.SerializeToString())
        writer.close()
        logging.info('Successfully serialized %i tfrecord(s)' % num_examples)

        reader = tf.TFRecordReader()
        filename_queue = tf.train.string_input_producer(
            [temp_file], num_epochs=1)
        _, serialized_example = reader.read(filename_queue)
        batch = tf.train.batch([serialized_example],
                               num_examples, capacity=num_examples)
        reconstructed_features = tf.parse_example(
            batch,
            features=self.featdef(),
            name='reconstruct_features')
        logging.info('Successfully reconstructed tfrecord')

        embedding = self.inference(reconstructed_features)
        with tf.Session() as sess:
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                vector = embedding.eval()
                logging.info('vector : %s', str(vector))
            except tf.errors.OutOfRangeError as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
