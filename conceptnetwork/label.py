#!/usr/bin/python

import logging
from collections import defaultdict
import abc


class Label(object):
    """Base class for Label"""

    def __init__(self):
        __metaclass__ = abc.ABCMeta

    def __repr__(self):
        return self.__class__.__name__ + self.version

    @abc.abstractmethod
    def _get_test_input(self):
        """This function will return a test input for encoding,
        this function is only required for testing"""
        raise NotImplementedError(
            'The base class needs to implement "get_test_input"')

    @abc.abstractmethod
    def preprocess(self, raw_input):
        """This function is responsible for encoding an raw_input object to a 
        dict of a single feature names and corresponding TensorFlow Example protobuffer"""
        raise NotImplementedError(
            'The base class needs to implement "encode"')

    @abc.abstractmethod
    def featdef(self):
        """This function defines thet interface between encode and model:
            Returns a dictionary of feature names and tf FeatureEncodingTypes"""
        raise NotImplementedError(
            'The base class needs to implement "featdef"')

    @classmethod
    def get_children(cls):
        """Helper function to get the list of all children in the global namespace"""
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

        self = cls()
        logging.info('\n' + '*' * 50 + '\n' + '*' * 50)
        logging.info('Test Concept : %s', self)
        features = self.preprocess(self._get_test_input())
        example = tf.train.Example(
            features=tf.train.Features(feature=features))
        serialized_example = example.SerializeToString()
        logging.info('Successfully serialized tfrecord')

        reconstructed_features = tf.parse_single_example(
            serialized_example,
            features=self.featdef(),
            name='reconstruct_features')
        logging.info('Successfully reconstructed tfrecord')

        embedding = self.inference(reconstructed_features)
        with tf.Session():
            tf.global_variables_initializer().run()
            vector = embedding.eval()
            logging.info('vector : %s', str(vector))
