#!/usr/bin/python
import logging
import abc
from collections import defaultdict
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType


class Network(object):
    """Base class for Network"""

    def __init__(self):
        __metaclass__ = abc.ABCMeta
        self.concepts = dict()
        self.label = None

    def _select_features(self, concept_key, features):
        """Select and translate features for this concept_key"""
        cfeatures = dict()
        for k, v in features.items():
            if k.startswith(concept_key):
                cfeatures['_'.join(k.split('_')[1:])] = v
        return cfeatures

    def get_vectors(self, features):
        vectors = dict()
        for cname, concept in self.concepts.items():
            vectors[cname] = concept.inference(
                self._select_features(cname, features))
        return vectors

    def __repr__(self):
        return self.__class__.__name__ + self.version + '__' + \
            '_'.join([str(c) for c in self.concepts.values()])

    @abc.abstractmethod
    def _get_test_input(self):
        """This function will return a test input for encoding,
        this function is only required for testing"""
        raise NotImplementedError(
            'The base class needs to implement "get_test_input"')

    @abc.abstractmethod
    def preprocess(self, raw_input):
        """This function is responsible for encoding an raw_input object to a
        TensorFlow Example object
        Defaults to a concatenation of the encodings of the multiple
        concepts"""
        features = dict()
        for cname, concept in self.concepts.items():
            cfeatures = concept.preprocess(raw_input)
            for k, v in cfeatures.items():
                features[cname + '_' + k] = v
        if self.label is not None:
            features['label'] = label.preprocess(raw_input)
        example = tf.train.Example(
            features=tf.train.Features(feature=features))
        return example

    @abc.abstractmethod
    def featdef(self):
        """This function defines thet interface between preprocess and inference:
        Returns a dictionary of feature names and tf FeatureEncodingTypes
        Defaults to a concatenation of the encodings of the multiple
        concepts"""
        features_def = dict()
        for cname, concept in self.concepts.items():
            cfeatures_def = concept.featdef()
            for k, v in cfeatures_def.items():
                features_def[cname + '_' + k] = v
        if self.label is not None:
            features_def['label'] = label.featdef()
        return features_def

    @abc.abstractmethod
    def inference(self, features):
        """This function takes a dictionary of tensors(features) and
        returns the logits calculate with this network"""
        raise NotImplementedError(
            'The base class needs to implement "inference"')

    @abc.abstractmethod
    def loss(self, predictions, labels):
        """This function calculates the loss between the labels and the predictions
        (output from inference)"""
        raise NotImplementedError(
            'The base class needs to implement "loss"')

    def train(self, loss):
        """This function takes a dictionary of tensors(features) and
        specifies the Tensorflow operations to transform these into a single tensor"""
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.1,
            optimizer='Adagrad')
        return train_op

    def build_model_fn(self):
        """Build model function.

        Returns:
          A model function that can be passed to `Estimator` constructor.
        """
        def _model_fn(features, labels, mode, input_dir):
            """Creates the prediction and its loss.

            Args:
              features: A dictionary of tensors keyed by the feature name.
              labels: A tensor representing the labels.
              mode: The execution mode, as defined in tf.contrib.learn.ModeKeys.

            Returns:
              A tuple consisting of the prediction, loss, and train_op.
            """
            predictions = self.inference(features)
            if mode == tf.contrib.learn.ModeKeys.INFER:
                return predictions, None, None

            loss = self.loss(predictions, labels)
            if mode == tf.contrib.learn.ModeKeys.EVAL:
                return predictions, loss, None

            train_op = self.train(loss)
            return predictions, loss, train_op

        return _model_fn

    def build_input_fn(self, batch_size, mode, input_dir):
        """Build input function.

        Args:
          batch_size: Batch size
          mode: The execution mode, as defined in tf.contrib.learn.ModeKeys.
          input_dir: path containing the input files

        Returns:
          Tuple, dictionary of feature column name to tensor and labels.
        """
        def _input_fn():
            """Supplies the input to the model.

            Returns:
              A tuple consisting of 1) a dictionary of tensors whose keys are
              the feature names, and 2) a tensor of target labels if the mode
              is not INFER (and None, otherwise).
            """
            input_files = sorted(list(tf.gfile.Glob(input_dir)))
            logging.info("Reading files from %s", input_dir)
            include_target_column = (mode != tf.contrib.learn.ModeKeys.INFER)

            reader_fn = tf.TFRecordReader(
                options=tf.python_io.TFRecordOptions(
                    compression_type=TFRecordCompressionType.GZIP))

            features = tf.contrib.learn.io.read_batch_features(
                file_pattern=input_dir,
                batch_size=batch_size,
                queue_capacity=3 * batch_size,
                randomize_input=mode == tf.contrib.learn.ModeKeys.TRAIN,
                feature_queue_capacity=5,
                reader=reader_fn,
                features=self.featdef())
            target = None
            if include_target_column:
                target = features.pop('label')
            return features, target

        return _input_fn

    @classmethod
    def _test(cls):
        """test is resposible for testing a Network class"""
        self = cls()
        logging.info('\n' + '*' * 50 + '\n' + '*' * 50)
        logging.info('\n' + '*' * 50 + '\n' + '*' * 50)
        logging.info('Test Network : %s', self)
        example = self.preprocess(self._get_test_input())
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
            vector = embedding["logits"].eval()
            logging.info('vector : %s', str(vector))
