#!/usr/bin/python

import argparse
import datetime
import logging
import os
from itertools import combinations

import apache_beam as beam
from apache_beam.io import fileio
from apache_beam.typehints import typehints
from apache_beam.pvalue import AsSingleton
from apache_beam.utils.pipeline_options import GoogleCloudOptions
from apache_beam.utils.pipeline_options import PipelineOptions
from apache_beam.utils.pipeline_options import SetupOptions
from apache_beam.utils.pipeline_options import WorkerOptions

import tensorflow as tf

import google.cloud.ml as ml
import google.cloud.ml.dataflow.io.tfrecordio as tfrecordio
from google.cloud.ml.io.coders import ExampleProtoCoder

from network import MinimalNetwork


def run(known_args, pipeline_args):
    network = MinimalNetwork()

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    pipeline_options.view_as(SetupOptions).extra_packages = [ml.sdk_location]
    pipeline_options.view_as(
        WorkerOptions).autoscaling_algorithm = 'THROUGHPUT_BASED'
    pipeline_options.view_as(GoogleCloudOptions).staging_location = \
        os.path.join(known_args.staging)
    pipeline_options.view_as(GoogleCloudOptions).temp_location = os.path.join(
        known_args.staging, 'tmp')
    pipeline_options.view_as(GoogleCloudOptions).job_name = str(network).replace('_','').lower()

    beam.coders.registry.register_coder(tf.train.Example, ExampleProtoCoder)
    p = beam.Pipeline(options=pipeline_options)

 

    # Read Example data
    def parse_example(example):
        #TODO: add actual implementation
        yield example

    network_input = (p
                  | 'readExamples' >> beam.io.ReadFromText(
                      known_args.input)
                  | 'processExamples' >> beam.FlatMap(
                      lambda example: parse_example(example)))

    examples = network_input | 'encodeExamples' >> beam.Map(
        lambda raw_input: network.preprocess(raw_input))

    # # Write the serialized compressed protocol buffers to Cloud Storage.
    _ = examples | beam.io.Write(
        'writeExamples',
        tfrecordio.WriteToTFRecord(
            file_path_prefix=os.path.join(known_args.output, 'examples'),
            compression_type=fileio.CompressionTypes.GZIP,
            coder=ExampleProtoCoder(),
            file_name_suffix='.tfrecord.gz'))

    # # Actually run the pipeline (all operations above are deferred).
    p.run()
