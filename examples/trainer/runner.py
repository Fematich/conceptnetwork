import sys
import logging
import subprocess
import argparse

from conceptnetwork import Concept

from networks.minimal_network import MinimalNetwork
from concepts.minimal_concept import MinimalConcept


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runner of MinimalNetwork')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--nogit', action='store_true')
    parser.add_argument('--get_data', action='store_true')
    parser.add_argument('--notest', action='store_true')
    parser.add_argument('--notf', action='store_true')
    parser.add_argument('--dfnetwork', type=str, help='reuse output from older DF network',
                        default=None, required=False)
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    network = MinimalNetwork()
    
    #############################################
    ## Data sync ################################
    #############################################
    if args.get_data:
        logging.info("Download data from GCS")
        data_locations = [
            "gs://<example_bucket_locations>",
        ]
        for dat in data_locations:
            logging.info("... " + dat)
            subprocess.call(["gsutil", "cp", dat, "../data"])
        sys.exit()

    #############################################
    ## Testing ##################################
    #############################################
    if not args.notest:
        logging.info("Test Individual Concept Classes")
        for name, cls in Concept.get_children():
            cls._test()

        logging.info("Test Network Class")
        network._test()

    #############################################
    ## Git ######################################
    #############################################
    if not args.nogit:
        logging.info("Git push: %s" % str(network))
        subprocess.call("git add .".split(" "))
        subprocess.call(["git", "commit", "-am", str(network)])
        subprocess.call("git pull origin master".split(" "))
        subprocess.call("git push origin master".split(" "))

    #############################################
    ## Dataflow #################################
    #############################################
    if args.dfnetwork is None:
        logging.info("Preprocessing with DataFlow")
        from networks.candidate_ae import preprocess
        if args.local:
            runner = "DirectRunner"
            input = "../data/cvs_sample.xml"
        else:
            runner = 'BlockingDataflowRunner'
            input = "gs://<example_bucket>/*"
        parser = argparse.ArgumentParser()
        parser.add_argument('--cand_input')
        parser.add_argument('--match_input')
        parser.add_argument('--output')
        parser.add_argument('--staging')
        known_args = parser.parse_args(
            ['--input', cinput,
             '--output', "gs://<example_bucket>/minimalnetwork/runs/%s/tfrecords"
                % str(network),
             '--staging', "gs://<example_bucket>/minimalnetwork/%s" % str(network)])

        pipeline_args = ['--setup_file', './setup.py',
                         '--project', '<gcp-project>',
                         '--runner', runner,
                         '--zone', 'europe-west1-d',
                         '--worker_machine_type', 'n1-standard-1']
        preprocess.run(known_args, pipeline_args)

    #############################################
    ## TensorFlow ###############################
    #############################################
    if not args.notf:
        logging.info("Training TensorFlow network")
        if args.dfnetwork is None:
            dfnetwork = network
        else:
            dfnetwork = args.dfnetwork
        params = ["--input_dir",
                  "gs://<example_bucket>/minimalnetwork/runs/%s/tfrecords/*" % str(dfnetwork),
                  "--output_path",
                  "gs://<example_bucket>/minimalnetwork/runs/%s/results" % str(network)]
        if args.local:
            logging.info("Running with Local TF")
            subprocess.call(["python", "train.py"] + params)
        else:
            logging.info("Running with CloudML")
            subprocess.call(["gcloud", "beta", "ml", "jobs", "submit", "training", str(network),
                             #" --config config.yaml" +
                             "--region", "europe-west1",
                             "--module-name", "trainer.train",
                             "--staging-bucket", "gs://<staging-bucket>",
                             "--package-path", ".",
                             "--project", "<gcp-project>",
                             "--"] + params)
