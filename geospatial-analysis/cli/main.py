import os
import sys

import click

ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(ABS_PATH)
import webapp.segment_tiff
from train import start_training


@click.group()
@click.option('--verbose', is_flag=True, help='Enables verbose mode')
@click.pass_context
def cli1(ctx, verbose):
    ctx.obj = {'VERBOSE': verbose}


@cli1.command()
@click.option('--dataset-dir', help='Batch size to use for training')
@click.pass_context
def train(ctx, dataset_dir):
    click.echo('Training model')
    start_training(dataset_dir=dataset_dir)


@click.group()
@click.option('--verbose', is_flag=True, help='Enables verbose mode')
@click.pass_context
def cli2(ctx, verbose):
    ctx.obj = {'VERBOSE': verbose}


@cli2.command()
@click.option('--path-to-image', help='Path to image to classify')
@click.option('--query', help='Query to use for classification')
@click.pass_context
def tiff_predict(ctx, path_to_image, query):
    click.echo('Classifying image')
    webapp.segment_tiff.load_model()
    webapp.segment_tiff.predict(sam_type="", box_threshold=0.31, text_threshold=0.31, tiff_path=path_to_image,
                                text_prompt=query, file_url="", center_coords="")


cli = click.CommandCollection(sources=[cli1, cli2])

if __name__ == '__main__':
    # the purpose of this file is to be the entry point for our cli application
    # we will be using the click library to create a command line interface
    # the application will manage the training, validation and testing of the models
    # the application will also manage the deployment of the models

    cli(obj={})
