import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='EPiC')

    # Genreal
    parser.add_argument('--emotion',        default='valence',  type=str,               help='emotion label')
    parser.add_argument('--scenario',       default=1,          type=int,               help='scenario number')
    parser.add_argument('--fold',           default=0,          type=int,               help='fold number')

    parser.add_argument('--modality',       default='ecg',      type=str,               help='bio-sginals')
    parser.add_argument('--lr',             default=0.0001,     type=float,             help='initial learning rate')
    parser.add_argument('--epochs',         default=50,         type=int,               help='epochs')
    parser.add_argument('--batch-size',     default=8,          type=int,               help='batch size')
    parser.add_argument('--optimizer',      default='sgd',      type=str,               help='optimizer choice')
    parser.add_argument('--pretraining',    default=False,      action="store_true",    help='pretraining')
    parser.add_argument('--use-pretrain',   default=False,      action="store_true",    help='use pretrained model')
    parser.add_argument('--final-flag',     default=False,      action="store_true",    help='obtain final test results')

    parser.add_argument('--use-scheduler',  default=False,      action="store_true",    help='learning rate scheduler')

    return parser








