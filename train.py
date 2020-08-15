from argparse import ArgumentParser

import deepspeech_pytorch.training as training


def main(args):
    training.train(args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = training.add_trainer_args(parser)
    main(parser.parse_args())
