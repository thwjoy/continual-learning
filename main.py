import os
import argparse
import sys
import network
sys.path.append('./torch_utils')


def main():
    parser = argparse.ArgumentParser(description="Arguments for running \
                            example networks in pytorch")
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train over (default: 50)')
    parser.add_argument('--test', default=False,
                        help='Set the phase of the network to test')
    parser.add_argument('--dataset_list', required=True,
                        help='Path to csv containing the dataset and labels')
    parser.add_argument('--log_message', required=False, default='',
                        help='A simple message to help debugging')
    args = parser.parse_args()


    if args.test:
        network.test(args)
    else:
        network.train(args)

if __name__ == '__main__':
    main()