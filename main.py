import os
import argparse
import sys
import network
import pandas as pd
from tensorboard_logger import configure

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
    parser.add_argument('--task_list', required=True,
                        help='Path to csv containing the dataset and labels')
    parser.add_argument('--log_message', required=False, default='',
                        help='A simple message to help debugging')
    parser.add_argument('--ckpt', required=False,
                        help='Patch model weights, only required for testing')
    args = parser.parse_args()

    #read the tasks from each list
    tasks = pd.read_csv(args.task_list, sep=",", usecols=range(1), header=None)

        # tensorboard
    run_name = "./runs/run-classifier_batch_" + str(args.batch_size) \
                    + "_epochs_" + str(args.epochs) + "_" + args.log_message

    configure(run_name)

    for i, t in tasks.iterrows():
        args.dataset_list = t[0]
        args.task = i
        network.train(args)
        network.test(args)


    #now train and test them sequentially

    # if args.test:
    #     network.test(args)
    # else:
    #     network.train(args)

if __name__ == '__main__':
    main()