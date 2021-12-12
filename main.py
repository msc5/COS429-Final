
import argparse

import train

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Train or Test a model')

trainer = subparsers.add_parser(
    'train',
    help='Train a model',
)
trainer.add_argument('model')
trainer.add_argument('dataset')

tester = subparsers.add_parser(
    'test',
    help='Test a model'
)
tester.add_argument('model')

if __name__ == '__main__':
    args = parser.parse_args()

    print(args.model)
    print(args.dataset)
