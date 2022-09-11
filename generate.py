import argparse
from pathlib import Path

from ngrams import N_grams


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=Path, default='model.pickle', dest='path')
parser.add_argument('--prefix', type=str, dest='prefix')
parser.add_argument('--length', type=int, default=30, dest='length')
args = parser.parse_args()

model = N_grams.load(args.path)

if args.prefix is None:
    prompt = [None]
else:
    prompt = args.prefix.split()[-2:]
    print(args.prefix, end=' ')
for i in range(args.length):
    prompt = prompt[-1], model.predict(*prompt)
    print(prompt[-1], end=' ')
