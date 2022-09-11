import argparse
from pathlib import Path

from ngrams import N_grams


parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, dest='input')
parser.add_argument('--model', type=Path,
                    default='model.pickle', dest='output')
args = parser.parse_args()

model = N_grams()
model.fit(args.input)
model.save(args.output)
