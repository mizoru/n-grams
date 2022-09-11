import argparse
from ngrams import N_grams
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, default='./data', dest='input') #type=pathlib.Path
parser.add_argument('--model', type=Path, default = 'model.pickle', dest='output')

args = parser.parse_args()

model = N_grams()
model.fit(args.input)
model.save(args.output)