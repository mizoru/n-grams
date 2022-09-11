import argparse
from ngrams import N_grams
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=Path, default='model.pickle', dest='path') #type=pathlib.Path
parser.add_argument('--prefix', type=str, dest='prefix')
parser.add_argument('--length', type=int, default=30, dest='length')
args = parser.parse_args()

model = N_grams.load(args.path)

f, s = args.prefix.split()[-2:]
print(f,s,end=' ')
for i in range(args.length):
    f, s = s, model.predict(f, s)
    print(s, end=' ')
