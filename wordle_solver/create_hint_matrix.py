from argparse import ArgumentParser
from itertools import count
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from assets import VALID_WORDS
from solver_entropy import compute_hint


def build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', default='hint_matrix.npy')
    parser.add_argument('-q', '--quota', default=1024)
    return parser


def create_hint_matrix(rank: int, start: int, end: int) -> np.ndarray:
    result = []
    for guess in tqdm(VALID_WORDS[start:end],
                      desc=f'Rank {rank:02d}', leave=False):
        gresult = []
        for answer in VALID_WORDS:
            gresult.append(compute_hint(guess, answer).encode())
        result.append(gresult)
    return np.asarray(result)


def main(output: str, quota: int):
    indices = list(range(0, len(VALID_WORDS), quota)) + [len(VALID_WORDS)]
    rankstartend = zip(count(), indices, indices[1:])
    with Pool() as pool:
        hint_matrices = pool.starmap(create_hint_matrix, rankstartend)
    np.save(output, np.concatenate(hint_matrices, axis=0))


if __name__ == '__main__':
    main(**vars(build_argument_parser().parse_args()))
