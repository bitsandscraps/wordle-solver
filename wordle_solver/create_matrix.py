from argparse import ArgumentParser
from multiprocessing import Pool
from typing import List

import numpy as np

from assets import VALID_WORDS, WORDLE_CANDIDATES
from solver import compute_letter_mask, str2array


def build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('-e', '--entire', action='store_true')
    parser.add_argument('-o', '--output', default='result')
    parser.add_argument('-q', '--quota', default=200)
    return parser


def compute_candidates(array: np.ndarray,
                       masks: np.ndarray,
                       guess: np.ndarray,
                       answer: np.ndarray) -> np.ndarray:
    masks_ = []
    correct = answer == guess
    if np.all(correct):
        return np.zeros(array.shape[0], dtype=np.bool8)
    masks_.append(np.all(array[:, correct] == guess[correct], axis=1))
    wrong = answer != guess
    masks_.append(np.all(array[:, wrong] != guess[wrong], axis=1))
    present = np.intersect1d(guess, answer)
    masks_.append(np.all(masks[present], axis=0))
    absent = np.setdiff1d(guess, answer)
    masks_.append(np.logical_not(np.any(masks[absent], axis=0)))
    return np.all(masks_, axis=0)


def count_candidates(rank: int, start: int, end: int, entire: bool, output: str):
    guess_array = str2array(VALID_WORDS)
    guess_masks = compute_letter_mask(guess_array)
    if entire:
        answer_array = guess_array
        answer_masks = guess_masks
    else:
        answer_array = str2array(WORDLE_CANDIDATES)
        answer_masks = compute_letter_mask(answer_array)
    length = answer_array.shape[0]
    result = np.empty((end - start, length, length), dtype=np.bool8)
    for index0, guess_index in enumerate(range(start, end)):
        guess = guess_array[guess_index]
        for answer_index, answer in enumerate(answer_array):
            result[index0, answer_index, :] = compute_candidates(
                answer_array, answer_masks, guess, answer)
    np.save(output + str(rank), result)


def main(entire: bool, output: str, quota: int):
    indices = list(range(0, len(VALID_WORDS), quota)) + [len(VALID_WORDS)]
    arguments = [(rank, start, end, entire, output)
                 for rank, (start, end)
                 in enumerate(zip(indices[:-1], indices[1:]))]
    with Pool() as pool:
        pool.starmap_async(count_candidates, arguments).get()


if __name__ == '__main__':
    main(**vars(build_argument_parser().parse_args()))
