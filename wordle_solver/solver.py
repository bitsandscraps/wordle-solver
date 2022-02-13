from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, NamedTuple, Set, Tuple

import numpy as np

from assets import VALID_WORDS, WORDLE_CANDIDATES


class HINTS(NamedTuple):
    absent: Set[int]
    present: Set[int]
    correct: Dict[int, int]
    wrong: Tuple[Set[int], ...]

    @classmethod
    def initialize(cls, length):
        wrong = tuple(set() for _ in range(length))
        return cls(present=set(), correct={}, wrong=wrong)

    def is_empty(self):
        if any((self.present, self.correct)):
            return False
        return all(not wrong for wrong in self.wrong)


ABSENT = 0
PRESENT = 1
CORRECT = 2
PROMPT = f'{ABSENT}: Absent, {PRESENT}: Present, {CORRECT}: Correct -> '
REPLY = 'Suggested Guess: {}'
ERROR_MESSAGE = f'The input should be one of {ABSENT},{PRESENT}, {CORRECT}'


def build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('-c', '--coefficient', default=2, type=float)
    parser.add_argument('-i', '--initial', nargs='?', const='', default='lares')
    parser.add_argument('-r', '--restrict', action='store_true')
    return parser


def compute_letter_mask(array: np.ndarray) -> np.ndarray:
    masks = np.empty((26, array.shape[0]), dtype=bool)
    for letter in range(26):
        masks[letter] = np.any(array == letter, axis=1)
    return masks


def filter_candidates(array: np.ndarray,
                      hints: HINTS) -> np.ndarray:
    mask = np.ones(array.shape[0], dtype=bool)
    present_mask = np.all(letter_mask[list(hints.present)], axis=0)
    mask[np.logical_not(present_mask)] = False
    correct_indices = list(hints.correct.keys())
    correct_letters = list(hints.correct.values())
    mask[np.any(array[:, correct_indices] != correct_letters, axis=1)] = False
    for index, wrong_letters in enumerate(hints.wrong):
        indexed_array = array[:, index]
        for letter in wrong_letters:
            mask[indexed_array == letter] = False 
    return mask


def str2array(strlist: List[str]) -> np.ndarray:
    array = np.empty((len(strlist), len(strlist[0])), dtype=int)
    for index0, word in enumerate(strlist):
        for index1, letter in enumerate(word):
            array[index0, index1] = ord(letter) - ord('a')
    return array


def suggest_heuristic(hints: HINTS,
                      restrict: bool,
                      initial: str) -> str:
    if initial and hints.is_empty():
        return initial
    array = str2array(WORDLE_CANDIDATES if restrict else VALID_WORDS)
    letter_mask = compute_letter_mask(array)
    candidates = array[filter_candidates(array, letter_mask, hints)]


def main(suggest: Callable[..., str] = suggest_heuristic, **kwargs):
    length = len(VALID_WORDS[0])
    hints = HINTS.initialize(length)
    guess = suggest(hints=hints, **kwargs)
    while True:
        print(REPLY.format(guess))
        while True:
            hints_str = input(PROMPT)
            if len(hints) > length:
                print(f'Expected {length} hints. Got {len(hints_str)}')
                continue
            for index, (hint_str, guess_str) in enumerate(zip(hints_str, guess)):
                hint = int(hint_str)
                guess_code = ord(guess_str) - ord('a')
                if hint == ABSENT:
                    for index_, wrong in enumerate(hints.wrong):
                        if hints.correct[index_] != guess_code:
                            wrong.add(guess_code)
                elif hint == PRESENT:
                    hints.present.add(guess_code)
                    hints.wrong[index].add(guess_code)
                elif hint == CORRECT:
                    hints.correct[index] = guess_code
                    hints.wrong[index].discard(guess_code)
                else:
                    print(ERROR_MESSAGE)
                    break
            else:
                break
        if hints_str == str(CORRECT) * length:
            return
        guess = suggest(hints=hints, **kwargs)


if __name__ == '__main__':
    main(**vars(build_argument_parser().parse_args()))
