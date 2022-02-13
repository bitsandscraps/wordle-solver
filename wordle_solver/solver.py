from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, Iterable

import math
import numpy as np
from tqdm import tqdm

from assets import VALID_WORDS


ABSENT_CODE = 'A'
PRESENT_CODE = 'P'
CORRECT_CODE = 'C'
PROMPT = f'{ABSENT_CODE}: Absent, {PRESENT_CODE}: Present, {CORRECT_CODE}: Correct -> '
REPLY = 'Suggested Guess: {}'
ERROR_MESSAGE = f'The input should be one of {ABSENT_CODE},{PRESENT_CODE}, {CORRECT_CODE}'


class Hint:
    UNINITIALIZED = -1
    ABSENT = 0
    PRESENT = 1
    CORRECT = 2
    BASE = 3
    LOOKUP = {
        ABSENT: ABSENT_CODE,
        PRESENT: PRESENT_CODE,
        CORRECT: CORRECT_CODE,
    }

    def __init__(self, hints: Iterable[int]):
        self.hints = list(hints)

    def __getitem__(self, index) -> int:
        return self.hints[index]

    def __setitem__(self, index, hint):
        assert hint in frozenset((self.ABSENT, self.PRESENT, self.CORRECT))
        self.hints[index] = hint

    def correct(self) -> bool:
        return all(hint == self.CORRECT for hint in self.hints)

    def encode(self) -> int:
        assert self.UNINITIALIZED not in self.hints
        return int(''.join(map(str, self.hints)), self.BASE)

    def __repr__(self):
        return ' '.join(self.LOOKUP[hint] for hint in self.hints)

    @classmethod
    def new(cls, length):
        return cls((cls.UNINITIALIZED,) * length)

    @classmethod
    def decode(cls, code: int, length: int) -> 'Hint':
        hints = []
        for _ in range(length):
            hints.append(code % cls.BASE)
            code //= cls.BASE
        return cls(hints[::-1])


def build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', default='hint_matrix.npy')
    return parser


def compute_hint(guess: str, answer: str) -> Hint:
    appeared: Dict[str, int] = defaultdict(int)
    hint = Hint.new(len(guess))
    for index, (gletter, aletter) in enumerate(zip(guess, answer)):
        if gletter == aletter:
            hint[index] = Hint.CORRECT
        else:
            appeared[aletter] += 1
    for index, gletter in enumerate(guess):
        if hint[index] == Hint.CORRECT:
            continue
        if appeared[gletter] > 0:
            hint[index] = Hint.PRESENT
            appeared[gletter] -= 1
        else:
            hint[index] = Hint.ABSENT
    return hint


def suggest(hint_matrix: np.ndarray) -> int:
    minnegent = math.inf
    minindex = 0
    for index, row in enumerate(tqdm(hint_matrix, leave=False)):
        counts = np.unique(row, return_counts=True)[1]
        negent = sum(cnt * math.log(cnt) for cnt in counts)
        if negent < minnegent:
            minnegent = negent
            minindex = index
    return minindex



def main(file: str):
    hint_matrix = np.load(file)
    length = len(VALID_WORDS[0])
    allcorrect = Hint([Hint.CORRECT for _ in range(length)]).encode()
    guess_index = suggest(hint_matrix)
    while True:
        print(REPLY.format(VALID_WORDS[guess_index]))
        while True:
            hints = Hint.new(length)
            hints_str = input(PROMPT)
            if len(hints_str) > length:
                print(f'Expected {length} hints. Got {len(hints_str)}')
                continue
            for index, hint_str in enumerate(hints_str):
                if hint_str == ABSENT_CODE:
                    hints[index] = Hint.ABSENT
                elif hint_str == PRESENT_CODE:
                    hints[index] = Hint.PRESENT
                elif hint_str == CORRECT_CODE:
                    hints[index] = Hint.CORRECT
                else:
                    print(ERROR_MESSAGE)
                    break
            else:
                break
        if hints_str == CORRECT_CODE * length:
            return
        mask = hint_matrix[guess_index] == hints.encode()
        hint_matrix = hint_matrix[:, mask]
        if hint_matrix.shape[1] == 1:
            guess_index = np.flatnonzero(hint_matrix == allcorrect)[0]
        else:
            guess_index = suggest(hint_matrix)


if __name__ == '__main__':
    main(**vars(build_argument_parser().parse_args()))
