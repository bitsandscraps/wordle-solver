import json

import numpy as np
from tqdm import tqdm

def main():
    matrix = np.load('./hint_matrix.npy')
    sizes = np.asarray([matrix.shape[0] // 12] * 12)
    sizes[:matrix.shape[0] % 12] += 1
    indices = [0] + np.cumsum(sizes).tolist()
    for part, (start, end) in enumerate(zip(indices, tqdm(indices[1:]))):
        with open(f'../hint_matrix_{part}.json', 'w', encoding='utf-8') as jfp:
            json.dump(matrix[start:end].tolist(), jfp)


if __name__ == '__main__':
    main()
