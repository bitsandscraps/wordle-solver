from os.path import abspath, dirname, join


def main():
    root = dirname(abspath(__file__))
    valid_words = []
    path = join(root, 'sorted_word_list.txt')
    with open(path, 'r', encoding='utf-8') as word_list:
        for word in word_list:
            valid_words.append(word.strip())
    path = join(root, 'sorted_word_list_python.txt')
    with open(path, 'w', encoding='utf-8') as word_list_python:
        indices = list(range(0, len(valid_words), 8)) + [len(valid_words)]
        for start, end in zip(indices[:-1], indices[1:]):
            print("'" + "', '".join(valid_words[start:end]) + "',",
                  file=word_list_python)


if __name__ == '__main__':
    main()
