import os


def remove_line_from_file(filename, line_to_remove, dirpath=''):
    """Remove all occurences of `line_to_remove` from file
    with name `filename`, contained at path `dirpath`.
    If `dirpath` is omitted, relative paths are used."""
    filename = os.path.join(dirpath, filename)
    temp_path = os.path.join(dirpath, 'temp.txt')

    with open(filename, 'r') as f_read, open(temp_path, 'w') as temp:
        for line in f_read:
            if line.strip() == line_to_remove:
                continue
            temp.write(line)

    os.remove(filename)
    os.rename(temp_path, filename)


def main():
    """Driver function"""
    directory = input('direcory: ')
    word = input('word:')

    dirpath, _, files = next(os.walk(directory))
	
    
    for f in files:
        remove_line_from_file(f, word, dirpath)

if __name__ == '__main__':
    main()