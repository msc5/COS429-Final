import os


def section(msg):
    n, _ = os.get_terminal_size()
    l = len(msg)
    print('')
    print(msg + ' ' + '-'*(n - 1 - l))
    print('')


def tabulate(dict):
    for key, val in dict.items():
        print(f'{key:>40} : {val}')
