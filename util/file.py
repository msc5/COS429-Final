import os


def mkdir(*paths):

    for p in paths:
        if not os.path.exists(p):
            os.mkdir(p)
