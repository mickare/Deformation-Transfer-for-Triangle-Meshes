from functools import reduce


def tween(seq, sep):
    """From: https://stackoverflow.com/questions/5920643/add-an-item-between-each-item-already-in-the-list"""
    return reduce(lambda r, v: r + [sep, v], seq[1:], seq[:1])
