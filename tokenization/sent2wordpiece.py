#!/usr/bin/env python3

import sys
import os
import re
import json


# SentencePiece boundary marker
SENTPIECE_BOUNDARY = '▁'    # (U+2581)

# WordPiece continuation marker
WORDPIECE_CONTINUATION = '##'

# SentencePiece special tokens, filtered out by default
SENTPIECE_SPECIAL = set(['<unk>', '<s>', '</s>'])

# BERT special tokens, added in by default
BERT_SPECIAL = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

# BERT "unused" special token format
UNUSED_FORMAT = '[unused{}]'

# SentencePiece line format regex
SENTPIECE_LINE_RE = re.compile(r'^(.*)\t(-?[0-9.]+)$')


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-c', '--add-chars', default=False, action='store_true',
                    help='add single character tokens')
    ap.add_argument('-n', '--no-special', default=False, action='store_true',
                    help='do not add special BERT tokens')
    ap.add_argument('-k', '--keep-special', default=False, action='store_true',
                    help='keep special SentencePiece tokens')
    ap.add_argument('-o', '--output', metavar='FILE', default=None,
                    help='output file (default STDOUT)')
    ap.add_argument('-u', '--unused', metavar='INT', default=100, type=int,
                    help='number of "[unused]" tokens to add')
    ap.add_argument('vocab', help='SentencePiece vocabulary')
    return ap


def load_vocab(path):
    vocab, seen = [], set()
    if path.endswith(".json"):
        data = json.load(open(path))["model"]["vocab"] # To read the new Tokenizers format
        for nr, piece in enumerate(data):
            if piece in seen:
                raise ValueError('duplicate {} on line {} in {}'.format(piece, nr, path))
            vocab.append(piece)
            seen.add(piece)
    else:
        with open(path) as f:
            for ln, l in enumerate(f, start=1):
                l = l.rstrip('\n')
                m = SENTPIECE_LINE_RE.match(l)
                if not m:
                    raise ValueError('line {} in {}: "{}"'.format(ln, path, l))
                piece, _ = m.groups()
                if piece in seen:
                    raise ValueError('duplicate {} on line {} in {}: "{}"'.format(
                        piece, ln, path, l))
                vocab.append(piece)
                seen.add(piece)
    print('read {} from {}'.format(len(vocab), path), file=sys.stderr)
    return vocab


def filter_vocab(vocab):
    filtered = [v for v in vocab if v not in SENTPIECE_SPECIAL]
    print('filtered from {} to {}'.format(len(vocab), len(filtered)),
          file=sys.stderr)
    return filtered


def convert_vocab(vocab):
    converted = []
    for v in vocab:
        if v.startswith(SENTPIECE_BOUNDARY):
            converted.append(v[len(SENTPIECE_BOUNDARY):])    # strip marker
        else:
            converted.append(WORDPIECE_CONTINUATION+v)    # add marker
    converted = [t for t in converted if t and not t.isspace()]
    return converted


def add_special(vocab, unused_count):
    unused = [UNUSED_FORMAT.format(i) for i in range(unused_count)]
    extended = [BERT_SPECIAL[0]] + unused + BERT_SPECIAL[1:] + vocab
    print('extended from {} to {}'.format(len(vocab), len(extended)),
          file=sys.stderr)
    return extended


def add_chars(vocab):
    chars = set()
    for v in vocab:
        if v.startswith(WORDPIECE_CONTINUATION):
            v = v[len(WORDPIECE_CONTINUATION):]
        for c in v:
            chars.add(c)
    extended = vocab[:]
    tokens = set(vocab)
    for c in chars:
        if c not in tokens:
            extended.append(c)
    print('added chars from {} to {}'.format(len(vocab), len(extended)),
          file=sys.stderr)
    return extended


def write_vocab(vocab, out):
    for v in vocab:
        print(v, file=out)


def output_vocab(vocab, path):
    if path is None:
        write_vocab(vocab, sys.stdout)
    else:
        with open(path, 'w') as f:
            write_vocab(vocab, f)
    print('output {} to {}'.format(
        len(vocab), path if path is not None else 'STDOUT'), file=sys.stderr)


def main(argv):
    args = argparser().parse_args(argv[1:])
    vocab = load_vocab(args.vocab)
    if not args.keep_special:
        vocab = filter_vocab(vocab)
    vocab = convert_vocab(vocab)
    if not args.no_special:
        vocab = add_special(vocab, args.unused)
    if args.add_chars:
        vocab = add_chars(vocab)
    output_vocab(vocab, args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
