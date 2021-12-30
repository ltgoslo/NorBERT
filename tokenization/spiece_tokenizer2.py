#! /bin/env python3

import sys
import itertools
from gensim import utils
import sentencepiece as spm


class LineFile:
    def __init__(self, source, max_sentence_length=100000, limit=None):
        """Iterate over a file that contains sentences: one line = one sentence.
        Parameters
        ----------
        source : string or a file-like object
            Path to the file on disk, or an already-open file object (must support `seek(0)`).
        limit : int or None
            Clip the file to the first `limit` lines. Do no clipping if `limit is None` (the default).
        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in itertools.islice(self.source, self.limit):
                line = utils.to_unicode(line)[: self.max_sentence_length]
                if line.strip():
                    yield line
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.open(self.source, "rb") as fin:
                for line in itertools.islice(fin, self.limit):
                    line = utils.to_unicode(line)[: self.max_sentence_length]
                    if line.strip():
                        yield line


data = iter(LineFile(sys.argv[1]))

spm.SentencePieceTrainer.train(
    sentence_iterator=data,
    model_prefix=sys.argv[2],
    vocab_size=int(sys.argv[3]),
    model_type="bpe",
    character_coverage=0.99,
    split_by_number=False,
    num_threads=16
)
