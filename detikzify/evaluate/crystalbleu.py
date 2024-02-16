from collections import Counter
from functools import cached_property
from hashlib import md5
from itertools import chain, tee
from pickle import dump, load
from typing import List

from crystalbleu import corpus_bleu
from datasets.config import HF_METRICS_CACHE
from datasets.utils.logging import get_logger
from pygments.lexers.markup import TexLexer
from pygments.token import Comment, Name, Text
from sacremoses import MosesTokenizer
from torchmetrics import Metric

logger = get_logger("datasets")

# adopted from nltk
def pad_sequence(sequence, n, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence

# adopted from nltk
def ngrams(sequence, n, **kwargs):
    sequence = pad_sequence(sequence, n, **kwargs)
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.

class CrystalBLEU(Metric):
    """Wrapper around https://github.com/sola-st/crystalbleu (adapted for LaTeX)"""

    def __init__(self, corpus, k=500, n=4, use_cache=True, **kwargs):
        super().__init__(**kwargs)
        self.lexer = TexLexer()
        self.tokenizer = MosesTokenizer()
        self.use_cache = use_cache
        self.corpus = corpus
        self.k = k
        self.n = n

        self.add_state("list_of_references", [], dist_reduce_fx="cat")
        self.add_state("hypotheses", [], dist_reduce_fx="cat")

    def __str__(self):
        return self.__class__.__name__

    @cached_property
    def trivially_shared_ngrams(self):
        """
        Computes trivially shared ngrams and caches them.
        """
        cache_dir = HF_METRICS_CACHE / self.__class__.__name__.lower()
        cache_dir.mkdir(parents=True, exist_ok=True)

        dhash = md5()
        dhash.update(str(sorted(self.corpus)).encode())
        hashname = f"{dhash.hexdigest()}.pkl"

        if (cache_file:=(cache_dir / hashname)).is_file() and self.use_cache:
            logger.info(f"Found cached trivially shared ngrams ({cache_file})")
            with open(cache_file, "rb") as f:
                return load(f)
        else:
            all_ngrams = list()
            for o in range(1, self.n+1):
                for tex in self.corpus:
                    all_ngrams.extend(ngrams(self._tokenize(tex), o))
            frequencies = Counter(all_ngrams)

            trivially_shared_ngrams = dict(frequencies.most_common(self.k))
            if self.use_cache:
                logger.info(f"Caching trivially shared ngrams ({cache_file})")
                with open(cache_file, "wb") as f:
                    dump(trivially_shared_ngrams, f)
            return trivially_shared_ngrams

    def _tokenize(self, text):
        tokens = list()
        for tokentype, value in self.lexer.get_tokens(text):
            if value.strip() and not tokentype is Comment:
                if any(tokentype is tp for tp in [Text, Name.Attribute, Name.Builtin]):
                    tokens.extend(self.tokenizer.tokenize(value.strip()))
                else:
                    tokens.append(value.strip())
        return tokens

    def update(
        self,
        list_of_references: List[List[str]],
        hypotheses: List[str],
    ):
        assert len(list_of_references) == len(hypotheses)
        self.list_of_references.extend([self._tokenize(ref) for ref in refs] for refs in list_of_references)
        self.hypotheses.extend(self._tokenize(hyp) for hyp in hypotheses)

    def compute(self):
        return corpus_bleu(
            list_of_references=self.list_of_references,
            hypotheses=self.hypotheses,
            ignoring=self.trivially_shared_ngrams
        )
