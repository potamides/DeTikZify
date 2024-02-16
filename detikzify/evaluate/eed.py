from pygments.lexers.markup import TexLexer
from pygments.token import Comment, Text
from torchmetrics.text import ExtendedEditDistance
from torchmetrics.functional.text.eed import (
    _compute_sentence_statistics,
    _preprocess_en,
    _preprocess_ja,
)
from torchmetrics.functional.text.helper import _validate_inputs

class TexEditDistance(ExtendedEditDistance):
    """Adapt torchmetrics ExtendedEditDistance for TeX"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lexer = TexLexer()

    def __str__(self):
        return self.__class__.__name__

    def _preprocess_sentences(self, preds, target, language):
        target, preds = _validate_inputs(hypothesis_corpus=preds, ref_corpus=target)

        def tokenize(text):
            tokens = list()
            for tokentype, value in self.lexer.get_tokens(text):
                if value.strip():
                    if tokentype is Text:
                        if language == "en":
                            preprocess_function = _preprocess_en
                        elif language == "ja":
                            preprocess_function = _preprocess_ja
                        else:
                            raise ValueError(f"Expected argument `language` to either be `en` or `ja` but got {language}")
                        tokens.extend(preprocess_function(value).split())
                    elif not tokentype is Comment:
                        tokens.extend(value.split())

            return " " + " ".join(tokens) + " "

        preds = [tokenize(pred) for pred in preds]
        target = [[tokenize(ref) for ref in reference] for reference in target]

        return preds, target

    def update(self, preds, target):
        """Update state with predictions and targets."""
        preds, target = self._preprocess_sentences(preds, target, self.language)

        if self.sentence_eed is None:
            self.sentence_eed = []

        if 0 in (len(preds), len(target[0])):
            return self.sentence_eed

        for (hypothesis, target_words) in zip(preds, target):
            score = _compute_sentence_statistics(
                hypothesis,
                target_words,
                self.alpha,
                self.rho,
                self.deletion,
                self.insertion
            )
            self.sentence_eed.append(score)

        return self.sentence_eed

    def compute(self, *args, **kwargs):
        return super().compute(*args, **kwargs).item() # type: ignore
