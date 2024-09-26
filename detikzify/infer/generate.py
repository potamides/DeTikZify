from collections import deque
from dataclasses import dataclass
from functools import cached_property
from math import sqrt
from multiprocessing.pool import ThreadPool
from re import sub
from time import time
from types import SimpleNamespace as Namespace
from typing import Any, Dict, Generator, List, Literal, Optional, Set, Tuple, Union

from PIL import Image
import torch
from torchmetrics import Metric
from transformers import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

from ..evaluate.imagesim import ImageSim
from ..mcts.montecarlo import MonteCarlo
from ..mcts.node import Node
from ..util import ExplicitAbort, StreamerList, TokenStreamer, cast_cache, expand, load
from .tikz import TikzDocument

Numeric = Union[int, float]

@dataclass(frozen=True)
class NodeState:
    token_ids: torch.Tensor
    num_lines: int = 0

    def __eq__(self, other: Any) -> bool:
        try:
            return self.token_ids.equal(other.token_ids)
        except (AttributeError, TypeError):
            return False

    def __hash__(self):
        return hash(tuple(self.token_ids.tolist()))


class WideNode(Node):
    state: NodeState

    def __init__(self, *args, exploration=0.6, is_widen_node=False, **kwargs):
        super().__init__(NodeState(*args, **kwargs))
        self.discovery_factor = exploration
        self.is_widen_node = is_widen_node
        self.update_policy_value(1.0)
        if not is_widen_node:
            self.add_child(WideNode(
                *args,
                exploration=exploration,
                is_widen_node=not is_widen_node,
                **kwargs
            ))

    def add_child(self, child):
        self.expanded = self.expanded or not child.is_widen_node
        super().add_child(child)

    @property
    def depth(self) -> int:
        depth, current = 0, self
        while parent:=current.parent:
            depth, current = depth + 1, parent
        return depth

    @property
    def token_ids(self):
        return self.state.token_ids

    @property
    def num_lines(self):
        return self.state.num_lines


class DynMinMaxNorm:
    def __init__(self, default_value: Numeric = 0):
        self.scores = set()
        self.default_value = default_value

    def normalize(self, score: Numeric) -> "MinMaxScore":
        self.scores.add(score)
        return self.MinMaxScore(score, all_scores=self.scores, default_value=self.default_value)

    def __call__(self, *args, **kwargs) -> "MinMaxScore":
        return self.normalize(*args, **kwargs)

    class MinMaxScore:
        def __init__(
            self,
            *scores: Numeric,
            all_scores: Set[Numeric],
            default_value: Numeric,
            no_minmax_scores: List[Numeric] = list(),
        ):
            self.scores = list(scores)
            self.all_scores = all_scores
            self.default_value = default_value
            self.no_minmax_scores = no_minmax_scores.copy()

        @property
        def score(self) -> Numeric:
            min_score, max_score = min(self.all_scores), max(self.all_scores)
            try:
                score = sum((score - min_score) / (max_score - min_score) for score in self.scores)
            except ZeroDivisionError:
                score = self.default_value
            return score + sum(self.no_minmax_scores)

        def __add__(self, other: Any) -> "DynMinMaxNorm.MinMaxScore":
            new = self.__class__(
                *self.scores,
                all_scores=self.all_scores,
                default_value=self.default_value,
                no_minmax_scores=self.no_minmax_scores
            )
            try:
                new.scores.extend(other.scores)
                new.no_minmax_scores.extend(other.no_minmax_scores)
            except AttributeError:
                new.no_minmax_scores.append(other)
            return new

        def __mul__(self, other: Any) -> "DynMinMaxNorm.MinMaxScore":
            return self.score * other

        def __truediv__(self, other: Any) -> "DynMinMaxNorm.MinMaxScore":
            return self.score / other

        def __rtruediv__(self, other: Any) -> "DynMinMaxNorm.MinMaxScore":
            return other / self.score

        __radd__, __rmul__ = __add__, __mul__


class DetikzifyGenerator:
    def __init__(
        self,
        model,
        processor,
        image: Image.Image,
        metric: Optional[Metric] = None,
        compile_timeout: Optional[int] = 60,
        mcts_timeout: Optional[int] = None,
        streamer: Optional[BaseStreamer] = None,
        control: Optional[ExplicitAbort] = None,
        exploration: float = 0.6, # exploration coefficient
        strict: bool = False, # if True, treat recoverable errors same as fatal errors when computing scores
        **gen_kwargs,
    ):
        self.model = model
        self.processor = processor
        self.metric = metric
        self.image = image
        self.compile_timeout = compile_timeout
        self.mcts_timeout = mcts_timeout
        self.streamer = streamer
        self.exploration = exploration
        self.strict = strict
        self.gen_kwargs = gen_kwargs

        self.solution = deque(maxlen=1)
        self.failed_rollouts = dict()
        self.norm = DynMinMaxNorm()
        self.control = control or ExplicitAbort()
        self.montecarlo = MonteCarlo(
            root_node=WideNode(
                processor(
                    images=self.image,
                    return_tensors="pt",
                ).input_ids.to(model.device).squeeze(),
                exploration=self.exploration
            )
        )

        self.montecarlo.child_finder = self.child_finder # type: ignore

    def __call__(self, *args, **kwargs):
        return self.simulate(*args, **kwargs)

    def simulate(self, expansions: Optional[Numeric] = 1) -> Generator[Tuple[Numeric, TikzDocument], None, None]:
        """
        Run the simulations. Returns all rollouts (successful or unsuccessful)
        in descending order (best rollouts first) of their score.
        """
        start_time = time()
        while expansions is None or (expansions:=expansions-1) >= 0:
            self.montecarlo.simulate()
            yield self.solution.pop()
            if self.mcts_timeout is not None and time() - start_time > self.mcts_timeout:
                return

    def generate(self, input_ids: torch.Tensor, streamer: Optional[BaseStreamer] = None, **gen_kwargs) -> torch.Tensor:
        streamers, numel = StreamerList(filter(bool, [streamer, self.streamer])), input_ids.numel()
        max_length = {**self.model.generation_config.to_dict(), **self.gen_kwargs, **gen_kwargs}["max_length"]
        if (numel and input_ids[-1] == self.processor.tokenizer.eos_token_id) or numel >= max_length:
            streamers.end()
            return input_ids # prevent continuing generation after eos
        with torch.inference_mode():
            return self.model.generate(
                input_ids=input_ids.unsqueeze(0),
                bad_words_ids=[[self.model.config.image_token_id]],
                pixel_values=self.processor(images=self.image, return_tensors="pt").pixel_values.to(self.model.device),
                streamer=streamers,
                **self.gen_kwargs,
                **gen_kwargs
            ).squeeze()

    @cached_property
    def newlineinfo(self):
        # tokens can potentially contain multiple newlines, so we need special
        # handling when we want to map error lines to tokens
        newlineinfo = dict()
        for token_id in self.processor.tokenizer.vocab.values():
            # NOTE: Newline normalization might lead to inaccurate estimations
            # for windows line separators (if split over two tokens). However,
            # the likeliness of such tokens being generated (not in training
            # data) as well as the potential impact is negligible.
            # https://www.overleaf.com/learn/latex/Articles/An_introduction_to_%5Cendlinechar%3A_How_TeX_reads_lines_from_text_files
            token = sub(r"\r\n|\r", r"\n", self.processor.decode([token_id]))
            if (num_lines:=token.count("\n")):
                newlineinfo[token_id] = Namespace(num_lines=num_lines, trailing=token.endswith("\n"))
        assert newlineinfo
        return newlineinfo

    def rollout(self, state: NodeState) -> Generator[Tuple[torch.Tensor, int], None, None]:
        input_ids, num_lines, continuation = state.token_ids, state.num_lines, False
        with ThreadPool(processes=1) as thread:
            streamer = TokenStreamer()
            async_result = thread.apply_async(
                func=self.generate,
                error_callback=streamer.propagate_error,
                args=[input_ids],
                kwds=dict(
                    stopping_criteria=StoppingCriteriaList([self.control.reset()]),
                    streamer=streamer,
                )
            )

            try:
                prev_ids, line = input_ids, list()
                for token in streamer:
                    line.append(token)
                    if info:=self.newlineinfo.get(token):
                        # continuations (newline followed by text in a single
                        # token) don't appear in the llama 3.1 tokenizer, but
                        # handling them now makes this code future proof
                        num_lines += info.num_lines - continuation
                        continuation = not info.trailing
                        prev_ids = torch.cat((prev_ids, torch.tensor(line, device=prev_ids.device)))
                        line.clear()
                        yield prev_ids, num_lines
                if line:
                    yield torch.cat((prev_ids, torch.tensor(line, device=prev_ids.device))), num_lines - continuation
            except (GeneratorExit, KeyboardInterrupt):
                self.control.abort()
                raise
            else:
                if self.control.should_stop:
                    raise InterruptedError
            finally:
                async_result.wait()

    @cast_cache(lambda token_ids: tuple(token_ids.tolist()))
    def decode(self, token_ids: torch.Tensor) -> TikzDocument:
        return TikzDocument(
            timeout=self.compile_timeout,
            code=self.processor.decode(
                token_ids=token_ids,
                skip_special_tokens=True
            )
        )

    @cast_cache(lambda image: image.tobytes())
    def score(self, image: Image.Image) -> Numeric:
        assert self.metric
        self.metric.update(image, self.image)
        score = self.metric.compute()
        self.metric.reset()
        return score

    def sample(self):
        return self.decode(self.generate(
            input_ids=self.montecarlo.root_node.token_ids,
        ))

    def child_finder(self, node: WideNode, montecarlo: MonteCarlo):
        new_nodes = list()
        for new_state in (rollout:=self.rollout(node.state)):
            new_node = WideNode(*new_state, exploration=self.exploration)
            if new_node.state in self.failed_rollouts:
                new_nodes.extend(self.failed_rollouts[new_node.state])
                rollout.close()
                break
            new_nodes.append(new_node)

        if node.is_widen_node:
            node.visits += 1
            node, new_nodes = self.merge(node.parent, new_nodes) # type: ignore

        tikz = self.decode((new_nodes or [node])[-1].token_ids)
        skip_idx = round(sqrt(len(new_nodes)))

        if scorable:=(tikz.is_rasterizable and not (self.strict and tikz.compiled_with_errors)):
            for new_node in new_nodes[:skip_idx]:
                node.add_child(node:=new_node)
        # Only process failed rollouts when we can locate the error (errorln !=
        # 0). In rare cases there is no error information even though the
        # tikzpic is not rasterizable because only cropping failed -> use [0].
        elif errorln:=min(tikz.errors or [0]):
            for idx, new_node in enumerate(new_nodes):
                ends_with_eol = self.newlineinfo.get(new_node.token_ids[-1])
                if new_node.num_lines < errorln and idx < skip_idx:
                    node.add_child(node:=new_node)
                elif new_node.num_lines > errorln or (new_node.num_lines == errorln and ends_with_eol):
                    self.failed_rollouts[new_node.state] = new_nodes[idx:]
                    break

        if self.metric:
            score = self.score(tikz.rasterize()) if scorable else -1 # type: ignore
        else: # if we do not have a metric, use compiler logs instead
            score = scorable - tikz.compiled_with_errors

        node.update_win_value(self.norm(score) if scorable and self.metric else score)
        self.solution.append((score, tikz))

    def merge(self, node: WideNode, nodes_to_merge: List[WideNode]) -> Tuple[WideNode, List[WideNode]]:
        for merge_node in nodes_to_merge:
            for child in node.children:
                if child.state == merge_node.state:
                    node, nodes_to_merge = child, nodes_to_merge[1:]
                    break
            else:
                break
        return node, nodes_to_merge


class DetikzifyPipeline:
    def __init__(
        self,
        model,
        processor,
        # hyperparams based on "a systematic evaluation of large language models of code"
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 0,
        compile_timeout: Optional[int] = 60, # same as old overleaf compile timeout
        metric: Union[Literal["model", "fast"], Metric]  = "model",
        **gen_kwargs,
    ):
        self.model = model
        self.processor = processor

        if metric == "model": # SelfSim
            self.metric = ImageSim.from_detikzify(model, processor, sync_on_compute=False)
        elif metric == "fast": # Compiler Diagnostics
            self.metric = None
        else:
            self.metric = metric

        self.gen_kwargs: Dict[str, Any] = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_length=processor.tokenizer.model_max_length,
            do_sample=True,
            compile_timeout=compile_timeout,
            **gen_kwargs
        )

    def load(self, image: Union[Image.Image, str], preprocess: bool = True):
        image = load(image)
        if preprocess:
            return expand(image, max(image.size), do_trim=True)
        return image

    def sample(self, image: Union[Image.Image, str], preprocess: bool = True, **gen_kwargs) -> TikzDocument:
        """
        DeTikZify a raster image. Samples a single image and returns it.
            image: the image
            preprocess: whether to preprocess the image (expand to square and
                trim to content)
            gen_kwargs: additional generation kwargs (potentially overriding
                the default ones)
        """
        generator = DetikzifyGenerator(
            model=self.model,
            processor=self.processor,
            image=self.load(image, preprocess=preprocess),
            **self.gen_kwargs,
            **gen_kwargs
        )

        return generator.sample()

    def simulate(
        self,
        image: Union[Image.Image, str],
        preprocess: bool = True,
        expansions: Optional[Numeric] = None,
        timeout: Optional[int] = None,
        **gen_kwargs,
    ) -> Generator[Tuple[Numeric, TikzDocument], None, None]:
        """
        DeTikZify a raster image using MCTS. Returns an iterator yielding
        (score, tikzdoc) tuples of TikZ documents created during rollouts.
            image: the image
            preprocess: whether to preprocess the image (expand to square and
                trim to content)
            expansions: number of attempted MCTS expansions (set to None, 0 or
                math.inf for infinite)
            timeout: timeout for MCTS in seconds (set to 0, math.inf, or
                None for infinite)
            gen_kwargs: additional generation kwargs (potentially overriding
                the default ones)
        """
        generator = DetikzifyGenerator(
            model=self.model,
            processor=self.processor,
            metric=self.metric,
            mcts_timeout=timeout or None,
            image=self.load(image, preprocess=preprocess),
            **self.gen_kwargs,
            **gen_kwargs
        )

        yield from generator.simulate(expansions or None)

    def __call__(self, *args, **kwargs) -> TikzDocument:
        return self.sample(*args, **kwargs)
