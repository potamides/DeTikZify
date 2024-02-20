from collections import deque
from dataclasses import dataclass
from math import sqrt
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from time import time

from PIL import Image
import torch
from transformers import StoppingCriteriaList

from ..evaluate.patchsim import PatchSim
from ..mcts.montecarlo import MonteCarlo
from ..mcts.node import Node
from ..util import (
    BaseStreamer,
    ExplicitAbort,
    StreamerList,
    TokenStreamer,
    cast_cache,
    expand,
    load,
)
from .tikz import TikzDocument

Numeric = Union[int, float]

@dataclass(frozen=True)
class NodeState:
    token_ids: torch.Tensor

    def __eq__(self, other: Any) -> bool:
        try:
            return self.token_ids.equal(other.token_ids)
        except (AttributeError, TypeError):
            return False

    def __hash__(self):
        return hash(tuple(self.token_ids.tolist()))


class WideNode(Node):
    state: NodeState

    def __init__(self, *args, **kwargs):
        super().__init__(NodeState(*args, **kwargs))
        self.discovery_factor = 0.9
        self.add_child(widen:=Node(self.state))
        widen.is_widen_node = True
        #widen.update_policy_value(0.2)

    @property
    def depth(self) -> int:
        depth, current = 0, self
        while parent:=current.parent:
            depth, current = depth + 1, parent
        return depth

    @property
    def token_ids(self):
        return self.state.token_ids


class DynMinMaxNorm:
    def __init__(self, default_value: Numeric = 0):
        self.scores = []
        self.default_value = default_value

    def normalize(self, score: Numeric) -> "MinMaxScore":
        self.scores.append(score)
        return self.MinMaxScore(score, self.scores, self.default_value)

    def __call__(self, *args, **kwargs) -> "MinMaxScore":
        return self.normalize(*args, **kwargs)

    class MinMaxScore:
        def __init__(self, score: Numeric, all_scores: List[Numeric], default_value: Numeric):
            self.scores = [score]
            self.all_scores = all_scores
            self.no_minmax_scores = list()
            self.default_value = default_value

        @property
        def score(self) -> Numeric:
            min_score, max_score = min(self.all_scores), max(self.all_scores)
            try:
                score = sum((score - min_score) / (max_score - min_score) for score in self.scores)
            except ZeroDivisionError:
                score = self.default_value
            return score + sum(self.no_minmax_scores)

        def __add__(self, other: Any) -> "DynMinMaxNorm.MinMaxScore":
            try:
                self.scores.extend(other.scores)
            except AttributeError:
                self.no_minmax_scores.append(other)
            return self

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
        tokenizer,
        image: Image.Image,
        fast_metric: bool = False,
        compile_timeout: Optional[int] = 60,
        mcts_timeout: Optional[int] = None,
        streamer: Optional[BaseStreamer] = None,
        **gen_kwargs,
    ):
        self.newline_id = tokenizer.text("\n", add_special_tokens=False)["input_ids"][-1]
        assert tokenizer.text.decode(self.newline_id) == "\n"

        self.model = model
        self.tokenizer = tokenizer
        self.image = image
        self.fast_metric = fast_metric
        self.compile_timeout = compile_timeout
        self.mcts_timeout = mcts_timeout
        self.streamer = streamer
        self.gen_kwargs = gen_kwargs

        self.solution = deque(maxlen=1)
        self.failed_rollouts = dict()
        self.metric = PatchSim()
        self.norm = DynMinMaxNorm()
        self.thread = ThreadPool(processes=1)
        self.montecarlo = MonteCarlo(
            root_node=WideNode(
                tokenizer.text(
                    tokenizer.text.convert_ids_to_tokens(model.config.patch_token_id) * model.config.num_patches,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids.to(model.device).squeeze()
            )
        )

        self.montecarlo.child_finder = self.child_finder # type: ignore

    def __call__(self, *args, **kwargs):
        return self.simulate(*args, **kwargs)

    def simulate(self, expansions: Optional[Numeric] = 1) -> Generator[TikzDocument, None, None]:
        """
        Run the simulations. Returns all rollouts (successful or unsuccessful)
        in descending order (best rollouts first) of their score.
        """
        start_time = time()
        while expansions is None or (expansions:=expansions-1) >= 0:
            self.montecarlo.simulate()
            try:
                yield self.solution.pop()
            except IndexError:
                pass
            if self.mcts_timeout is not None and time() - start_time > self.mcts_timeout:
                return

    def generate(self, input_ids: torch.Tensor, streamer: Optional[BaseStreamer] = None, **gen_kwargs) -> torch.Tensor:
        if input_ids[-1] == self.tokenizer.eos_token_id:
            return input_ids # prevent continuing generation after encountering eos token
        with torch.inference_mode():
            return self.model.generate(
                input=input_ids,
                images=self.tokenizer.image(self.image).unsqueeze(0).to(self.model.device, self.model.dtype),
                streamer=StreamerList(filter(bool, [streamer, self.streamer])),
                **self.gen_kwargs,
                **gen_kwargs
            )

    def rollout(self, input_ids: torch.Tensor) -> Generator[torch.Tensor, None, None]:
        rollout_control, streamer = ExplicitAbort(), TokenStreamer()
        async_result = self.thread.apply_async(
            func=self.generate,
            args=[input_ids],
            kwds=dict(
                stopping_criteria=StoppingCriteriaList([rollout_control]),
                streamer=streamer,
            )
        )

        try:
            prev, line = input_ids, list()
            for token in streamer:
                line.append(token)
                if token == self.newline_id:
                    prev = torch.cat((prev, torch.tensor(line, device=prev.device)))
                    line.clear()
                    yield prev
        except GeneratorExit:
            rollout_control.abort()
            async_result.wait()

    @cast_cache(lambda tensor: tuple(tensor.tolist()))
    def decode(self, token_ids: torch.Tensor) -> TikzDocument:
        return TikzDocument(
            timeout=self.compile_timeout,
            code=self.tokenizer.text.decode(
                token_ids=token_ids,
                skip_special_tokens=True
            )
        )

    @cast_cache(lambda img: img.tobytes(), Image.frombytes)
    def score(self, image: Image.Image) -> Numeric:
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
        for new_state in (rollout:=self.rollout(node.token_ids)):
            if (new_node:=WideNode(new_state)).state in self.failed_rollouts:
                new_nodes.extend(self.failed_rollouts[new_node.state])
                rollout.close()
                break
            new_nodes.append(new_node)

        if node.is_widen_node:
            node.visits += 1
            node, new_nodes = self.merge(node.parent, new_nodes) # type: ignore

        tikz = self.decode(new_nodes[-1].token_ids)
        skip_idx = round(sqrt(len(new_nodes)))

        if tikz.has_content:
            for new_node in new_nodes[:skip_idx]:
                node.add_child(node:=new_node)
        else:
            error_idx = max(min(tikz.errors), 1) - 1 - node.depth
            for new_node in new_nodes[:min(error_idx, skip_idx)]:
                node.add_child(node:=new_node)
            self.failed_rollouts[new_nodes[error_idx].state] = new_nodes[error_idx:]

        if self.fast_metric:
            score = tikz.has_content - tikz.compiled_with_errors
        else:
            score = self.score(tikz.rasterize()) if tikz.has_content else -1 # type: ignore

        node.update_win_value(self.norm(score) if tikz.has_content and not self.fast_metric else score)
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
        tokenizer,
        temperature: float = 0.8, # based on "a systematic evaluation of large language models of code"
        top_p: float = 0.95,
        top_k: int = 0,
        compile_timeout: Optional[int] = 60, # same as old overleaf compile timeout
        fast_metric: bool = False,
        **gen_kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_kwargs: Dict[str, Any] = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_length=tokenizer.text.model_max_length,
            do_sample=True,
            compile_timeout=compile_timeout,
            fast_metric=fast_metric,
            **gen_kwargs
        )

    def load(self, image: Union[Image.Image, str], preprocess: bool = True):
        image = load(image)
        if preprocess:
            return expand(image, max(image.size), trim=True)
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
            tokenizer=self.tokenizer,
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
    ) -> Generator[TikzDocument, None, None]:
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
            tokenizer=self.tokenizer,
            mcts_timeout=timeout or None,
            image=self.load(image, preprocess=preprocess),
            **self.gen_kwargs,
            **gen_kwargs
        )

        yield from generator.simulate(expansions or None)

    def __call__(self, *args, **kwargs) -> TikzDocument:
        return self.sample(*args, **kwargs)
