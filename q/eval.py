import logging
import os
from typing import Any, Iterator, List, Literal, Optional, Tuple

import mlx.core as mx
from lm_eval import utils
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator
from mlx.nn import log_softmax
from tqdm import tqdm

from .common import ModelSize
from .encoder import Encoder, load_encoder
from .generation import TokenGenerator
from .gpt2 import GPT2Model
from .params import load_hparams_and_params

eval_logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@register_model("q")
class QLM(TemplateLM):
    """
    A custom language model class for lm-evaluation-harness.
    """

    model: GPT2Model

    encoder: Encoder

    generator: TokenGenerator

    max_length: int

    batch_size: int

    def __init__(
        self,
        model_size: ModelSize = "124M",
        models_dir: str = DEFAULT_MODELS_DIR,
        max_length: int = 1024,
        batch_size=1,
    ) -> None:
        super().__init__()

        # print(
        #     f"Loading model {model_size}, batch_size={batch_size}, max_length={max_length}"
        # )

        # load encoder, hparams, and params from the released open-ai gpt-2 files
        self.encoder = load_encoder(model_size, models_dir)
        hparams, params = load_hparams_and_params(
            model_size=model_size,
            models_dir=models_dir,
        )

        self.model = GPT2Model(params, hparams)
        self.generator = TokenGenerator(self.model)
        self.max_length = max_length
        self.batch_size = batch_size

    @property
    def eot_token_id(self) -> int:  # type: ignore
        # In the original GPT-2 implementation, there is no explicit EOS (End of
        # Sequence) token. citeturn0search7 Consequently, during text
        # generation, the model may continue producing tokens until it reaches
        # the maximum sequence length.
        #
        # So we use 0 as a placeholder for the end of the sequence token.
        return 0

    @property
    def prefix_token_id(self) -> int:  # type: ignore
        return self.eot_token_id

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        """
        Tokenize a string using the model's tokenizer and return a list of token IDs.
        """
        return self.encoder.encode(string)

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.encoder.decode(tokens)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(tokens) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            tokens = req[1] + req[2]
            return -len(tokens), tuple(tokens)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by=None,
            group_fn=_lookup_one_token_cont,
        )

        chunks: Iterator[Iterator[tuple[Any, list[int], list[int]]]] = (
            re_ord.get_batched(n=self.batch_size)
        )
        tqdm_bar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )

        for chunk in chunks:
            inputs = []
            cont_tokens_list = []
            input_lens = []

            padding_len_inp = 0
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                total_length = len(context_enc) + len(continuation_enc)
                if total_length > self.max_length + 1:
                    eval_logger.warn(
                        f"Combined length of context ({len(context_enc)}) and continuation ({len(continuation_enc)}) "
                        f"exceeds model's maximum length ({self.max_length}). "
                        f"Truncating {total_length - self.max_length + 1} tokens from the left."
                    )
                inp = mx.array(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=mx.int32,
                )
                (input_len,) = inp.shape

                padding_len_inp = (
                    max(padding_len_inp, input_len)
                    if padding_len_inp is not None
                    else input_len
                )

                inputs.append(inp)  # [1, inp_length]
                cont_tokens_list.append(continuation_enc)
                input_lens.append(input_len)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            batched_inputs = pad_and_concat(
                padding_len_inp, inputs, padding_side="right"
            )  # [batch, padding_len_inp]

            # [batch, padding_length (inp or cont), vocab]
            multi_logits = log_softmax(self._model_call(batched_inputs, **call_kwargs))

            for (request_str, ctx_tokens, _), logits, input_len, cont_tokens in zip(
                chunk, multi_logits, input_lens, cont_tokens_list
            ):
                # Slice to original seq length
                cont_len = len(cont_tokens)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = input_len + (logits.shape[0] - padding_len_inp)
                logits = self._select_cont_tokens(
                    logits, cont_len=cont_len, input_len=ctx_len
                )
                logits = mx.expand_dims(logits, axis=0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(axis=-1)

                # [1, seq]
                cont_tokens = mx.array(cont_tokens, dtype=mx.int32)
                cont_tokens = mx.expand_dims(cont_tokens, axis=0)
                max_equal = mx.all(mx.equal(greedy_tokens, cont_tokens))

                # logits: [batch_size, seq_length, vocab_size]
                # cont_tokens: [batch_size, seq_length]

                # cont_tokens に新たな次元を追加して [batch_size, seq_length, 1] にする
                cont_tokens_expanded = mx.expand_dims(cont_tokens, axis=-1)

                # 指定軸 (-1) に沿って値を抽出し、不要な次元を除去
                logits = mx.take_along_axis(
                    logits, cont_tokens_expanded, axis=-1
                ).squeeze(-1)

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                res.append(answer)

                if request_str is not None:
                    # special case: loglikelihood_rolling produces a number of loglikelihood requests
                    # all with cache key None. instead do add_partial on the per-example level
                    # in the loglikelihood_rolling() function for those.
                    self.cache_hook.add_partial("loglikelihood", request_str, answer)

                tqdm_bar.update(1)

        tqdm_bar.close()

        return re_ord.get_original(res)

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        def _collate(req: Tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size if adaptive_batch_size is not None else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and not adaptive_batch_size
            else None
        )

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                # add EOS token to stop sequences
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            if self.backend == "causal":
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
                assert (
                    max_ctx_len > 0
                ), f"Invalid configuration: requested max tokens to generate ({max_gen_toks}) must be less than model's maximum sequence length ({self.max_length})."
            elif self.backend == "seq2seq":
                # max len for inputs = encoder's whole max_length
                max_ctx_len = self.max_length

            # encode, pad, and truncate contexts for this batch
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            context_enc = context_enc.to(self.device)
            attn_masks = attn_masks.to(self.device)

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            # perform batched generation
            cont = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                stop=until,
                **kwargs,
            )

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                # discard context + left-padding toks if using causal decoder-only LM
                if self.backend == "causal":
                    cont_toks = cont_toks[context_enc.shape[1] :]

                s = self.tok_decode(cont_toks)

                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                for term in until:
                    if len(term) > 0:
                        # ignore '' separator,
                        # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                        s = s.split(term)[0]

                res.append(s)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        return res

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        adaptive_batch_size = None

        # First, collect all windows from all requests
        all_windows = []  # List of (request_idx, window) tuples
        request_window_counts = []  # Track number of windows per request

        for req_idx, (string,) in enumerate(
            tqdm(
                [req.args for req in requests],
                disable=(disable_tqdm or (self.rank != 0)),
            )
        ):
            rolling_token_windows: List[Tuple[List[int], List[int]]] = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the
            # full context to the decoder, in seq2seq case
            windows = [(None,) + x for x in rolling_token_windows]

            # Store windows with their request index
            all_windows.extend((req_idx, window) for window in windows)
            request_window_counts.append(len(windows))

        # Handle distributed case padding
        pad_amnt = 0
        all_nlls = []
        batch_size = adaptive_batch_size or self.batch_size
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]
            # Extract just the windows for processing, keeping track of request indices
            batch_indices, batch_windows = zip(*batch)

            batch_nlls = self._loglikelihood_tokens(
                requests=batch_windows,
                disable_tqdm=False,
                override_bs=len(batch_windows),
            )
            # Store results with their request indices
            all_nlls.extend(zip(batch_indices, batch_nlls))

        # Remove padding if necessary
        if (self.world_size > 1) and (pad_amnt > 0):
            all_nlls = all_nlls[:-pad_amnt]

        # Reconstruct per-request loglikelihoods
        loglikelihoods = []
        current_idx = 0
        for window_count in request_window_counts:
            # Get all nlls for this request
            request_nlls = all_nlls[current_idx : current_idx + window_count]
            # Sum up the nlls for this request (discarding is_greedy)
            request_total = sum(nll[0] for _, nll in request_nlls)
            loglikelihoods.append(request_total)
            current_idx += window_count

            string = requests[len(loglikelihoods) - 1].args[0]
            self.cache_hook.add_partial(
                "loglikelihood_rolling", (string,), request_total
            )

        return loglikelihoods

    def _model_call(self, inputs: mx.array, attn_mask=None, labels=None) -> mx.array:
        """
        :param inputs: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """

        assert attn_mask is None
        assert labels is None
        return self.model(inputs).logits

    def _model_generate(
        self, context: list[int], max_length: int, stop, **generation_kwargs
    ):
        return list(self.generator(context, max_length=max_length, **generation_kwargs))

    def _select_cont_tokens(
        self,
        logits: mx.array,
        cont_len: Optional[int] = None,
        input_len: Optional[int] = None,
    ) -> mx.array:
        assert (
            cont_len and input_len
        ), "Must pass input len and cont. len to select scored logits for causal LM"
        # discard right-padding.
        # also discard the input/context tokens. we'll only score continuations.
        logits = logits[input_len - cont_len : input_len]

        return logits


# ----------------
# Utilities
# ----------------
def pad_and_concat(
    max_length: int,
    arrays: List[mx.array],
    padding_side: Literal["right", "left"] = "right",
):
    """
    Method for padding a list of MLX arrays given the maximum array
    length in the batch. Used for batching inputs.
    """
    assert (
        padding_side == "left" or padding_side == "right"
    ), f"Unrecognized padding type: '{padding_side}' not 'left' or 'right'"

    for i, array in enumerate(arrays):
        if len(array.shape) == 2:
            # Squeeze, in case passed [1, seq] size
            array = array[0]
        tensor_len = array.shape[0]

        if tensor_len < max_length:
            if padding_side == "right":
                # right-pad
                padding = mx.zeros((max_length - tensor_len,), dtype=mx.int32)
                arrays[i] = mx.expand_dims(
                    mx.concatenate([array, padding], axis=0), axis=0
                )
            else:
                # left-pad
                padding = mx.zeros((max_length - tensor_len,), dtype=mx.int32)
                arrays[i] = mx.expand_dims(
                    mx.concatenate([padding, array], axis=0), axis=0
                )
        else:
            arrays[i] = mx.expand_dims(array, axis=0)

    return mx.concatenate(arrays, axis=0)


def main():
    mlx_random_seed = 1234
    mx.random.seed(mlx_random_seed)
    print(f"Setting mlx seed to {mlx_random_seed}")

    cli_evaluate()
