# q

![workflow](https://github.com/ishikawa/q/actions/workflows/q.yml/badge.svg)

> Homebrew small-scale LLM based on GPT-2

I'd like to gain practical experience with transformers, particularly by understanding their architecture and real-world applications, with a focus on small-scale LLMs. To achieve this, I decided to create _a tiny LLM_. First, I plan to study [excellent articles and papers](#References) to understand the basic concepts and architecture. Next, I will build and improve _my own GPT model_. My goal is to integrate it into web applications, games, and iOS apps that interest me.

Currently, I am studying by building a LLM based on [OpenAI's GPT-2 model](https://github.com/openai/gpt-2/tree/master). I used an [extremely simple numpy-based model](https://jaykmody.com/blog/gpt-from-scratch/) as a baseline and am experimenting with an implementation using [mlx](https://ml-explore.github.io/mlx/build/html/index.html).

## Install

```sh
$ poetry install
```

## Download model parameters

You have to download an OpenAI GPT-2 model parameters before executing `q`:

```sh
$ poetry install --extras download
$ poetry run download --model-size 124M
```

Available models:

- `124M`
- `355M`
- `774M`
- `1558M`

## Run

```sh
$ poetry run q "Alan Turing theorized that computers would one day become"
Generating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:00<00:00, 42.19it/s]
Generated 41.35 tokens/sec

Alan Turing theorized that computers would one day become the most powerful machines on the planet.

The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.
```

### Stream output

You can enable stream output by setting `--stream` flag:

```sh
$ poetry run q --stream "Alan Turing theorized that computers would one day become"
Alan Turing theorized that computers would one day become the most powerful machines on the planet.

The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.

Generated 37.19 tokens/sec
```

## References

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT in 60 Lines of NumPy | Jay Mody](https://jaykmody.com/blog/gpt-from-scratch/)
- [jaymody/picoGPT: An unnecessarily tiny implementation of GPT-2 in NumPy.](https://github.com/jaymody/picoGPT/tree/main)
- [Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20 · karpathy/llm.c · Discussion #481](https://github.com/karpathy/llm.c/discussions/481)
