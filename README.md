# q

I'd like to learn about how transformers work in practice, including their architecture and applications in real-world scenarios. Therefore, I decided to create my own GPT model. First, I plan to learn from excellent articles and papers written by other people to understand basic concepts and its architecture. Gradually, I will build my own GPT model. Finally, I aim to use it in web application and games that interest me.

## Install

```sh
$ poetry install
```

## Download model parameters

You have to download an OpenAI model parameters before executing `q`:

```sh
$ poetry install --extras download
$ poetry run download --model_size 124M
```

Available models:

- `124M`
- `355M`
- `774M`
- `1558M`

## Run

numpy

```sh
poetry run q "Alan Turing theorized that computers would one day become"
Generating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:04<00:00,  8.54it/s]
Generated 8.50 tokens/sec

Alan Turing theorized that computers would one day become the most powerful machines on the planet.

The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.
```

mlx

```sh
$ poetry install --extras mlx
$ poetry run q --backend mlx "Alan Turing theorized that computers would one day become"
Generating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:01<00:00, 36.94it/s]
Generated 36.40 tokens/sec

Alan Turing theorized that computers would one day become the most powerful machines on the planet.

The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.
```

## References

- [GPT in 60 Lines of NumPy | Jay Mody](https://jaykmody.com/blog/gpt-from-scratch/)
  - [jaymody/picoGPT: An unnecessarily tiny implementation of GPT-2 in NumPy.](https://github.com/jaymody/picoGPT/tree/main)
