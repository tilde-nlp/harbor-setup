# Golden tokenizations

Ground truth for the tokenizer checks: one file per model family, `<model dir name>.jsonl`, each line
`{"messages": [...], "ids": [...]}`: a chat prompt and the token ids the model must see for it
(chat template applied, generation prompt included).

A golden file depends on the tokenizer and the chat format, not on the weights: every checkpoint of
a model family shares it. The checks look for `<model dir>/golden_tokenization.jsonl` first (an
override for one directory), then for `checks/golden/<basename of the model dir>.jsonl`.

Build one with `python -m checks.make_golden` from data whose token ids are trusted, i.e. produced
by the pipeline that trained or sampled the model.

| file | prompts | token ids from |
|---|---|---|
| `tildeopen.jsonl` | 300 (63 with a system message) | the tokenized so_openq rollouts sampled from TildeOpen chat-sft-v1 with vLLM |

## Attribution and license

The prompts in `tildeopen.jsonl` come from the `so_openq` split of
[nvidia/Nemotron-RL-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Science-v1),
licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Changes made: a subset
of 300 prompts, a tool-description system message on some of them, token ids added.
`tildeopen.jsonl` is distributed under the same license, CC BY-SA 4.0. That applies to that file,
not to the code in this repository.
