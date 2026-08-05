# Gemma 3 27B IT Tokenizer

This directory contains the tokenizer used to reproduce prompt token counts for
the Gemma 3 27B IT prefix-cache experiments.

- Source model: `google/gemma-3-27b-it`
- Runtime file: `tokenizer.json`
- SHA-256: `4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795`

The model weights and other tokenizer export files are not required because
`prompts/generate_long_prompts.py` loads `tokenizer.json` directly with the
`tokenizers` package and renders the one-turn Gemma 3 chat prompt explicitly.
