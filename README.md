# zh-to-en

Walks every file in a target repo and translates Chinese characters to English
using a LiteLLM proxy running at `localhost:4000/v1` (on a different machine).

## Features

- **File-by-file walk**: recursively processes a repo folder passed as a CLI arg
- **Binary-safe**: skips binary files and common vendor/build/source-control dirs
- **Preserves formatting**: replaces only the Chinese text runs, leaving the rest
  of the file untouched, so indentation and surrounding code/comments are intact
- **Same-length aware**: replaces in place, run-by-run, not line-by-line, keeping
  code structure valid
- **Dry-run**: show what would change without writing
- **Filename translation**: optionally translates Chinese characters in filenames
  (stem only, extension preserved) with `--rename`
- **Concurrency**: processes files with a thread pool for speed

## Requirements

- Python 3.8+
- `openai` client library (to talk to the LiteLLM OpenAI-compatible endpoint)
- A running LiteLLM proxy on another machine reachable at `localhost:4000`

## Setup

```bash
pip install -r requirements.txt
```

The LiteLLM proxy must expose `/v1` on `localhost:4000` on this machine (e.g. via
`ssh -L 4000:localhost:4000 user@proxy-machine` if the proxy runs elsewhere).

## Usage

```bash
# dry run (default, shows what would change)
python zh_to_en.py /path/to/repo

# actually write changes
python zh_to_en.py /path/to/repo --apply

# run only on one file
python zh_to_en.py /path/to/repo/file.py --apply

# include vendored dirs like node_modules
python zh_to_en.py /path/to/repo --apply --no-ignore

# use the whole file as context for better translations
python zh_to_en.py /path/to/repo --apply --mode wholefile

# point at a different model / endpoint
python zh_to_en.py /path/to/repo --apply --model gpt-4o-mini --base-url http://localhost:4000/v1

# also translate Chinese characters in filenames
python zh_to_en.py /path/to/repo --apply --rename

# dry run showing what would be renamed
python zh_to_en.py /path/to/repo --rename
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LITELLM_API_KEY` | `sk-dummy` | API key accepted by the LiteLLM proxy |
| `LITELLM_BASE_URL` | `http://localhost:4000/v1` | OpenAI-compatible base URL |
| `LITELLM_MODEL` | `gpt-4o-mini` | Model name registered in the proxy |

## How it works

1. Recursively walk the target directory.
2. Skip ignored dirs and binary/non-UTF8 files.
3. Detect runs of Chinese characters (`\u4e00-\u9fff`, with full-width punctuation).
4. Translate each run, then replace them in place, preserving surrounding
   whitespace and structure.
5. Write the file back only when `--apply` is given.
6. If `--rename` is passed, translate Chinese characters in filenames after all
   content translation completes. Files are renamed deepest-first to avoid path
   conflicts; renames are skipped if the target name already exists.

A small in-session cache avoids re-translating identical strings.

## Modes

- **`spans`** (default) — each Chinese snippet is sent to the model on its own.
  Cheap and surgical; the model sees only the string to translate.
- **`wholefile`** — the entire file is sent along with the list of Chinese spans;
  the model translates them all in one call using the surrounding code as context.
  Produces more consistent, style-aware output but costs more tokens and rounds
  back to per-span translation for any span the model misses.

In both modes only the exact Chinese spans are replaced — surrounding code is
never trusted or rewritten by the model.

## Caveats

- Inline (comment-adjacent) translations can change line length; that is
  expected. Output is functionally equivalent but comments/strings become
  English.
- Batch translation uses a single continuation string; large files are handled
  in smaller per-file batches to avoid reply truncation.
- This rewrites files in place. Use `--apply` only after reviewing `--dry-run`.
- `--rename` processes filenames after all file contents are translated. If a
  translated filename already exists, that rename is skipped to avoid data loss.
