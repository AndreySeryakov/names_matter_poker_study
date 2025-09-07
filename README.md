In general this is a script which allows to ask the same question to a given LLM many times, but it was finetuned for a given poker situation.

However, it's not hard to remove all poker stuff and make it whatever you whish (even dragon, not you can't make a dragon out of it). 

Here is a readme I asked chatGPT5 to create with thecnical information. You are always welcome to ask questions, I will help a.u.seryakov@gmail.com

---

# Names Matter — Poker Decision Study

Quick-start script for probing how LLM decisions in a fixed poker spot vary with **names**.
Two experiment modes are supported:

* **OPPONENT\_NAME** — vary the opponent’s first name in the prompt.
* **USER\_NAME** — vary *your* (the player’s) first name in the prompt.

> ⚠️ Analysis scripts downstream are tuned to parse answers as one of:
> **`Fold`**, **`Call`**, or **`Raise Xbb`** (e.g., `Raise 10bb`). 

---

## What’s in this repo

* `names_matter_poker_study.py` — main runner (your script).
* `instrumentation.py` — tiny helpers for timestamps and (WIP) JSONL logging.
* `.env` — **not committed**; holds API keys.

AS: analysis scripts, I didn't provide them to chatGPT, so there is nothing about them.
But one of them runs authomaticaly so you will get a hist with disctibution of answers.

---

## Requirements

* Python 3.10+ (other versions likely fine).
* Install deps (keep it simple, taken directly from code):

  ```bash
  pip install openai together python-dotenv
  ```

---

## Setup

1. Create a `.env` in the project root:

   ```ini
   OPENAI_API_KEY=your_openai_key_here
   TOGETHER_API_KEY=your_together_key_here
   ```

2. Open `names_matter_poker_study.py` and set your defaults at the top of the file:

   * Choose **provider** (OpenAI or Together) and model.
   * Set temperature / token limits if desired.
   * Pick the **EXPERIMENT**: `OPPONENT_NAME` or `USER_NAME`.
   * Adjust the long prompts (`system` + task prompts) **in the file**.
     (This is intentional: prompts are meant to be long and curated.)

   **Name rules** used by the script:

   * If you provide a first name, default last name is **Smith**.
   * If you provide **no name at all**, the baseline is **“no-name”** phrasing (e.g., “You are now playing heads-up.”).
     AS: no name mode is for the baseline. Just don't provide names and the code will run it. 

---

## Running

You can pass names via CLI. Everything else can stay in code (typical workflow).

* **Baseline (no name)**

  ```bash
  python names_matter_poker_study.py
  ```

* **Single first name**

  ```bash
  python names_matter_poker_study.py --first-name Mia
  ```

* **Multiple names inline**

  ```bash
  python names_matter_poker_study.py --names Mia,Olivia,Emma
  ```

* **Names from file (one per line)**

  ```bash
  python names_matter_poker_study.py --names-file file_of_names.txt
  ```

  > This is the common usage:
  > `python system_prompt_study.py --names-file file_of_names.txt`
  > (If you renamed the script, just use your filename.)

### Thinking mode (experimental)

* Available **only for `OPPONENT_NAME`** experiment.
* Runs a two-stage flow: *analysis (“thinking”)* → *final decision*.
* **Produces many tokens**. “Reasoning” models are **not yet tested**.

Enable/disable via the script flags (documented in the code), e.g.:

```bash
python names_matter_poker_study.py --names-file file_of_names.txt --thinking
```

---

## Outputs

* Per-run transcript:

  ```
  results/<PREFIX>_llm_responses_<timestamp>_{direct|thinking}.txt
  ```

  Contains prompts and model responses for each tested name.

* Token spend tracker:

  ```
  token_usage.txt
  ```

  A simple running log to watch costs.

* JSONL logging: **work in progress** (you can ignore for now).

* Logprobs: **work in progress** (first-token logprobs; model/provider support varies).

---

## Prompts & answer format

* **Edit prompts directly in the file** (recommended).
* Keep the model’s final answer in one of these forms for compatibility with analysis:

  * `Fold`
  * `Call`
  * `Raise Xbb` (e.g., `Raise 8bb`, `Raise 12bb`)

You can add richer reasoning text, but ensure the final actionable line follows that format.

---

## Notes

* Models intentionally **not** documented here (they depend on your API access).
  Set them at the top of the script.
* If you later wire up JSONL or logprobs fully, update this README’s “Outputs” section.

---

## Example one-liners

* Opponent-name sweep from file:

  ```bash
  python names_matter_poker_study.py --names-file file_of_names.txt
  ```
* Single controlled test:

  ```bash
  python names_matter_poker_study.py --first-name Liam
  ```
* Quick multi-name run:

  ```bash
  python names_matter_poker_study.py --names Ava,Isabella,Evelyn
  ```

---

That’s it — keep prompts long in the file, vary names via flags, and read results from `results/` with costs tracked in `token_usage.txt`.
