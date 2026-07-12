# Pruned BPE Tokenization

This project presents a Python/Cython implementation of the Pruned-BPE tokenization algorithm, including its Trainers and Tokenizer.

The main Cython-based trainer entry file is:

`PrunedBPETrainerCython.py` or `PrunedBPETrainerCythonParallel.py`

The first is single-threaded, while the second supports parallel processing and can efficiently use multiple CPU cores. Therefore, the parallel version is the better choice if you have an advanced multi-core CPU.

Both trainers require their corresponding Cython core to be compiled before running.

All trainers support checkpoints, which allow you to resume training if you feel the training process is too long. This is especially helpful when you need to adjust `train_vocab_size` or `visible_vocab_size`, because it may not be clear which size is suitable when working with a large amount of training data. You can update `train_vocab_size`, `visible_vocab_size`, and `min_exposure_count` when you resume training from a checkpoint.

In a production environment, when you want an exact number of model-visible tokens (`visible_vocab_size`), it is suggested to use two-stage training:

```text
Stage 1:
    Set train_vocab_size = visible_vocab_size.
    Train standard BPE until desired visible_vocab_size.
    Run visibility analysis and export vocab.txt and inter_vocab.txt.
    Save checkpoint at visible_vocab_size.

Stage 2:
    Set min_exposure_count = a percent (suggested to be 20%, 30% or 40%) of the frequency/count of the last token learned in Stage 1.
    Set MAX_TRAIN_VOCAB_SIZE = ceil(1.2 * visible_vocab_size) as the safety upper bound.
    Resume training from the checkpoint.
    Run visibility pruning again, stop exporting once next_visible_id >= visible_vocab_size and discard all later extra tokens.
```

This shows the basic idea, which has already been implemented in the code. You only need to update the related parameters under `__main__`, stop training when needed, and resume training.

A pure Python version of the trainer is also included: `PrunedBPETrainer.py`.

After training with any of the above trainers, two vocab files will be generated: `vocab.txt` and `inter_vocab.txt`. You can then use `PrunedBPETokenizer.py` to perform tokenization.

`BPETrainer.py` and `BPETokenizer.py` are implementations of the vanilla BPE algorithm, which are included here for reference only.

## License

Copyright 2026 Kenny Shao

Licensed under the Apache License, Version 2.0. See [LICENSE.txt](LICENSE.txt) for details.

## Basic Introduction of the Pruned BPE Algorithm

Pruned BPE is based on the standard Byte Pair Encoding training process. It still learns merge rules by repeatedly merging the most frequent adjacent token pair in the training corpus. The difference is that, after training, learned tokens are analyzed by their final exposure counts. Frequently exposed tokens are saved in `vocab.txt` as model-visible tokens, while low-exposure tokens are saved in `inter_vocab.txt` as internal construction tokens. The stage 2 training described above can later continue training so that `visible_vocab_size` can reach the originally desired number.

Standard BPE keeps all learned tokens in the model-visible vocabulary. This can expose many intermediate tokens that are useful for constructing longer tokens but rarely appear in the final tokenized corpus. Pruned BPE keeps the standard BPE merge process, but separates learned tokens into visible tokens and internal-only tokens after training. This saves valuable visible vocabulary space for other later-learned, much higher-frequency tokens.

Scaffold BPE also focuses on tokens that mainly serve as intermediate construction pieces. It dynamically removes low-frequency scaffold tokens from token representations to reduce frequency imbalance. Pruned BPE is different because it performs visibility analysis after normal BPE training, making the statistics more stable and predictable.

Vocabulary Trimming is usually a post-processing step that removes rare subwords from an already trained BPE vocabulary, often using a third-party tool such as `subword-nmt`. Pruned BPE, instead, is integrated with token-ID remapping and token reallocation. It allows you to seamlessly train extra tokens to make use of the saved visible token space until `visible_vocab_size` reaches the desired target.

## Introduction of the Pretokenization Step

The pretokenization step splits input text into smaller chunks before byte-level BPE training or tokenization. In this implementation, the same pretokenization logic is used by both the trainer and the tokenizer. This helps avoid undesirable merges across boundaries such as markup tags, punctuation boundaries, or code-like structures. The pretokenizer is an implementation detail of this project; the core Pruned BPE idea can still be applied to other BPE training pipelines.

## Corpus Data

Two training corpora are provided. They were also used during the development and testing of the proposed algorithm to verify the correctness of all trainer implementations and the tokenizer.

### Corpus I

[This corpus](Corpus/Corpus1/) contains approximately **640 MB** of data, including:

1. **English text** (approximately **430 MB**), consisting primarily of a locally collected sample from **FineWeb-Edu**, an educational English web-text dataset derived from the FineWeb/Common Crawl pipeline, together with a sample of Reddit posts.

2. **Chinese text** (approximately **204 MB**), consisting of conversational text collected from several Chinese social media platforms and text from Chinese Wikipedia pages, including both Simplified and Traditional Chinese.

3. **A small amount of source code**, including Java, Python, JavaScript, TypeScript, HTML, JSON, XML, and related formats.

4. **A small multilingual corpus** covering numerous additional languages, including French, German, Portuguese, and Finnish.

### Corpus II

[This corpus](Corpus/Corpus2/) contains approximately **1 GB** of data, including:

1. **English text**, consisting of:

   * A **360 MB** subset randomly sampled from **CC-News**, a Common Crawl-derived corpus containing news articles from a wide range of news websites.
   * Approximately **180 MB** of randomly sampled Reddit posts.

2. **Chinese text**, consisting of:

   * A **382 MB** subset randomly sampled from **THUCNews**, a Chinese news text classification corpus released as part of the THUCTC project. The selected subset covers 14 news categories: **Sports**, **Entertainment**, **Home**, **Lottery**, **Real Estate**, **Education**, **Fashion**, **Politics**, **Horoscopes**, **Gaming**, **Society**, **Technology**, **Stocks**, and **Finance**.
   * Approximately **30 MB** of text from Chinese Wikipedia pages, including both Simplified and Traditional Chinese.

3. **A small amount of source code**, including Java, Python, JavaScript, TypeScript, C++, Markdown, and HTML.

4. **An 84 MB multilingual corpus** covering **42 languages** other than English and Chinese, with approximately equal amounts of text for each language. These languages include Arabic, French, Spanish, Portuguese, Russian, Japanese, Korean, and many others.

The two corpora do not overlap. During preprocessing, boilerplate content, social-media sharing widgets, duplicated adjacent lines, encoding artifacts, and website-specific templates or navigation texts were removed.

Due to GitHub file size limitations, some corpus files are stored in compressed .7z format. Two .txt files in Corpus1 and four .txt files in Corpus2 are provided as .7z archives. These files should be extracted before training or evaluation.

You are welcome to incorporate additional training data, provided that all input files are encoded in **UTF-8**.

---

## 1. Python Version Requirement

This project requires **Python 3.10+**.

Python 3.10 or newer is needed because the project (the pure Python trainer) uses:

```python
from itertools import pairwise
```

Check your Python version:

```bash
python --version
```
---

## 2. Create and Activate a Virtual Environment

From the project root folder, create a virtual environment:

```bash
python -m venv .venv
```

Activate it on **Windows PowerShell**:

```bash
.venv\Scripts\Activate.ps1
```

Activate it on **Windows Command Prompt**:

```bash
.venv\Scripts\activate.bat
```

After activation, the terminal should show something like:

```text
(.venv)
```

---

## 3. Install Required Packages

Install the required Cython build tools:

```bash
pip install -r requirements.txt
```

The minimum `requirements.txt` for this project is:

```txt
cython
setuptools
wheel
```
---

## 4. Project Files

`pruned_bpe_pretokenizer.py` is used by all Pruned BPE trainers and by the Pruned BPE tokenizer:

```text
PrunedBPETrainer.py
PrunedBPETrainerCython.py
PrunedBPETrainerCythonParallel.py
PrunedBPETokenizer.py
```

### Trainers and Dependent Files

`PrunedBPETrainer.py`

The pure python version of the trainer file. Other than `pruned_bpe_pretokenizer.py`, it does not depend on any other files.

This is useful if you do not want to set up a Cython environment. It is also the base/parent class file for the other two trainer classes.

`PrunedBPETrainerCython.py`

This depends on `bpe_fast_core.pyx`. The `setup_bpe_fast.py` file shows how the core should be compiled. `bpe_fast_core.pyi` is optional, which just provides type annotations and API structures for the main Cython.py file.

`PrunedBPETrainerCythonParallel.py`

This depends on `bpe_fast_core_parallel.pyx`. The `setup_bpe_fast_parallel.py` file shows how the core should be compiled.

---

## 5. Compile the Cython Extension

### Option 1: Run PrunedBPETrainerCython Trainer

From the project root folder, run:

```bash
python setup_bpe_fast.py build_ext --inplace
```

This command compiles the `.pyx` file and creates a compiled extension file in the project folder.

On Windows, the generated file may look like `bpe_fast_core.cp313-win_amd64.pyd` or `bpe_fast_core.cp314-win_amd64.pyd`.

The exact file name depends on your Python version.

After compilation, `PrunedBPETrainerCython.py` should be able to import the compiled Cython module, for example:

```python
from bpe_fast_core import find_best_pair
```
---

### Option 2: Run PrunedBPETrainerCythonParallel Trainer

From the project root folder, run:

```bash
python setup_bpe_fast_parallel.py build_ext --inplace
```

Similar `.pyd`, `.c`, or `.cpp` files will be generated.

Note that the `build` folder generated during the above compilation processes is temporary. It can be kept or safely deleted.

## 6. Run the Cython Trainer

After the Cython extension has been compiled successfully, run:

```bash
python PrunedBPETrainerCython.py
```
OR
```bash
python PrunedBPETrainerCythonParallel.py
```

If you are using PyCharm, make sure the project interpreter is set to the same virtual environment where Cython was installed.

In PyCharm:

```text
File → Settings → Project → Python Interpreter
```

Choose the `.venv` interpreter for this project.

The interpreter path usually looks like:

```text
.venv\Scripts\python.exe
```
---

## 7. Rebuild After Changing Cython Code

If you modify the `.pyx` file, you must compile again, such as (or the parallel one):

```bash
python setup_bpe_fast.py build_ext --inplace
```

Then run the corresponding trainer again.

Changing only `.py` files usually does **not** require recompiling.

Recompilation is mainly needed after changing files such as:

```text
.pyx
.pxd
setup_bpe_fast.py
setup_bpe_fast_parallel.py
```
---

## 8. Common Problems

### Problem: `ModuleNotFoundError: No module named 'bpe_fast_core'`

This usually means the Cython extension has not been compiled yet.

Run:

```bash
python setup_bpe_fast.py build_ext --inplace
```

Then run the trainer.

---

### Problem: `Cython is not installed`

Install the requirements:

```bash
pip install -r requirements.txt
```

or install Cython directly:

```bash
pip install cython
```

---

### Problem: PyCharm uses the wrong Python environment

Make sure PyCharm uses the project virtual environment:

```text
File → Settings → Project → Python Interpreter
```

Select the interpreter inside the project `.venv` folder:

```text
.venv\Scripts\python.exe
```

Then reinstall the requirements if needed:

```bash
pip install -r requirements.txt
```

---

### Problem: PowerShell blocks virtual environment activation

If PowerShell does not allow activation, you may see an execution policy error.

You can use Command Prompt instead:

```bash
.venv\Scripts\activate.bat
```

Or, in PowerShell, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try again:

```bash
.venv\Scripts\Activate.ps1
```

---

## 9. Quick Start Summary

For Windows Command Prompt:

```bash
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set DISTUTILS_USE_SDK=1
set MSSdk=1
python setup_bpe_fast.py build_ext --inplace
python PrunedBPETrainerCython.py
```

For the parallel trainer, change the commands above accordingly.