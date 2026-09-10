# Pruned BPE and DH-BPE Tokenization

This project provides Python/Cython implementations and experimental data for two publications: Pruned BPE and DH-BPE.

Both tokenization methods share the same main trainer implementation. The Cython-based trainer entry points are:

`PrunedBPETrainerCython.py` and `PrunedBPETrainerCythonParallel.py`

The first is single-threaded, while the second supports parallel processing across multiple CPU cores. Therefore, the parallel version is generally recommended on multi-core systems.

Both Cython trainers require their corresponding Cython core to be compiled before use.

All trainers support checkpoints, allowing long-running training jobs to be resumed. This is especially useful when adjusting `train_vocab_size` or `visible_vocab_size`, since appropriate values may not be known in advance when working with large training corpora. When resuming from a checkpoint, you can update `train_vocab_size`, `visible_vocab_size`, and `min_exposure_count`.

`PrunedBPETrainerCythonParallel.py` provides two main training methods: one for Pruned BPE and one for DH-BPE.

A pure Python version of the trainer is also included: `PrunedBPETrainer.py`.

After training with any of the trainers above, two vocabulary files are generated: `vocab.txt` and `inter_vocab.txt`. You can then use `PrunedBPETokenizer.py` to perform tokenization. `inter_vocab.txt` is used only in Pruned BPE and is discarded in DH-BPE.

`BPETrainer.py` and `BPETokenizer.py` are implementations of the vanilla BPE algorithm and are included for reference only.

## License

Copyright 2026 Kenny Shao

Licensed under the Apache License, Version 2.0. See [LICENSE.txt](LICENSE.txt) for details.

## Overview of Pruned BPE and DH-BPE

Pruned BPE is based on the standard Byte Pair Encoding training process. It still learns merge rules by repeatedly merging the most frequent adjacent token pair in the training corpus. The difference is that, after training, learned tokens are analyzed by their final exposure counts. Frequently exposed tokens are saved in `vocab.txt` as model-visible tokens, while low-exposure tokens are saved in `inter_vocab.txt` as internal construction tokens.

In `PrunedBPETrainerCythonParallel.py`, Pruned BPE uses a two-stage training procedure. Stage 1 performs standard BPE training, while Stage 2 continues training until the desired model-visible vocabulary is fully filled.

For a detailed description and evaluation of the Pruned BPE algorithm, please refer to the arXiv preprint: [Pruned BPE: Post-training Visibility Pruning and Token Reallocation for Byte Pair Encoding](https://arxiv.org/abs/2608.00837).

DH-BPE is a vocabulary-construction method that combines token exposure under exact minimum-token segmentation with the hierarchical dependencies induced by BPE training. Starting from a modestly overshot BPE candidate vocabulary, DH-BPE uses dynamic programming to measure candidate utility and applies exposure-guided, dependency-aware pruning to select a fixed-size model-visible vocabulary.

In this project, `PrunedBPETrainerCythonParallel.py` is used to train an overshot vocabulary based on the desired overshoot factor, after which `PruneDPVocab.py` performs vocabulary pruning using the specified pruning ratio.

For a detailed description and evaluation of the DH-BPE algorithm, please refer to the arXiv preprint: [Dynamic-Programming-Guided Hierarchical BPE and Empirical Analysis of Vocabulary Pruning](https://arxiv.org/abs/2609.06898).

## Pretokenization

The pretokenization step splits input text into smaller chunks before byte-level BPE training or tokenization. In this implementation, the same pretokenization logic is used by both the trainer and tokenizer. This helps avoid undesirable merges across boundaries such as markup tags, punctuation boundaries, or code-like structures.

The pretokenizer is an implementation detail of this project; the core Pruned BPE idea can still be applied to other BPE training pipelines.

## Corpus Data

Two training corpora and one additional evaluation corpus are provided. Corpus I and Corpus II are used for training, while Corpus III is used for evaluation and parameter tuning in the DH-BPE experiments.

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

The two training corpora do not overlap. During preprocessing, boilerplate content, social-media sharing widgets, duplicated adjacent lines, encoding artifacts, and website-specific templates or navigation text were removed.

Due to GitHub file size limitations, some corpus files are stored in compressed `.7z` format. Two `.txt` files in Corpus I and four `.txt` files in Corpus II are provided as `.7z` archives. These files should be extracted before training or evaluation.

You are welcome to incorporate additional training data, provided that all input files are encoded in **UTF-8**.

### Corpus III

[This corpus](Corpus/Corpus3) contains approximately **900 MB** of data across 17 files, all distributed in compressed `.7z` format. It is an evaluation and parameter-tuning corpus used only in the DH-BPE paper. It contains five subsets with different degrees of similarity to the training data:

1. Approximately **170 MB** of Chinese medical-domain text.

2. **40 MB** of Chinese Weibo text drawn from the same general source and with a similar distribution to the social-media data in Corpus I, but without overlapping samples.

3. **90 MB** of Simplified and Traditional Chinese Wikipedia text drawn from the same source type as the Wikipedia data in Corpus I and Corpus II, but using non-overlapping pages covering different topics.

4. **240 MB** of English legal text sampled from the CourtListener opinions component of the Pile of Law dataset.

5. **363 MB** of English Reddit text drawn from source files not used for the Reddit data in Corpus I and Corpus II.

---

## 1. Python Version Requirement

This project requires **Python 3.10+**.

Python 3.10 or newer is needed because the pure Python trainer uses:

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

`pruned_bpe_pretokenizer.py` contains the shared pretokenization logic used by the Pruned BPE trainers and `PrunedBPETokenizer.py`.

The main trainer, tokenizer, and DH-BPE vocabulary-processing files are:

```text
PrunedBPETrainer.py
PrunedBPETrainerCython.py
PrunedBPETrainerCythonParallel.py
PrunedBPETokenizer.py
PruneDPVocab.py
MinTokenDPTokenizer.py
```

### Trainers, Dependent Files, and Other Core Scripts

#### `PrunedBPETrainer.py`

This is the pure Python version of the trainer. Other than `pruned_bpe_pretokenizer.py`, it does not depend on any other project files.

It is useful if you do not want to set up a Cython environment. It is also the base class for the other two trainer classes.

#### `PrunedBPETrainerCython.py`

This trainer depends on `bpe_fast_core.pyx`. The `setup_bpe_fast.py` file shows how the Cython core should be compiled.

`bpe_fast_core.pyi` is optional and provides type annotations and API structure information for the compiled Cython module.

#### `PrunedBPETrainerCythonParallel.py`

This trainer depends on `bpe_fast_core_parallel.pyx`. The `setup_bpe_fast_parallel.py` file shows how the parallel Cython core should be compiled.

Similar to the `.pyi` file described above, `bpe_fast_core_parallel.pyi` is optional.

#### `PruneDPVocab.py`

This vocabulary-pruning script is specifically used by DH-BPE. Given a modestly overshot BPE candidate vocabulary, the original training corpus, the desired overshoot factor, and the pruning ratio, it outputs the final target vocabulary file.

The minimum-token DP encoding algorithm in this script uses the same implementation as the one in `MinTokenDPTokenizer.py`. Because vocabulary pruning is a core training step in DH-BPE, the implementation is self-contained.

#### `MinTokenDPTokenizer.py`

This is an independent dynamic-programming encoder that minimizes the number of output tokens using only a list of model-visible tokens.

`ConvertVocab.py` in the tools folder converts the `vocab.txt` file generated by any Pruned BPE trainer into the format required by this encoder.

This tokenizer is also used by the `MinTokenDPTokenizerExp.py` script in the experiment folder, which was used for the experimental evaluations in the DH-BPE publication.

---

## 5. Compile the Cython Extension

### Option 1: Run the `PrunedBPETrainerCython` Trainer

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

### Option 2: Run the `PrunedBPETrainerCythonParallel` Trainer

From the project root folder, run:

```bash
python setup_bpe_fast_parallel.py build_ext --inplace
```

Similar `.pyd`, `.c`, or `.cpp` files will be generated.

Note that the `build` folder generated during the compilation process is temporary. It can be kept or safely deleted.

---

## 6. Run the Cython Trainer

After the Cython extension has been compiled successfully, run:

```bash
python PrunedBPETrainerCython.py
```

or:

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

If you modify a `.pyx` file, you must compile it again. For example:

```bash
python setup_bpe_fast.py build_ext --inplace
```

For the parallel trainer, use:

```bash
python setup_bpe_fast_parallel.py build_ext --inplace
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

Then run the trainer again.

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

### Problem: PyCharm Uses the Wrong Python Environment

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

### Problem: PowerShell Blocks Virtual Environment Activation

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

The Visual Studio Build Tools path may vary depending on the installed Visual Studio version and installation location.

For the parallel trainer, replace the final two commands with:

```bash
python setup_bpe_fast_parallel.py build_ext --inplace
python PrunedBPETrainerCythonParallel.py
```
