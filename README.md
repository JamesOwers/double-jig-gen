[build-image]: https://travis-ci.com/JamesOwers/double-jig-gen.svg?branch=master
[build-url]: https://travis-ci.com/JamesOwers/double-jig-gen
[coverage-image]: https://codecov.io/gh/JamesOwers/double-jig-gen/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/JamesOwers/double-jig-gen?branch=master
<!-- [docs-image]: https://readthedocs.org/projects/midi_degradation_toolkit/badge/?version=latest
[docs-url]: https://midi_degradation_toolkit.readthedocs.io/en/latest/?badge=latest
[pypi-image]: https://badge.fury.io/py/midi_degradation_toolkit.svg
[pypi-url]: https://pypi.python.org/pypi/midi_degradation_toolkit -->

- [1. Setup](#1-setup)
  - [1.1. System](#11-system)
  - [1.2. Python environment](#12-python-environment)
  - [1.3. Get data](#13-get-data)
  - [1.4. Split abc into tokens](#14-split-abc-into-tokens)
  - [1.5. Create the experimental splits](#15-create-the-experimental-splits)
- [2. Dev Setup](#2-dev-setup)
- [3. Archive](#3-archive)
  - [3.1. Configure Music21](#31-configure-music21)

# `double_jig_gen` - Double Jig Folk Music Generation  <!-- omit in toc -->

[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]
<!-- [![PyPI Version][pypi-image]][pypi-url] -->
<!-- [![Docs Status][docs-image]][docs-url] -->

My entry for the AI Music Generation Challenge 2020 at The 2020 Joint Conference on AI
Music Creativity <https://boblsturm.github.io/aimusic2020/>

The model generates a specific sort of folk music called a double jig which exhibits the
following musical features:
* **disclaimer:** I don't know a thing about folk music!
* in 6/8 time
* melody has (loosely) two groups of quavers (8th notes) per bar
* phrases may end in a dotted quaver (three 8th notes)
* ornaments like substituting the second quaver in a triplet for two semiquavers
* probably more things - people disagree: <https://thesession.org/discussions/4231>

# 1. Setup

## 1.1. System
It's recommended to install MuseScore for visualising scores.

## 1.2. Python environment
```bash
conda create -y -n dj-gen python=3.8
conda activate dj-gen
# pip install .
pip install -e ".[dev]"
# TODO: auto install correct pytorch for given platform
pip install torch torchvision
# pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 \
#     -f https://download.pytorch.org/whl/torch_stable.html
```

## 1.3. Get data
```bash
DATA_HOME="data"
RAW_HOME=${DATA_HOME}/raw
DERIVED_HOME=${DATA_HOME}/derived

scripts/dj-gen-download-data ${RAW_HOME}
```

## 1.4. Split abc into tokens
```bash
scripts/dj-gen-tokenize-abc \
    --data-path "${RAW_HOME}/thesession.org/all_tunes.abc" \
    --output-path "${DERIVED_HOME}/thesession.org/tokenized_tunes.txt" \
    --log-level INFO
```

## 1.5. Create the experimental splits
```bash
DATA_HOME=data
DERIVED_HOME=${DATA_HOME}/derived
dj-gen-get-vocab-and-make-splits \
    --data-path ${DERIVED_HOME}/thesession.org/tokenized_tunes.txt \
    --seed 42 \
    --max_tune_length 500 \
    --min_tune_length 60 \
    --test_prop 0.05 \
    --valid_prop 0.05 \
    --train_prop 0.9 \
    --batch_size 256
```

# 2. Dev Setup
Install pre-commit hooks to automatically check code with isort, black, and flake8.

```bash
git clone git@github.com:JamesOwers/double-jig-gen.git
cd double-jig-gen
pre-commit install
```


# 3. Archive

Previous instructions

## 3.1. Configure Music21

Then run the following:
```python
from music21 import configure
configure.run()
```

Also, make sure you open MuseScore and accept the query box first, else you'll get
errors in the notebook e.g.
```bash
dlopen error : dlopen(/usr/local/lib/libjack.0.dylib, 1): image not found
Creating main window…
ZoomBox::setLogicalZoom(): Formatting logical zoom level as 100% (rounded from 1.000000)
Reading translations…
```

```bash
scripts/dj-gen-tokenize-abc \
    --data-path ${RAW_HOME}/folk-rnn/data_v1 \
    --output-path ${DERIVED_HOME}/folk-rnn/clean-folk-rnn.txt \
    --log-level INFO

scripts/dj-gen-tokenize-abc \
    --data-path ${RAW_HOME}/folk-rnn/data_v1 \
    --output-path ${DERIVED_HOME}/folk-rnn/clean-folk-rnn-100.txt \
    --token-separator § \
    --nr-tunes 100 \
    --log-level INFO
scripts/dj-gen-tokenize-abc \
    --data-path ${RAW_HOME}/folk-rnn/data_v1 \
    --output-path ${DERIVED_HOME}/folk-rnn/clean-folk-rnn-1000.txt \
    --token-separator § \
    --nr-tunes 1000 \
    --log-level INFO
scripts/dj-gen-tokenize-abc \
    --data-path ${RAW_HOME}/folk-rnn/data_v1 \
    --output-path ${DERIVED_HOME}/folk-rnn/clean-folk-rnn-10_000.txt \
    --token-separator § \
    --nr-tunes 10000 \
    --log-level INFO

dj-gen-get-vocab-and-make-splits \
    --data-path ${DERIVED_HOME}/folk-rnn/clean-folk-rnn.txt \
    --seed 42 \
    --max_tune_length 500 \
    --min_tune_length 60 \
    --test_prop 0.05 \
    --valid_prop 0.05 \
    --train_prop 0.9 \
    --batch_size 256

dj-gen-get-vocab-and-make-splits \
    --data-path ${DERIVED_HOME}/folk-rnn/clean-folk-rnn-100.txt \
    --seed 42 \
    --max_tune_length 500 \
    --min_tune_length 60 \
    --test_prop 0.05 \
    --valid_prop 0.05 \
    --train_prop 0.9 \
    --batch_size 256
dj-gen-get-vocab-and-make-splits \
    --data-path ${DERIVED_HOME}/folk-rnn/clean-folk-rnn-1000.txt \
    --seed 42 \
    --max_tune_length 500 \
    --min_tune_length 60 \
    --test_prop 0.05 \
    --valid_prop 0.05 \
    --train_prop 0.9 \
    --batch_size 256
dj-gen-get-vocab-and-make-splits \
    --data-path ${DERIVED_HOME}/folk-rnn/clean-folk-rnn-10_000.txt \
    --seed 42 \
    --max_tune_length 500 \
    --min_tune_length 60 \
    --test_prop 0.05 \
    --valid_prop 0.05 \
    --train_prop 0.9 \
    --batch_size 256
```

