[build-image]: https://travis-ci.com/JamesOwers/double-jig-gen.svg?branch=master
[build-url]: https://travis-ci.com/JamesOwers/double-jig-gen
[coverage-image]: https://codecov.io/gh/JamesOwers/double-jig-gen/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/JamesOwers/double-jig-gen?branch=master
<!-- [docs-image]: https://readthedocs.org/projects/midi_degradation_toolkit/badge/?version=latest
[docs-url]: https://midi_degradation_toolkit.readthedocs.io/en/latest/?badge=latest
[pypi-image]: https://badge.fury.io/py/midi_degradation_toolkit.svg
[pypi-url]: https://pypi.python.org/pypi/midi_degradation_toolkit -->

# `double_jig_gen` - Double Jig Folk Music Generation

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

## Setup

### System
```bash
brew install wget
```

### Python environment
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

### Configure Music21

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

### Get data
```bash
wget http://www.norbeck.nu/abc/book/oneills/1001/DoubleJig0001-0365.abc -P data/
```

## Dev Setup
Install pre-commit hooks to automatically check code with isort, black, and flake8.

```bash
git clone git@github.com:JamesOwers/double-jig-gen.git
cd double-jig-gen
pre-commit install
```

