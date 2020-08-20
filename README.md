# Double Jig Generation

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

Download and install musescore (for visualising ABC with music21)
https://musescore.org/en/download. I then needed to create a shortcut for music21 to
find my copy of MuscScore:
```bash
ln -s /Applications/MuseScore\ 3.5.app /Applications/MuseScore\ 3.app
```

Then run the following:
```python
from music21 import *
confgigure.run()
```

Also, make sure you open MuseScore and accept the query box first, else you'll get
errors in the notebook e.g.
```bash
dlopen error : dlopen(/usr/local/lib/libjack.0.dylib, 1): image not found
Creating main window…
ZoomBox::setLogicalZoom(): Formatting logical zoom level as 100% (rounded from 1.000000)
Reading translations…
```

### Python environment
```bash
conda create -n dj-gen python=3.8 black flake8 isort jupyterlab matplotlib numpy \
    pandas pre-commit pylint pytest seaborn tqdm
conda activate dj-gen
conda install pytorch torchvision -c pytorch
conda install -c conda-forge pre-commit
# conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install music21
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
