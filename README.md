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

### Python environment
```bash
conda create -n dj-gen python=3.8 black flake8 isort jupyterlab matplotlib numpy \
    pandas pre-commit pylint pytest seaborn tqdm
conda activate dj-gen
conda install pytorch torchvision -c pytorch
conda install -c conda-forge pre-commit
# conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
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
