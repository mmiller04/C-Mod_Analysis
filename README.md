# cmod_thomson_analysis
Routines to fetch and fit TS data from C-Mod MDSplus tree

## Installation

Presently this codebase only works for python 3.8.10 (found on mfews03) and is held together by scotch tape and good intentions.
A more comprehensive look at the packages should happen in the future, but for now we're just getting everyone using the same package versions.

0. Set up .bashrc

`vim ~/.bashrc`
Add the following lines:
```
export POETRY_VIRTUALENVS_IN_PROJCT = 1
export DISPY_DIR=/usr/local/mfe/disruptions/disruption-py
export PATH=$DISPY_DIR/poetry/bin:$PATH
export PYTHONPATH=/usr/local/mdsplus/python
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```
To submit updates
`source ~/.bashrc`

Verify it is installed by doing `poetry --version`. It should say 1.8.3

1. Install repo

```
git clone https://github.com/mmiller04/C-Mod_Analysis.git`
cd C-Mod_Analysis
poetry env use /usr/bin/python
poetry env info # ensure this is /usr/bin/python
python --version # ensure this is 3.8.10
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry install
```

This should automatically install everything.
To activate the environment, `source .venv/bin/activate`
To deactivat the environment, `source deactivate`

2. Add/remove dependency
`poetry add dependency==version`
`poetry 
If you ever run into an issue where poetry is hanging, hold ctrl+c to stop and then hit em with a `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`