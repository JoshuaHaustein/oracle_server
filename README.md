# Installation

## Clone and create virtual environment
```
git clone <repo>
cd <repo-dir>
virtualenv -p python3.5 venv
source venv/bin/activate
```

## PyTorch
PyTorch is not on PyPi, so this needs to be installed separately:
```
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl"
```

## Install this package
```
pip install .
```

## Setup communication pipes and environment variables
This can be run several times if the variables needs setting multiple times
(needed also on client side).
```
source setup.sh
```

## Start server
For now, you have to launch the server from the repo directory in order for the server to
find the trained models.
```
python python_src/server.py
```
