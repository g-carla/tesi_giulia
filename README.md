# tesi
questa è la mia tesi



## Installation

Create an environment and install main packages from conda (needed only once)

```
conda create -n tesi python=3.6 numpy scipy matplotlib astropy ipython
```

```
conda install -n tesi -c astropy photutils ccdproc
```


Activate the environment

```
source activate tesi
```

Then, from the tesi folder (where setup.py is)

```
pip install -e .
```


### Don't use pip if conda is available

As a general approach, if a module is available in conda install it from there, not from pip. [How to do it automatically for dependencies in setup.py?]

For instance, use 

```
conda install -c astropy photutils ccdproc
```

instead of 

```
pip install photutils ccdproc
```

### Update all packages

```
conda update -n tesi --all
```


