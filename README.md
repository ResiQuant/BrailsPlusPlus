<h1 style="text-align: center;">BRAILS++: Building Regional Asset Inventories at Large Scale</h1>

[![Tests](https://github.com/NHERI-SimCenter/BrailsPlusPlus/actions/workflows/tests.yml/badge.svg)](https://github.com/NHERI-SimCenter/BrailsPlusPlus/actions/workflows/tests.yml/badge.svg)
[![Lint Code](https://github.com/NHERI-SimCenter/BrailsPlusPlus/actions/workflows/lint_code.yml/badge.svg)](https://github.com/NHERI-SimCenter/BrailsPlusPlus/actions/workflows/lint_code.yml/badge.svg)
[![DOI](https://zenodo.org/badge/184673734.svg)](https://zenodo.org/badge/latestdoi/184673734)
[![PyPi version](https://badgen.net/pypi/v/BRAILS/)](https://pypi.org/project/BRAILS/)
[![PyPI download month](https://img.shields.io/pypi/dm/BRAILS.svg)](https://pypi.python.org/pypi/BRAILS/)

## What is it?

```BRAILS++``` is an object-oriented framework for building applications that focus on generating asset inventories for large geographic regions.

## How is the repo laid out?

#### :building_construction:UNDER CONSTRUCTION!! :building_construction: 

+ ```brails```: A directory containing the classes
  - ```brails/types```: directory containing useful datatypes, e.g., ```ImageSet``` and ```AssetInventory```
  - ```brails/processors```: directory containing classes that do ```image_processing``` to make predictions, e.g. RoofShape</li>
  - ```brails/segmenters```: directory containing classes that do image segmentation
  - ```brails/scrapers```: directory containing classes that do internet downloads, e.g., footprint scrapers, image scrapers
  - ```brails/filters```: directory containing image filters, e.g., classes that take images and revise or filter out
  - ```brails/imputaters```: directory containing classes that fill in missing ```AssetInventory``` datasets
  - ```brails/utils```: directory containing misc classes that do useful things, e.g. geometric conversions
+ ```examples```: A directory containing examples
+ ```tests```: A directory containing unit tests. The directory structure follows that of ```brails```

## Documentation

You can find the documentation for ```BRAILS++``` [here]().

## Installation instructions

```BRAILS++``` is available on PyPI under the name ```BRAILS```.

```shell
pip install BRAILS
```
Developers and contributors should read the [Contributing Code to Brails]() page of the documentation.


<!-- todo: instructions on how to lint the code, and specific subfolder or file. -->
<!-- todo: example with the test suite. -->
<!-- todo: instructions on how to run the tests -->
<!-- todo: instructions on how to check coverage -->
<!-- python -m pytest tests --cov=brails --cov-report html -->
