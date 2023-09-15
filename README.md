# DANCER
Dancer: Domain Adaptation and Neurosymbolic inference for Complex Event Reasoning

## Introduction

TBD

## Overview

TBD - add figures and some demo stuff


## Usage

To use this repo, please clone with the necessary submodules:
```
git clone https://github.com/nesl/DANCER.git --recurse-submodules
```
Then please install all the necessary packages via
```
pip install -r requirements.txt
```

Please follow the instructions for usage depending on whether you are planning to run this system over multiple machines (i.e. with a physically separate DANCER client and coordinator) or all within the same machine.

### For single machine usage




### For client-server usage




## Acknowledgements

The DANCER client uses stripped down versions of both [ByteTrack](https://github.com/ifzhang/ByteTrack) and [Yolov5 detection](https://github.com/ultralytics/yolov5).



## Links

Datasets used in this paper:
https://github.com/nesl/ComplexEventDatasets

## Todos

- Add domain adapt
- Fix coordinator/local_analysis.py
- retest detection pipeline
