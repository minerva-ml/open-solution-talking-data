# TalkingData AdTracking Fraud Detection Challenge: open solution

This is an open solution to the [TalkingData Challenge](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection).

## Goal
Deliver open source, ready-to-use and extendable solution to this competition. This solution should - by itself - establish solid benchmark, as well as provide good base for your custom ideas and experiments.

## Disclaimer
In this open source solution you will find references to the neptune.ml. It is free platform for community Users, which we use daily to keep track of our experiments. Please note that using neptune.ml is not necessary to proceed with this solution. You may run it as plain Python script :wink:.

## Usage: Fast Track
1. clone this repository: `git clone https://github.com/neptune-ml/open-solution-talking-data.git`
1. install requirements
1. register to [Neptune](https://neptune.ml/ 'machine learning lab') *(if you wish to use it)*
1. run experiment:
```bash
$ neptune login
$ neptune experiment send --config neptune.yaml --worker gcp-large --environment base-cpu-py3 main.py train_evaluate_predict --pipeline_name solution_1
```
collect submit from `/output/solution-1` directory.

## Usage: Detailed
1. clone this repository: `git clone https://github.com/neptune-ml/open-solution-talking-data.git`
1. install [PyTorch](http://pytorch.org/) and `torchvision`
1. install requirements: `pip3 install -r requirements.txt`
1. register to [Neptune](https://neptune.ml/ 'machine learning lab') *(if you wish to use it)*
1. open [Neptune](https://neptune.ml/ 'machine learning lab') and create new project called: `talking-data` with project key: `TDAT`
1. run experiment:
```bash
$ neptune login
$ neptune experiment send --config neptune.yaml --worker gcp-large --environment base-cpu-py3 main.py train_evaluate_predict --pipeline_name solution_1
```
collect submit from `/output/solution-1` directory.

## User support
There are several ways to seek help:
1. Kaggle [discussion](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion) is our primary way of communication.
1. You can submit an [issue](https://github.com/neptune-ml/open-solution-talking-data/issues) directly in this repo.

## Contributing
1. Check [CONTRIBUTING](CONTRIBUTING.md) for more information.
1. Check [issues](https://github.com/neptune-ml/open-solution-talking-data/issues) and [project](https://github.com/neptune-ml/open-solution-talking-data/projects/1) to check if there is something you would like to contribute to.
