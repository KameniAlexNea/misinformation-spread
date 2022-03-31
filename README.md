# misinformation-spread

## Dev

Utilisation de pyenv : https://github.com/pyenv/pyenv

## Version python

3.6, 3.7, 3.8, 3.9, 3.10

## Needed

> pyenv install 3.6.9 3.7.10 3.8.9 3.9.4 3.10a7 \
> python -m venv .venv  \
> . .venv/bin/activate \
> pip install -r requirements.txt

## Load detection dataset

> Download [CoAID](https://github.com/cuilimeng/CoAID)  \
> Add dowloaded folder in dataset/detection \
> run script : `python load_detection_dataset.py` \
> run script : `python create_detection_dataset.py`

This will create json file [here](./Detecting/dataset.json)