# ImageSearchProject

## Instruction

### Install requirements

1. Creating coda environtment:

```
conda create --name py38 python==3.8.16
conda avtive py38
```

3. Install Pytorch-cuda==11.7, following [official instruction](pytorch.org):

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

3. Install the necessary dependencies by running:

pip install -r requirements.txt

### Preparing datasets

1. Downloading Oxford5k datasets in [here](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/).
2. Downloading Paris6k datasets in [here](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)
3. Put the dataset following this structure:

```
ImageSearchProject
│
├── static/
│ ├── oxford
| | ├── feature
| | | ├── LBP.index.bin
| | | ├── Resnet50.index.bin
| | | ├── VGG16.index.bin
| | | ├── RGBHistogram.index.bin
| | | ├── ViT.index.vin
| | ├── evaluation
| | | ├── crop
| | | ├── original
| | ├── groundtruth
| | └── image
| | └── ...
| ├── paris
| | ├── feature
| | | ├── LBP.index.bin
| | | ├── Resnet50.index.bin
| | | ├── VGG16.index.bin
| | | ├── RGBHistogram.index.bin
| | | ├── ViT.index.vin
| | ├── evaluation
| | | ├── crop
| | | ├── original
| | ├── groundtruth
| | ├── image
| | | ├── paris_defense_000000.jpg
| | | ├── paris_defense_000042.jpg
| | | └── ...
└── ...
```

### Running code

1.Indexing (feature extraction)

```
python indexing.py --data_path static/oxford --feature_extractor resnet50
```

The resnet50.index.bin will be at _static/oxford/feature_ 2. Ranking

```
python ranking.py --data_path static/oxford --feature_extractor resnet50 --k 10
```

3. Evaluation

```
python evaluation --data_path static/oxford --feature_extractor resnet50
```

## Running demo

- Using _flask_ to build web
- Running command line below to start:

```
flask run
```

## Reference

```
@article{johnson2019billion,
title={Billion-scale similarity search with {GPUs}},
author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
journal={IEEE Transactions on Big Data},
volume={7},
number={3},
pages={535--547},
year={2019},
publisher={IEEE}
}
```
