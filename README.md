# Titanic Disaster
Kaggle Competition : https://www.kaggle.com/c/titanic

## Preparation
```
mkdir data/
```
Download [all.zip](https://www.kaggle.com/c/3136/download-all) and put files in data/

## Usage

### Hyperparameters in flags.py
`batch_size` : batch size / one training step  
`units` : numbers of neuron of layers  

### Processing data
```
python process.py
```

### Train
```
python main.py --mode train
```

### Test
```
python main.py --mode test [--load (step)]
```
the output file would be data/prediction.csv

## Files

### Folders
`data/` : all data ( {}.csv / train / test )  
`model/` : store models 

### Files
`process.py` : processing training / testing data  
`flags.py` : all setting  
`main.py` : main function  
`utils.py` : get date batch  
`model.py` : model structure  


