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
`dp` : keep rate  
`units` : numbers of neuron of layers  

### Processing data
```
python process.py
```

### Train
for DNN (76%):
```
python main.py --mode train
```

for SVC (78%):
```
python ml.py
```
this will output directly data/prediction.csv.  

### Test
for DNN output:
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


