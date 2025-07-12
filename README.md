# STHDformer: A Novel Dual-Branch Transformer for Traffic Flow Forecasting

## Description

Code for the paper "STHDformer: A Novel Dual-Branch Transformer for Traffic Flow Forecasting". 

## STHDformer



## Results



## Train

Run the following command to train a model on PEMS04 dataset:

```bash
python model/train.py -d PEMS04
```

To ensure that the program can be executed, you must run the python files create_spatial_semantics.py and create_traffic_pattern.py to create temporal pattern difference matrix and spatial semantic difference matrix

```bash
python model/utils/create_spatial_semantics.py or python model/utils/create_traffic_pattern.py
```

