# MRC-PyTorch

Here we implement the MatchLSTM (Wang and Jiang, 2016) model and R-Net(MSRA, 2017) model on SQuAD (Rajpurkar et al., 2016).

Maybe there are some details different from initial paper.

## Requirements

- python3
- anaconda
- [pytorch 0.4](https://github.com/pytorch/pytorch/tree/v0.4.0)
- [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/)

## Experiments

We have trained on Match-LSTM and R-NET model, but both are not as good as the paper given. The best model is Match-LSTM+ 
that a little change from Match-LSTM. 

Here are some changes on Match-LSTM with boundary+search methods.
- replace LSTM with GRU
- add gated-attention match
- add separated char-level encoding
- add aggregation layer with one GRU layer
- initial GRU first state in pointer-net
    - add full-connect layer after match layer
    - or add attention-pooling layer after question encoding

Evaluate results on SQuAD dev set:

model|em|f1
---|---|---|
Match-LSTM+ with linear|66.72|76.05
Match-LSTM+ with pooling and bp|**66.93**|**76.09**
R-NET-45(our version)|64.19|73.62
R-NET(paper)|72.3|80.6

> - 'bp' refers to bidirectional ptr-net
> - 'linear' refers to linear initial pointer-net with FC layer
> - 'pooling' refers to attention-pooling inital pointer-net
> - 'R-NET-45' refers to R-NET with hidden size of 45


## Usage

### Preprocess

1. Put the GloVe embeddings file(*you have downloaded before*) to the `data/` directory
2. Run `python helper_run/preprocess.py` to generate hdf5 file of SQuAD dataset

### Train

Run `python train.py [-c config_file]`.

- -c config_file: Defined model hyperparameters. Default: `config/model_config.yaml`

> Note that there are some config templates you can choose in directory `config/`, such as `config/match-lstm.yaml`, `config/r-net.yaml`, and so on. You can also try to modify `config/model_config.yaml` for default arguments.

### Test

Run `python test.py [-c config_file] [-o ans_file]`.

- -c config_file: Defined model hyperparameters. Default: `config/model_config.yaml`
- -o ans_file: Output the answer of question and context with a unique id to ans_file. Default: `None`, means no write file and just calculate the score of em and f1(not same with standard score).

> Note that we use `data/model-weight.pt` as our model weights by default. You can modify the config_file to set model weights file.

### Evaluate

Run `python helper_run/evaluate-v1.1.py [dataset_file] [prediction_file]` to get standard score of em and f1.

- dataset_file: ground truth of dataset. example: `data/SQuAD/dev-v1.1.json`
- prediction_file: your model predict on dataset. you can use the `ans_file` from test step.

### Analysis

Run `python helper_run/analysis_[*].py`.

Here we provide some scipt to analysis your model output, such as `analysis_log.py`, `analysis_ans.py`, `analysis_dataset.py` and so on. Please read the scipt first to know how to use it or what it does.

## Reference

- [Wang, Shuohang, and Jing Jiang. "Machine comprehension using match-lstm and answer pointer." arXiv preprint arXiv:1608.07905 (2016).](https://arxiv.org/abs/1608.07905)
- [R-NET: MACHINE READING COMPREHENSION WITH SELF-MATCHING NETWORKS](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)

## License

[MIT](https://github.com/laddie132/MRC-PyTorch/blob/master/LICENSE)

