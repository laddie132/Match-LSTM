# Match-LSTM

Here we implement the MatchLSTM (Wang and Jiang, 2016) model and R-Net(MSRA, 2017) model on SQuAD (Rajpurkar et al., 2016).

Maybe there are some details different from initial paper.

## Requirements

- python3
- anaconda
- hdf5
- [pytorch 0.4](https://github.com/pytorch/pytorch/tree/v0.4.0)
- [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/)

## Experiments

The Match-LSTM+ model is a little change from Match-LSTM.

- replace LSTM with GRU
- add gated-attention match
- add separated char-level encoding
- add aggregation layer with one GRU layer
- initial GRU first state in pointer-net
    - linear method: add full-connect layer after match layer
    - pooling method: add attention-pooling layer after question encoding

Evaluate results on SQuAD dev set:

model|em|f1
---|---|---|
Match-LSTM+ with linear|**66.94**|**76.20**
Match-LSTM+ with pooling and bp|66.93|76.09
R-NET-45(our version)|64.19|73.62
R-NET(paper)|72.3|80.6

> - 'bp' refers to bidirectional ptr-net
> - 'linear' refers to linear initial pointer-net with FC layer
> - 'pooling' refers to attention-pooling inital pointer-net
> - 'R-NET-45' refers to R-NET with hidden size of 45


## Usage

`python run.py [preprocess/train/test] [-c config_file] [-o ans_path]`

- -c config_file: Defined dataset, model, train methods and so on. Default: `config/global_config.yaml`
- -o ans_path: *see in test step*

> there several models you can choose in `config/global_config.yaml`, like 'match-lstm', 'match-lstm+' and 'r-net'. view and modify. 

### Preprocess

1. Put the GloVe embeddings file to the `data/` directory
2. Put the SQuAD dataset to the `data/` directory
3. Run `python run.py preprocess` to generate hdf5 file of SQuAD dataset

### Train

Run `python run.py train`

### Test

Run `python run.py test [-o ans_file]`

- -o ans_file: Output the answer of question and context with a unique id to ans_file. 

> Note that we use `data/model-weight.pt` as our model weights by default. You can modify the config_file to set model weights file.

### Evaluate

Run `python helper_run/evaluate-v1.1.py [dataset_file] [prediction_file]`

- dataset_file: ground truth of dataset. example: `data/SQuAD/dev-v1.1.json`
- prediction_file: your model predict on dataset. you can use the `ans_file` from test step.

### Analysis

Run `python helper_run/analysis_[*].py`

Here we provide some scipt to analysis your model output, such as `analysis_log.py`, `analysis_ans.py`, `analysis_dataset.py` and so on. view and explore. 

## Reference

- [Wang, Shuohang, and Jing Jiang. "Machine comprehension using match-lstm and answer pointer." arXiv preprint arXiv:1608.07905 (2016).](https://arxiv.org/abs/1608.07905)
- [R-NET: MACHINE READING COMPREHENSION WITH SELF-MATCHING NETWORKS](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)

## License

[MIT](https://github.com/laddie132/MRC-PyTorch/blob/master/LICENSE)

