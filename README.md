# Match-LSTM

Here we implement the MatchLSTM (Wang and Jiang 2016) model, R-Net(Wang et al. 2017) model and M-Reader(Hu et al. 2017) on SQuAD (Rajpurkar et al. 2016).

Maybe there are some details different from initial paper.

## Requirements

- python3
- anaconda
- hdf5
- [spaCy 2.0](https://spacy.io/)
- [pytorch 0.4](https://github.com/pytorch/pytorch/tree/v0.4.0)
- [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/)

## Experiments

The Match-LSTM+ model is a little change from Match-LSTM.

- replace LSTM with GRU
- add gated-attention match like r-net
- add separated char-level encoding
- add additional features like M-Reader
- add aggregation layer with one GRU layer
- initial GRU first state in pointer-net
    - add full-connect layer after match layer

Evaluate results on SQuAD dev set:

model|em|f1
---|---|---|
Match-LSTM+ (our version)|**70.2**|**79.2**
Match-LSTM (paper)|64.1|73.9
R-NET-45 (our version)|64.2|73.6
R-NET (paper)|72.3|80.6
M-Reader (our version)|**70.4**|**79.6**
M-Reader+RL (paper)|72.1|81.6

> 'R-NET-45' refers to R-NET with hidden size of 45

## Usage

```bash
python run.py [preprocess/train/test] [-c config_file] [-o ans_path]
```

- -c config_file: Defined dataset, model, train methods and so on. Default: `config/global_config.yaml`
- -o ans_path: *see in test step*

> there several models you can choose in `config/global_config.yaml`, like 'match-lstm', 'match-lstm+', 'r-net' and 'm-reader'. view and modify. 

### Preprocess

1. Put the GloVe embeddings file to the `data/` directory
2. Put the SQuAD dataset to the `data/` directory
3. Run `python run.py preprocess` to generate hdf5 file of SQuAD dataset

> Note that preprocess will take a long time if multi-features used. Maybe close to an hour.

### Train

```bash
python run.py train
```

### Test

```bash
python run.py test [-o ans_file]
```

- -o ans_file: Output the answer of question and context with a unique id to ans_file. 

> Note that we use `data/model-weight.pt` as our model weights by default. You can modify the config_file to set model weights file.

### Evaluate

```bash
python helper_run/evaluate-v1.1.py [dataset_file] [prediction_file]
```

- dataset_file: ground truth of dataset. example: `data/SQuAD/dev-v1.1.json`
- prediction_file: your model predict on dataset. you can use the `ans_file` from test step.

### Analysis

```bash
python helper_run/analysis_[*].py
```

Here we provide some scipt to analysis your model output, such as `analysis_log.py`, `analysis_ans.py`, `analysis_dataset.py` and so on. view and explore. 

## Reference

- [Wang, Shuohang, and Jing Jiang. "Machine comprehension using match-lstm and answer pointer." arXiv preprint arXiv:1608.07905 (2016).](https://arxiv.org/abs/1608.07905)
- [R-NET: MACHINE READING COMPREHENSION WITH SELF-MATCHING NETWORKS](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)
- [Hu, Minghao, Yuxing Peng, and Xipeng Qiu. "Reinforced mnemonic reader for machine comprehension." CoRR, abs/1705.02798 (2017).](https://arxiv.org/abs/1705.02798)

## License

[MIT](https://github.com/laddie132/MRC-PyTorch/blob/master/LICENSE)

