#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import torch
import logging
from dataset.squad_dataset import SquadDataset
from models.match_lstm import MatchLSTMModel
from utils.load_config import init_logging, read_config
from utils.utils import MyNLLLoss
from utils.eval import eval_on_model

init_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info('------------Match-LSTM Evaluate--------------')
    logger.info('loading config file...')
    global_config = read_config()

    # set random seed
    seed = global_config['model']['random_seed']
    enable_cuda = global_config['test']['enable_cuda']
    torch.manual_seed(seed)

    if torch.cuda.is_available() and not enable_cuda:
        logger.warning("CUDA is avaliable, you can enable CUDA in config file")
    elif not torch.cuda.is_available() and enable_cuda:
        logger.error("CUDA is not abaliable, please unable CUDA in config file")
        exit(-1)

    logger.info('reading squad dataset...')
    dataset = SquadDataset(global_config)

    logger.info('constructing model...')
    model = MatchLSTMModel(global_config)
    if enable_cuda:
        model = model.cuda()
    model.eval()        # let training = False, make sure right dropout

    # load model weight
    logger.info('loading model weight...')
    model_weight_path = global_config['data']['model_path']
    is_exist_model_weight = os.path.exists(model_weight_path)
    if not is_exist_model_weight:
        logger.info("not found model weight file on '%s'" % model_weight_path)
        exit(-1)

    weight = torch.load(model_weight_path, map_location=lambda storage, loc: storage)
    if enable_cuda:
        weight = torch.load(model_weight_path, map_location=lambda storage, loc: storage.cuda())
    model.load_state_dict(weight, strict=False)

    # forward
    logger.info('forwarding...')

    batch_size = global_config['test']['batch_size']
    batch_dev_data = dataset.get_dataloader_dev(batch_size)

    criterion = MyNLLLoss()
    score_em, score_f1 = eval_on_model(model=model,
                                       criterion=criterion,
                                       batch_data=batch_dev_data,
                                       epoch=None,
                                       enable_cuda=enable_cuda,
                                       func=dataset.sentence_id2word)
    logger.info("test: ave_score_em=%.2f, ave_score_f1=%.2f" % (score_em, score_f1))
    logging.info('finished.')


if __name__ == '__main__':
    main()