#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import json
import os
import torch
import logging
import argparse
from dataset.squad_dataset import SquadDataset
from models import *
from utils.load_config import init_logging, read_config
from models.loss import MyNLLLoss
from utils.eval import eval_on_model

logger = logging.getLogger(__name__)


def test(config_path, out_path):
    logger.info('------------MODEL PREDICT--------------')
    logger.info('loading config file...')
    global_config = read_config(config_path)

    # set random seed
    seed = global_config['global']['random_seed']
    torch.manual_seed(seed)

    enable_cuda = global_config['test']['enable_cuda']
    device = torch.device("cuda" if enable_cuda else "cpu")
    if torch.cuda.is_available() and not enable_cuda:
        logger.warning("CUDA is avaliable, you can enable CUDA in config file")
    elif not torch.cuda.is_available() and enable_cuda:
        raise ValueError("CUDA is not abaliable, please unable CUDA in config file")

    torch.set_grad_enabled(False)  # make sure all tensors below have require_grad=False,

    logger.info('reading squad dataset...')
    dataset = SquadDataset(global_config)

    logger.info('constructing model...')
    model_choose = global_config['global']['model']
    dataset_h5_path = global_config['data']['dataset_h5']
    if model_choose == 'base':
        model_config = read_config('config/base_model.yaml')
        model = BaseModel(dataset_h5_path,
                          model_config)
    elif model_choose == 'match-lstm':
        model = MatchLSTM(dataset_h5_path)
    elif model_choose == 'match-lstm+':
        model = MatchLSTMPlus(dataset_h5_path)
    elif model_choose == 'r-net':
        model = RNet(dataset_h5_path)
    elif model_choose == 'm-reader':
        model = MReader(dataset_h5_path)
    else:
        raise ValueError('model "%s" in config file not recoginized' % model_choose)

    model = model.to(device)
    model.eval()  # let training = False, make sure right dropout

    # load model weight
    logger.info('loading model weight...')
    model_weight_path = global_config['data']['model_path']
    assert os.path.exists(model_weight_path), "not found model weight file on '%s'" % model_weight_path

    weight = torch.load(model_weight_path, map_location=lambda storage, loc: storage)
    if enable_cuda:
        weight = torch.load(model_weight_path, map_location=lambda storage, loc: storage.cuda())
    model.load_state_dict(weight, strict=False)

    # forward
    logger.info('forwarding...')

    batch_size = global_config['test']['batch_size']

    num_workers = global_config['global']['num_data_workers']
    batch_dev_data = dataset.get_dataloader_dev(batch_size, num_workers)

    # to just evaluate score or write answer to file
    if out_path is None:
        criterion = MyNLLLoss()
        score_em, score_f1, sum_loss = eval_on_model(model=model,
                                                     criterion=criterion,
                                                     batch_data=batch_dev_data,
                                                     epoch=None,
                                                     device=device)
        logger.info("test: ave_score_em=%.2f, ave_score_f1=%.2f, sum_loss=%.5f" % (score_em, score_f1, sum_loss))
    else:
        context_right_space = dataset.get_all_ct_right_space_dev()
        predict_ans = predict_on_model(model=model,
                                       batch_data=batch_dev_data,
                                       device=device,
                                       id_to_word_func=dataset.sentence_id2word,
                                       right_space=context_right_space)
        samples_id = dataset.get_all_samples_id_dev()
        ans_with_id = dict(zip(samples_id, predict_ans))

        logging.info('writing predict answer to file %s' % out_path)
        with open(out_path, 'w') as f:
            json.dump(ans_with_id, f)

    logging.info('finished.')


def predict_on_model(model, batch_data, device, id_to_word_func, right_space):
    batch_cnt = len(batch_data)
    answer = []

    cnt = 0
    for bnum, batch in enumerate(batch_data):
        batch = [x.to(device) if x is not None else x for x in batch]
        bat_context = batch[0]
        bat_answer_range = batch[-1]

        # forward
        batch_input = batch[:len(batch) - 1]
        _, tmp_ans_range, _ = model.forward(*batch_input)

        tmp_context_ans = zip(bat_context.cpu().data.numpy(),
                              tmp_ans_range.cpu().data.numpy())

        # generate initial answer text
        i = 0
        for c, a in tmp_context_ans:
            cur_no = cnt + i
            tmp_ans = id_to_word_func(c[a[0]:(a[1] + 1)])
            cur_space = right_space[cur_no][a[0]:(a[1]+1)]

            cur_ans = ''
            for j, word in enumerate(tmp_ans):
                cur_ans += word
                if cur_space[j]:
                    cur_ans += ' '
            answer.append(cur_ans.strip())
            i += 1
        cnt += i
        logging.info('batch=%d/%d' % (bnum, batch_cnt))

        # manual release memory, todo: really effect?
        del bat_context, bat_answer_range, batch, batch_input
        del tmp_ans_range
        # torch.cuda.empty_cache()

    return answer


if __name__ == '__main__':
    init_logging()

    parser = argparse.ArgumentParser(description="evaluate on the model")
    parser.add_argument('--config', '-c', required=False, dest='config_path', default='config/global_config.yaml')
    parser.add_argument('--output', '-o', required=False, dest='out_path')
    args = parser.parse_args()

    test(config_path=args.config_path, out_path=args.out_path)
