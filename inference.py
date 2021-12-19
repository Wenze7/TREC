import argparse
import torch
from Model import Model
import torch.nn as nn
import torch.nn.functional as F
from DataLoader import DataLoader, create_dataloader
import pickle
from tqdm import tqdm
import functools
import pandas as pd
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./save', help='保存路径')
    parser.add_argument('--bert_path', type=str, default='./bert/bert-base-uncased', help='bert的路径，这里为本地')
    parser.add_argument('--batch_size', type=int, default=256, help='mini_batch_size')
    parser.add_argument('--max_length_query', type=int, default=128, help='句子长度')
    parser.add_argument('--max_length_passage', type=int, default=512, help='句子长度')
    parser.add_argument('--output_dim', type=int, default=300, help='输出层大小')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--load_model_epoch', type=int, default=5, help='读取第几个epoch的模型')
    parser.add_argument('--eval', type=str, default='test', help='valid or test')

    return parser.parse_args()


def inference(args, model, data_loader):
    results = []
    model.eval()
    with torch.no_grad():
        for sample in tqdm(data_loader):
            query_id = sample['query_id']
            passage_id = sample['passage_id']
            query_input_ids = sample['query_input_ids'].to(args.device, dtype=torch.long)
            query_attention_mask = sample['query_attention_mask'].to(args.device, dtype=torch.long)

            passage_input_ids = sample['passage_input_ids'].to(args.device, dtype=torch.long)
            passage_attention_mask = sample['passage_attention_mask'].to(args.device, dtype=torch.long)

            query_embedding = model(input_ids=query_input_ids, attention_mask=query_attention_mask)
            passage_embedding = model(input_ids=passage_input_ids, attention_mask=passage_attention_mask)

            score = F.pairwise_distance(query_embedding, passage_embedding, p=2, keepdim=True)
            score = score.squeeze()
            score = score.tolist()
            query_id = query_id.tolist()
            passage_id.tolist()

            size = len(query_id)

            results.extend([(query_id[i], passage_id[i].item(), score[i]) for i in range(size)])

        pickle.dump(results, open('./save/eval/results.pkl', 'wb'))


def sort_rule(tuple1, tuple2):
    if tuple1[0] == tuple2[0]:
        if tuple1[2] < tuple2[2]:
            return -1
        elif tuple1[2] > tuple2[2]:
            return 1
        return 0
    if tuple1[0] < tuple2[0]:
        return -1
    elif tuple1[0] > tuple2[0]:
        return 1
    return 0


def get_last_results():
    results = pickle.load(open('./save/eval/results.pkl', 'rb'))
    sorted_results = sorted(results, key=functools.cmp_to_key(sort_rule))
    rank = 1
    for i in range(len(results)):
        if i > 0 and sorted_results[i][0] != sorted_results[i-1][0]:
            rank = 1
        a, b, c = sorted_results[i]
        sorted_results[i] = (a, 'Q0', b, rank, -c, 'BERT')
        rank += 1

    sorted_results = np.array(sorted_results)
    fo = open('./dataset/res.txt', 'w')
    for item in sorted_results:
        fo.writelines('{}\t{}\t{}\t{}\t{}\t{}\n'.format(item[0], item[1], item[2], item[3], item[4], item[5]))
    fo.close()


if __name__ == '__main__':
    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(args)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('./save/model/model_epoch{}/model'.format(args.load_model_epoch)))
    model = model.to(args.device)

    dataset = DataLoader(args, args.eval)
    data_loader = create_dataloader(dataset, args.batch_size)
    inference(args, model, data_loader)
    get_last_results()



