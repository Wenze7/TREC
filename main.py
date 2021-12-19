from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import argparse
from torch import nn
from Model import Model
from DataLoader import DataLoader, create_dataloader
from Train import train
from DataProcesser import read_data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_passages_path', type=str,
                        default='./dataset/collection.train.sampled.tsv',
                        help='训练文章路径（tid, passage）')

    parser.add_argument('--train_queries_path', type=str,
                        default='./dataset/queries.train.sampled.tsv',
                        help='训练查询路径 (qid, query)')

    parser.add_argument('--train_triples_path', type=str,
                        default='./dataset/qidpidtriples.train.sampled.tsv',
                        help='训练三元组路径 (qid, pos_pid, neg_pid)')

    parser.add_argument('--valid_top_file_path', type=str,
                        default='./dataset/msmarco-passagetest2019-43-top1000.tsv',
                        help='验证集文件路径 (qid, pid, query, passage)')

    parser.add_argument('--valid_label_path', type=str,
                        default='./dataset/2019qrels-pass.txt',
                        help='验证集标签路径 (qid, q0, pid, rating)')

    parser.add_argument('--test_top_file_path', type=str,
                        default='./dataset/msmarco-passagetest2020-54-top1000.tsv',
                        help='测试集文件路径 (qid, pid, query, passage)')

    parser.add_argument('--test_label_path', type=str,
                        default='./dataset/2020qrels-pass.txt',
                        help='测试集标签路径 (qid, 0, pid, rating)')

    parser.add_argument('--save_path', type=str, default='./save', help='保存路径')

    parser.add_argument('--bert_path', type=str, default='./bert/bert-base-uncased', help='bert的路径，这里为本地')
    parser.add_argument('--batch_size', type=int, default=8, help='mini_batch_size')
    parser.add_argument('--max_length_query', type=int, default=128, help='句子长度')
    parser.add_argument('--max_length_passage', type=int, default=512, help='句子长度')
    parser.add_argument('--warm_up_ratio', type=float, default=0.0, help='warm_up')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='L2 norm')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--output_dim', type=int, default=300, help='输出层大小')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--margin', type=float, default=1, help='loss间隔')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮次')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #read_data(args)
    Criterion = nn.MarginRankingLoss(args.margin, reduction='mean')
    model = Model(args)
    model = nn.DataParallel(model)
    model = model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    trainset = DataLoader(args, 'train')
    train_loader = create_dataloader(trainset, args.batch_size)

    total_steps = len(train_loader) * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warm_up_ratio * total_steps,
        num_training_steps=total_steps
    )

    train(args, model, optimizer, Criterion, scheduler, train_loader)

