from torch.utils.data import DataLoader as DL
from torch.utils.data.dataset import Dataset
from transformers import BertTokenizer
import pickle


class DataLoader(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_path)
        self.mode = mode
        if mode == 'train':
            print('start train...')
            self.id2queries, self.id2passages, self.triples = pickle.load(open(args.save_path+'/train_data.pkl', 'rb'))
            print('queries:{},passages:{},triples:{}'
                  .format(len(self.id2queries), len(self.id2passages), len(self.triples)))
        elif mode == 'valid':
            print('start valid...')
            self.id2passages, self.id2queries, self.tuples = pickle.load(
                open(args.save_path + '/valid_data.pkl', 'rb'))
            print('queries:{},passages:{},tuples:{}'
                  .format(len(self.id2queries), len(self.id2passages), len(self.tuples)))
        else:
            print('start test...')
            self.id2passages, self.id2queries, self.tuples = pickle.load(
                open(args.save_path + '/test.pkl', 'rb'))
            print('queries:{},passages:{},tuples:{}'
                  .format(len(self.id2queries), len(self.id2passages), len(self.tuples)))

    def __getitem__(self, index):
        if self.mode == 'train':
            query_id, pos_passage_id, neg_passage_id = self.triples[index]
            query = self.id2queries[query_id]
            pos_passage = self.id2passages[pos_passage_id]
            neg_passage = self.id2passages[neg_passage_id]

            query_encoding = self.tokenizer(query,
                                            add_special_tokens=True,
                                            max_length=self.args.max_length_query,
                                            return_token_type_ids=True,
                                            pad_to_max_length=True,
                                            return_attention_mask=True,
                                            return_tensors='pt'
                                            )

            pos_passage_encoding = self.tokenizer(pos_passage,
                                                  add_special_tokens=True,
                                                  max_length=self.args.max_length_passage,
                                                  return_token_type_ids=True,
                                                  pad_to_max_length=True,
                                                  return_attention_mask=True,
                                                  return_tensors='pt'
                                                  )

            neg_passage_encoding = self.tokenizer(neg_passage,
                                                  add_special_tokens=True,
                                                  max_length=self.args.max_length_passage,
                                                  return_token_type_ids=True,
                                                  pad_to_max_length=True,
                                                  return_attention_mask=True,
                                                  return_tensors='pt'
                                                  )

            sample = {
                'query_id': query_id,
                'pos_passage_id': pos_passage_id,
                'neg_passage_id': neg_passage_id,
                'query_input_ids': query_encoding['input_ids'].flatten(),
                'query_attention_mask': query_encoding['attention_mask'].flatten(),
                'pos_passage_input_ids': pos_passage_encoding['input_ids'].flatten(),
                'pos_passage_attention_mask': pos_passage_encoding['attention_mask'].flatten(),
                'neg_passage_input_ids': neg_passage_encoding['input_ids'].flatten(),
                'neg_passage_attention_mask': neg_passage_encoding['attention_mask'].flatten(),

            }

        else:
            query_id, passage_id = self.tuples[index]
            query = self.id2queries[query_id]
            passage = self.id2passages[passage_id]
            query_encoding = self.tokenizer(query,
                                            add_special_tokens=True,
                                            max_length=self.args.max_length_query,
                                            return_token_type_ids=True,
                                            pad_to_max_length=True,
                                            return_attention_mask=True,
                                            return_tensors='pt'
                                            )

            passage_encoding = self.tokenizer(passage,
                                              add_special_tokens=True,
                                              max_length=self.args.max_length_passage,
                                              return_token_type_ids=True,
                                              pad_to_max_length=True,
                                              return_attention_mask=True,
                                              return_tensors='pt'
                                              )
            sample = {
                'query_id': query_id,
                'passage_id': passage_id,
                'query_input_ids': query_encoding['input_ids'].flatten(),
                'query_attention_mask': query_encoding['attention_mask'].flatten(),
                'passage_input_ids': passage_encoding['input_ids'].flatten(),
                'passage_attention_mask': passage_encoding['attention_mask'].flatten(),
            }

        return sample

    def __len__(self):
        if self.mode == 'train':
            return len(self.triples)
        return len(self.tuples)


def create_dataloader(dataset, batch_size):
    data_loader = DL(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

