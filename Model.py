from transformers import BertModel
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(self.args.bert_path)
        self.out_Linear = nn.Linear(768, self.args.output_dim)
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        out = self.dropout(self.out_Linear(bert_out))

        return out

