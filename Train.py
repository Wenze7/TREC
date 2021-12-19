import time
import torch
import torch.nn.functional as F
import numpy as np
import os


def train(args, model, optimizer, Criterion, scheduler, train_loader):
    model_save_path = args.save_path + '/model/model_remove_stop_words'
    model.train()
    start_time = time.time()
    log_steps = 100
    global_step = 0

    for epoch in range(args.epochs):
        print('start training epoch:{}'.format(epoch))
        epoch_start_time = time.time()
        losses = []

        for step, sample in enumerate(train_loader):

            query_input_ids = sample['query_input_ids'].to(args.device, dtype=torch.long)
            query_attention_mask = sample['query_attention_mask'].to(args.device, dtype=torch.long)

            pos_passage_input_ids = sample['pos_passage_input_ids'].to(args.device, dtype=torch.long)
            pos_passage_attention_mask = sample['pos_passage_attention_mask'].to(args.device, dtype=torch.long)

            neg_passage_input_ids = sample['neg_passage_input_ids'].to(args.device, dtype=torch.long)
            neg_passage_attention_mask = sample['neg_passage_attention_mask'].to(args.device, dtype=torch.long)

            query_embedding = model(input_ids=query_input_ids, attention_mask=query_attention_mask)
            pos_passage_embedding = model(input_ids=pos_passage_input_ids, attention_mask=pos_passage_attention_mask)
            neg_passage_embedding = model(input_ids=neg_passage_input_ids, attention_mask=neg_passage_attention_mask)

            pos_score = F.pairwise_distance(query_embedding, pos_passage_embedding, p=1, keepdim=True)
            neg_score = F.pairwise_distance(query_embedding, neg_passage_embedding, p=1, keepdim=True)

            label_y = -torch.ones(pos_score.shape).to(args.device)  # pos_score < neg_score

            loss = Criterion(pos_score, neg_score, label_y)
            losses.append(loss.item())
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % log_steps == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s, lr: %.10f"
                      % (global_step, epoch, step, np.mean(losses), global_step / (time.time() - start_time),
                         float(scheduler.get_last_lr()[0])))

            del label_y
            del query_input_ids
            del query_attention_mask
            del pos_passage_input_ids
            del pos_passage_attention_mask
            del neg_passage_input_ids
            del neg_passage_attention_mask
            del query_embedding
            del pos_passage_embedding
            del neg_passage_embedding

        print('saving model for epoch {}'.format(epoch + 1))
        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        torch.save(model.state_dict(), model_save_path + '/model_epoch{}'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))
        epoch_end_time = time.time()
        print('time for one epoch: {}'.format(epoch_end_time - epoch_start_time))

    print('training finished')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    torch.save(model.state_dict(), model_save_path + '/final_model')
