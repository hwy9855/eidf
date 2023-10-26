import argparse
from model import SBA
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
import torch
import random
from tqdm import tqdm
import numpy as np

def load_raw_data(path):
    raw_data = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            raw_data.append(line.split('\n')[0])
    return raw_data


def load_data(raw_data, batch_size=128, shuffle=False):
    if shuffle:
        random.shuffle(raw_data)
    data = []
    data_batch = []
    for data_piece in raw_data:
        data_batch.append(data_piece)
        if len(data_batch) == batch_size:
            data.append(data_batch)
            data_batch = []
    if len(data_batch) > 0:
        data.append(data_batch)
    return data

def main(args):
    random.seed(42)
    torch.manual_seed(42)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    sba_config = RobertaConfig.from_pretrained('roberta-base')
    sba_config.is_decoder = True
    sba_config.add_cross_attention = True
    sba_config.num_hidden_layers = args.layers
    sba = SBA(sba_config).cuda()
    pretrained_LM = RobertaModel.from_pretrained('roberta-base').half().cuda()

    optimizer = torch.optim.Adam(sba.parameters(), lr=args.lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, start_factor=0.1, total_iters=Warmup)
    for params in sba.named_parameters():
        print(params[0])

    train_data_raw = load_raw_data('../rag/msmarco/train.target')
    val_data_raw = load_raw_data('../rag/msmarco/val.target')

    curr_patience = 0
    curr_best_val_loss = 1e7
    for epoch in range(args.epochs):
        sba.train()
        train_data = load_data(train_data_raw, batch_size=args.batch_size)
        losses = []
        for batch in tqdm(train_data, 'Epoch#{}'.format(epoch)):
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            input_ids = inputs['input_ids'].cuda()
            attention_mask = inputs['attention_mask'].cuda()
            with torch.no_grad():
                hidden_states = pretrained_LM(input_ids=input_ids, attention_mask=attention_mask)[0]
            hidden_states = hidden_states.clone().detach()
            loss = sba(hidden_states, input_ids, is_train=True)[1]['loss']
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            # lr_scheduler.step()
        print('Finish Epoch#{} with loss {}.'.format(epoch, np.mean(losses)))
        torch.save(sba.state_dict(), 'ckpts_fp16/last.model')

        val_data = load_data(val_data_raw, batch_size=args.batch_size)
        losses = []

        with torch.no_grad():
            for batch in tqdm(val_data, 'Validation: Epoch#{}'.format(epoch)):
                inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
                input_ids = inputs['input_ids'].cuda()
                attention_mask = inputs['attention_mask'].cuda()
                hidden_states = pretrained_LM(input_ids=input_ids, attention_mask=attention_mask)[0]
                loss = sba(hidden_states, input_ids, is_train=True)[1]['loss']
                losses.append(loss.item())
            val_loss = np.mean(losses)
            if val_loss < curr_best_val_loss:
                print('Epoch#{} validation loss {}, beat previous best loss {}.'.format(
                    epoch, val_loss, curr_best_val_loss))
                curr_best_val_loss = val_loss
                print('Saving as best model to ckpts_fp16/best.model')
                torch.save(sba.state_dict(), 'ckpts_fp16/best.model')
                curr_patience = 0
            else:
                print('Epoch#{} validation loss {}, did not beat previous best loss {}.'.format(
                    epoch, val_loss, curr_best_val_loss))
                curr_patience += 1
                print('Current early stop patience {}/{}.'.format(curr_patience, args.early_stop_patience))
                if curr_patience == args.early_stop_patience:
                    print('Early stop!')
                    break

    print('Finish training!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=15
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=1
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )

    args = parser.parse_args()

    main(args)