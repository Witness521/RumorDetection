import os

import LSTM_Attention.models.TextRNN_Att as TextRNN_Att
import torch
import tqdm
import pickle as pkl
from LSTM_Attention.utils import DatasetIterater

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


'''
    返回模型的注意力机制之后的save_out
'''
def readModelLSTMAtt(config, review):
    model = TextRNN_Att.Model(config=config)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    _, encode = build_dataset(config, review)
    t = (torch.tensor([encode[0][0]]), torch.tensor([123]))
    out = model(t)
    # print(model.save_out)
    return model.save_out


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

# def encodeReview(config, review, ues_word = False, pad_size = 32):
#     pad_size = config.pad_size
#     if ues_word:
#         tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level，英文单词的级别
#     else:
#         tokenizer = lambda x: [y for y in x]  # char-level，中文单个字的级别
#     if os.path.exists(config.vocab_path):
#         vocab = pkl.load(open(config.vocab_path, 'rb'))
#     else:
#         vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
#         pkl.dump(vocab, open(config.vocab_path, 'wb'))
#     print(f"Vocab size: {len(vocab)}")
#     words_line = []
#     token = tokenizer(review)
#     seq_len = len(token)
#     if pad_size:
#         if len(token) < pad_size:
#             token.extend([PAD] * (pad_size - len(token)))
#         else:
#             token = token[:pad_size]
#             seq_len = pad_size
#     # word to id
#     for word in token:
#         words_line.append(vocab.get(word, vocab.get(UNK)))
#     label = 1  # label默认等于1
#     return torch.tensor(words_line)  # [([...], 0), ([...], 1), ...]


def build_dataset(config, review, ues_word = False):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    # print(f"Vocab size: {len(vocab)}")

    def load_dataset(review, pad_size=32):
        contents = []
        words_line = []
        token = tokenizer(review)
        seq_len = len(token)
        if pad_size:
            if len(token) < pad_size:
                token.extend([PAD] * (pad_size - len(token)))
            else:
                token = token[:pad_size]
                seq_len = pad_size
        # word to id
        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))

        contents.append((words_line, int(1), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    encode = load_dataset(review, config.pad_size)
    return vocab, encode

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

if __name__ == '__main__':
    dataset = 'dataSet'
    config = TextRNN_Att.Config(dataset, embedding='embedding_SougouNews.npz')
    review = '我和我的小伙伴们都惊呆了！这种路段警方再找不到逃逸司机。就真的要滚回去种田了！！！！'
    readModelLSTMAtt(config, review)