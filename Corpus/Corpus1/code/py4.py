import numpy as np
import os

PUNC_TOKENS = ['，', '。', '？', '！']
POEM_TEMP_STARTS = ['[JJ5]', '[JJ7]', '[LS5]', '[LS7]', '[OT0]']

TEMP_WEIGHT = 0.5
CONTENT_WEIGHT = 1.0  # Same for TANG POEMS and SONG CI

# For TANG POEMS
P_PRE_PUNC_WEIGHT = 1.1  # Use 1.1 when training the Tang poems only
P_PUNC_WEIGHT = 1.2      # Use 1.2 when training the Tang poems only
P_END_WEIGHT = 1.4       # Use 1.4 when training the Tang poems only

# P_PRE_PUNC_WEIGHT = 1.0  # Use 1.1 when training the Tang poems only
# P_PUNC_WEIGHT = 1.0      # Use 1.2 when training the Tang poems only
# P_END_WEIGHT = 1.0       # Use 1.4 when training the Tang poems only

# For SONG CI
C_PRE_PUNC_WEIGHT = 1.0
C_PUNC_WEIGHT = 1.0
C_END_WEIGHT = 1.2


def cut_poem_line(line):
    out = []
    tmp = ''
    for char in list(line.upper()):
        tmp = tmp.strip()
        if ord(char) > 127:
            if len(tmp) > 0:
                out.append(tmp)
                tmp = ''
            out.append(char)
        elif 48 <= ord(char) <= 57 or 65 <= ord(char) <= 90 or char == ']':
            # 0 - 9 or A - Z or [ or ]
            tmp += char
        elif char == '[':
            if len(tmp) > 0:
                out.append(tmp)
            tmp = char
        # else skipped

    if len(tmp) > 0:
        out.append(tmp)

    return out


def load_poems(corpus_dir, start_token, end_token, poems_file, check_end_sign=False):
    poem_file = os.path.join(corpus_dir, poems_file)
    poems = []
    with open(poem_file, "r", encoding='utf-8') as f:
        for line in f:
            content = line.strip()
            if len(content) < 17:
                print(content)
                continue
            if check_end_sign and content[-1] not in ['。', '？', '！']:
                print(content)
                continue

            content = start_token + content + end_token
            poems.append(content)

    return poems


def load_vocab(data_dir, vocab_file):
    word_int_table, words = {}, []

    vocab_file = os.path.join(data_dir, vocab_file)
    with open(vocab_file, "r", encoding='utf-8') as f:
        idx = 0
        for line in f:
            ln = line.strip()
            if ln:
                words.append(ln)
                word_int_table[ln] = idx
                idx += 1

    words.append(' ')  # Add a space character to the vocabulary
    word_int_table[' '] = len(words) - 1

    return word_int_table, words


def load_poem_yun(data_dir, init_yun_file='poem_init_yun.txt', other_yun_file='poem_other_yun.txt'):
    init_yun_dict, other_yun_dict, yun_list, weight_list = {}, {}, [], []

    init_file = os.path.join(data_dir, init_yun_file)
    with open(init_file, "r", encoding='utf-8') as f:
        for line in f:
            ln = line.strip()
            if ln:
                yun = ln[:5]
                hanzi_list = list(ln[5:])
                init_yun_dict[yun] = hanzi_list

    other_file = os.path.join(data_dir, other_yun_file)
    cnt = 0
    cnt_list = []
    with open(other_file, "r", encoding='utf-8') as f:
        for line in f:
            ln = line.strip()
            if ln:
                yun = ln[:5]
                hanzi_list = list(ln[5:])
                other_yun_dict[yun] = hanzi_list
                yun_list.append(yun)
                this_cnt = len(hanzi_list)
                cnt_list.append(this_cnt)
                cnt += this_cnt

    for c in cnt_list:
        weight_list.append(c / (cnt * 1.0))

    return init_yun_dict, other_yun_dict, yun_list, weight_list


def load_songci_yun(data_dir, init_yun_file='songci_init_yun.txt', other_yun_file='songci_other_yun.txt'):
    init_yun_dict, other_yun_dict, yun_list, p_yun_list, z_yun_list, weight_list = {}, {}, [], [], [], []

    init_file = os.path.join(data_dir, init_yun_file)
    with open(init_file, "r", encoding='utf-8') as f:
        for line in f:
            ln = line.strip()
            if ln:
                yun = ln[:5]
                hanzi_list = list(ln[5:])
                init_yun_dict[yun] = hanzi_list

    other_file = os.path.join(data_dir, other_yun_file)
    cnt = 0
    cnt_list = []
    with open(other_file, "r", encoding='utf-8') as f:
        for line in f:
            ln = line.strip()
            if ln:
                yun = ln[:5]
                hanzi_list = list(ln[5:])
                other_yun_dict[yun] = hanzi_list
                yun_list.append(yun)
                if yun[1] == 'P':
                    p_yun_list.append(yun)
                else:
                    z_yun_list.append(yun)
                this_cnt = len(hanzi_list)
                cnt_list.append(this_cnt)
                cnt += this_cnt

    for c in cnt_list:
        weight_list.append(c / (cnt * 1.0))

    return init_yun_dict, other_yun_dict, yun_list, p_yun_list, z_yun_list, weight_list


def process_poems(data_dir, start_token, end_token, poems_file, vocab_file):
    corpus_dir = os.path.join(data_dir, 'Corpus')
    poems = load_poems(corpus_dir, start_token, end_token, poems_file)
    print("## {} of poems will be used for training.".format(len(poems)))

    word_int_table, words = load_vocab(data_dir, vocab_file)
    vocab_size = len(words)
    poems_id_list = [list(map(lambda word: word_int_table.get(word, vocab_size), cut_poem_line(poem)))
                     for poem in poems]

    return poems_id_list, word_int_table, words


class BatchGenerator(object):
    """
    Common class for both TANG poems and SONG CI
    """
    def __init__(self, batch_size, word_int_table, end_token_id):
        self.batch_size = batch_size
        self.punc_token_ids = []
        self.poem_temp_ids = []
        self.end_token_id = end_token_id
        self.space_token_id = word_int_table.get(' ')

        for t in PUNC_TOKENS:
            id = word_int_table.get(t)
            self.punc_token_ids.append(id)

        for t in POEM_TEMP_STARTS:
            id = word_int_table.get(t)
            self.poem_temp_ids.append(id)

    def __call__(self, poem_id_list):
        self.poem_id_list = poem_id_list
        return self

    def __iter__(self):
        set_size = len(self.poem_id_list)
        base_cnt = self.batch_size * (set_size // self.batch_size)
        idx = np.random.permutation(set_size)

        for batch in range(0, set_size, self.batch_size):
            if batch < base_cnt:
                batch_set = [self.poem_id_list[i] for i in idx[batch:batch + self.batch_size]]
            else:  # last batch containing those items may not exactly fit the batch_size
                batch_set = [self.poem_id_list[i] for i in idx[batch:]]
                # Fill up the batch_set with items from the very beginning based on the permutation
                # for this epoch
                batch_set.extend([self.poem_id_list[i] for i in idx[:self.batch_size - len(batch_set)]])

            max_len = max(map(len, batch_set)) - 1
            x_batch = np.full((self.batch_size, max_len), self.space_token_id, np.int32)
            y_batch = np.full((self.batch_size, max_len), self.space_token_id, np.int32)
            for row in range(self.batch_size):
                row_len = len(batch_set[row]) - 1
                # [0,2,4,6,8]
                x_batch[row, :row_len] = batch_set[row][:-1]
                # [2,4,6,8,1]
                y_batch[row, :row_len] = batch_set[row][1:]

            # Initialized to be CONTENT_WEIGHT
            y_mask = np.full((self.batch_size, max_len), CONTENT_WEIGHT, np.float32)
            for row in range(self.batch_size):
                line = y_batch[row]
                line_len = len(line)

                # Set the template weight very low So that the model is not trained to predict
                # the template type, which will be fed to the model at prediction time
                y_mask[row, 0] = TEMP_WEIGHT

                if line[0] in self.poem_temp_ids:  # TANG POEMS
                    for j in range(1, line_len):
                        if line[j] in self.punc_token_ids:
                            y_mask[row, j] = P_PUNC_WEIGHT
                            y_mask[row, j - 1] = P_PRE_PUNC_WEIGHT
                        elif line[j] == self.end_token_id:
                            y_mask[row, j] = P_END_WEIGHT
                        # elif line[j] == self.space_token_id:
                        #     y_mask[row, j] = 0.0
                else:  # SONG CI
                    for j in range(1, line_len):
                        if line[j] in self.punc_token_ids:
                            y_mask[row, j] = C_PUNC_WEIGHT
                            y_mask[row, j - 1] = C_PRE_PUNC_WEIGHT
                        elif line[j] == self.end_token_id:
                            y_mask[row, j] = C_END_WEIGHT
                        # elif line[j] == self.space_token_id:
                        #     y_mask[row, j] = 0.0

            yield x_batch, y_batch, y_mask


# if __name__ == "__main__":
#     from settings import PROJECT_ROOT
#
#     start_token = '[S]'
#     end_token = '[E]'
#
#     data_dir = os.path.join(PROJECT_ROOT, 'addons', 'poembot', 'Data')
#
#     poems_id_list, word_int_table, words = process_poems(data_dir, start_token, end_token,
#                                                          poems_file='songci.txt',
#                                                          vocab_file='songci_vocab.txt')
#     bg = BatchGenerator(128, word_int_table, word_int_table.get(end_token))
#
#     for epoch in range(2):
#         cnt = 0
#         for x_batch, y_batch, y_mask in bg(poems_id_list):
#             for j in range(4):
#                 if y_batch[j, 0] == 49:
#                     cnt += 1
#                     if cnt > 5:
#                        break
#                     print("{}\n".format(str(x_batch[j])))
#                     print("{}\n".format(str(y_batch[j])))
#                     print("{}\n".format(str(y_mask[j])))
