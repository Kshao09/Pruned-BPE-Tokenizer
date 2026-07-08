import numpy as np
import os
import random
import tensorflow as tf

from addons.poembot.poetparams import hparams
from addons.poembot.rnnmodel import RNNModel
from addons.poembot.poemutils import cut_poem_line, load_vocab, load_poem_yun

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ShiWriter(object):
    poem_pattern_list = ['[JJ5]', '[JJ7]', '[LS5]', '[LS7]']

    poem_type_patterns = {
        '[JJ5]': '[5Z5P][NL][5Z5P]',  # 20 汉字 加 4 标点
        '[JJ7]': '[7Z7P][NL][7Z7P]',  # 28 汉字 加 4 标点
        '[LS5]': '[5Z5P][NL][5Z5P][NL][5Z5P][NL][5Z5P]',  # 40 汉字 加 8 标点
        '[LS7]': '[7Z7P][NL][7Z7P][NL][7Z7P][NL][7Z7P]'   # 56 汉字 加 8 标点
    }

    poem_reversed_patterns = {k: v[::-1].replace('[', 'X').replace(']', '[').replace('X', ']')
                              for k, v in poem_type_patterns.items()}

    # poem_seed_chars5 = "江春南山天白高万秋东何我清昔故朝北夜远古寒旧不日野上闻相有客"
    # poem_seed_chars7 = "一三十曾金莫千黄碧红年洛柳满六银一三十千黄红龙吴翠谢两锦金年"
    #
    # poem_seed_words = [
    #     "春风", "秋雨", "山水", "江海", "故人", "家乡", "东风", "山城", "独闻", "天秋",
    #     "江山", "春秋", "白日", "水寒", "万远", "长上", "少昔", "寒旧", "天春", "山天"
    # ]
    # poem_seed_words5 = [
    #     "春寒", "秋色", "山海", "江月", "天水", "高乡", "万风", "山城", "昔闻", "远秋",
    #     "江风", "春长", "白天", "夜寒", "万朝", "南山", "昔故", "寒山", "远古", "清山"
    # ]
    # poem_seed_words7 = [
    #     "三山", "三中", "长上", "万远", "独闻", "行短", "自钓", "断风", "洞万", "一乱",
    #     "少昔", "一云", "巴沉", "锦沧", "岁野", "凤三", "摇江", "群画", "孤乱", "金须"
    # ]

    def __init__(self, data_dir, model_file='poems', vocab_file='poems_vocab.txt',
                 scope='poembot_shi'):
        """
        Args:
            data_dir: Name of the folder storing vocab information.
            model_file: The file name of the trained model.
            vocab_file: The vocab file used in the training.
            scope: Variable/name scope of the model.
        """
        self.word_int_table, self.words = load_vocab(data_dir, vocab_file=vocab_file)
        self.init_yun_dict, self.other_yun_dict, self.yun_list, self.weight_list = load_poem_yun(data_dir)
        self.vocab_size = len(self.words)

        self.rnn_model = RNNModel(hparams, self.vocab_size, name=scope)

        # Restore model weights
        print("# Restoring model weights for Shi Writer ...")
        result_dir = os.path.join(data_dir, 'Result')

        ckpt = tf.train.Checkpoint(model=self.rnn_model)
        ckpt.restore(os.path.join(result_dir, model_file)).expect_partial()

    def run_first_time(self):
        self.write_poem('[JJ5]')

    def write_poem(self, poem_type):
        if poem_type not in self.poem_pattern_list:
            return ''

        poem = ''

        # [S]
        x = np.array([[self.word_int_table.get(hparams['start_token'])]])
        # Discard the prediction due to start_token, change state only
        _, last_state = self.rnn_model(tf.convert_to_tensor(x), initial_state=None)

        # Such as [JJ5]
        x = np.array([[self.word_int_table[poem_type]]])
        prediction, last_state = self.rnn_model(
            tf.convert_to_tensor(x), initial_state=last_state
        )

        pat_list = cut_poem_line(self.poem_reversed_patterns.get(poem_type))
        init_id, other_id_list = self._get_yun_lists()
        pos = 0
        used_list = []

        for r_pat in pat_list:
            if r_pat == '[PN]':
                poem += '_pn_'  # reversed
                continue
            elif r_pat == '[LN]':  # reversed
                poem += '_ln_'  # reversed
                continue

            word = r_pat
            word_id = self.word_int_table[word]

            for ii in range(3):
                if ii > 0:  # The first one is the pattern word
                    if ii == 1:
                        word_id = self.word_int_table['。']
                    else:
                        assert r_pat[1] == 'P'

                        if pos == 1:
                            word_id = init_id
                        else:
                            word_id = self._get_word_id_with_yun(prediction, other_id_list, used_list)
                        used_list.append(word_id)
                    pos += 1
                    word = self.words[word_id]

                    poem += word

                x = np.array([[word_id]])
                prediction, last_state = self.rnn_model(
                    tf.convert_to_tensor(x), initial_state=last_state
                )
                # Only the last returns are useful and go to the next loop
                word, word_id = self._to_word(prediction)

            cnt = self.get_pat_count(r_pat)
            for ii in range(cnt - 2):
                poem += word
                x = np.array([[word_id]])
                prediction, last_state = self.rnn_model(
                    tf.convert_to_tensor(x), initial_state=last_state
                )
                # The last one word predicted in this inner loop should be the next pattern,
                # and it will be discarded.
                word, word_id = self._to_word(prediction)

        return poem[::-1]

    def write_valid_poem(self, seed_chars, tries=2):
        for _ in range(tries):
            content = self.write_poem(seed_chars)
            if self.is_jj5(content) or self.is_jj7(content) or self.is_ls5(content) or self.is_ls7(content):
                return content
        return None

    def write_poem_by_type(self, poem_type, tries=2):
        if poem_type == 'jj57':
            for _ in range(tries):
                seed_form = random.choice(['[JJ5]', '[JJ7]'])
                content = self.write_poem(seed_form)
                if self.is_jj5(content) or self.is_jj7(content):
                    return content
        elif poem_type == 'ls57':
            for _ in range(tries):
                seed_form = random.choice(['[LS5]', '[LS7]'])
                content = self.write_poem(seed_form)
                if self.is_ls5(content) or self.is_ls7(content):
                    return content
        elif poem_type == 'jj5':
            for _ in range(tries):
                content = self.write_poem('[JJ5]')
                if self.is_jj5(content):
                    return content
        elif poem_type == 'jj7':
            for _ in range(tries):
                content = self.write_poem('[JJ7]')
                if self.is_jj7(content):
                    return content
        elif poem_type == 'ls5':
            for _ in range(tries):
                content = self.write_poem('[LS5]')
                if self.is_ls5(content):
                    return content
        elif poem_type == 'ls7':
            for _ in range(tries):
                content = self.write_poem('[LS7]')
                if self.is_ls7(content):
                    return content

        return None

    def _to_word(self, predict_id):
        """
        In order to let the generated poem be more interesting, we do not pick the word having the highest
        probability. Instead, we map the predicted probability into an area. We then randomly pick a word
        within the area, in which the word that has higher predicted probability gets a greater chance to
        be selected.
        """
        t = np.cumsum(predict_id)
        s = np.sum(predict_id)
        sample_id = int(np.searchsorted(t, np.random.rand(1) * s))
        if sample_id >= len(self.words):
            sample_id = len(self.words) - 1
        return self.words[sample_id], sample_id

    def _get_yun_lists(self):
        picked = random.choices(self.yun_list, weights=self.weight_list, k=1)[0]
        init_list = self.init_yun_dict.get(picked)
        other_list = self.other_yun_dict.get(picked)

        init_char = random.choice(init_list)

        other_id_list = []
        for w in other_list:
            other_id_list.append(self.word_int_table[w])

        return self.word_int_table[init_char], other_id_list

    @staticmethod
    def _get_word_id_with_yun(prediction, id_list, used_list):
        max_prob, max_id = 0, 0

        for id in id_list:
            if id in used_list:
                continue
            this_prob = prediction[0, id]
            if this_prob > max_prob:
                max_prob = this_prob
                max_id = id
        return max_id

    @staticmethod
    def get_pat_count(reversed_pattern):
        pat = reversed_pattern[1:-1]  # Remove brackets
        assert len(pat) == 4
        cnt = int(pat[1]) + int(pat[3]) + 2
        return cnt

    @staticmethod
    def is_jj5(content):
        if content.count('_nl_') != 1:
            return False
        if content.count('，') + content.count('。') + content.count('？') + content.count('！') != 4:
            return False
        tmp = content.replace('_nl_', '')
        if len(tmp) != 24 or tmp.count('，') + tmp.count('。') + tmp.count('？') + tmp.count('！') != 4:
            return False
        if tmp[5] not in ['，', '？'] or tmp[17] not in ['，', '？']:
            return False
        if tmp[11] not in ['。', '？', '！'] or tmp[23] not in ['。', '？', '！']:
            return False
        return True

    @staticmethod
    def is_jj7(content):
        if content.count('_nl_') != 1:
            return False
        if content.count('，') + content.count('。') + content.count('？') + content.count('！') != 4:
            return False
        tmp = content.replace('_nl_', '')
        if len(tmp) != 32 or tmp.count('，') + tmp.count('。') + tmp.count('？') + tmp.count('！') != 4:
            return False
        if tmp[7] not in ['，', '？'] or tmp[23] not in ['，', '？']:
            return False
        if tmp[15] not in ['。', '？', '！'] or tmp[31] not in ['。', '？', '！']:
            return False
        return True

    @staticmethod
    def is_ls5(content):
        if content.count('_nl_') != 3:
            return False
        if content.count('，') + content.count('。') + content.count('？') + content.count('！') != 8:
            return False
        tmp = content.replace('_nl_', '')
        if len(tmp) != 48 or tmp.count('，') + tmp.count('。') + tmp.count('？') + tmp.count('！') != 8:
            return False
        for i in [5, 17, 29, 41]:
            if tmp[i] not in ['，', '？']:
                return False
        for i in [11, 23, 35, 47]:
            if tmp[i] not in ['。', '？', '！']:
                return False
        return True

    @staticmethod
    def is_ls7(content):
        if content.count('_nl_') != 3:
            return False
        if content.count('，') + content.count('。') + content.count('？') + content.count('！') != 8:
            return False
        tmp = content.replace('_nl_', '')
        if len(tmp) != 64 or tmp.count('，') + tmp.count('。') + tmp.count('？') + tmp.count('！') != 8:
            return False
        for i in [7, 23, 39, 55]:
            if tmp[i] not in ['，', '？']:
                return False
        for i in [15, 31, 47, 63]:
            if tmp[i] not in ['。', '？', '！']:
                return False
        return True


if __name__ == '__main__':
    import sys
    from settings import PROJECT_ROOT

    data_dir = os.path.join(PROJECT_ROOT, 'addons', 'poembot', 'Data')
    sw = ShiWriter(data_dir=data_dir)
    sw.run_first_time()

    sys.stdout.write("请输入诗歌的类型简称（[JJ5],[JJ7],[LS5],[LS7]）：> ")
    sys.stdout.flush()
    seed_chars = sys.stdin.readline().strip()
    while seed_chars:
        if seed_chars == 'exit':
            print("感谢邀请小瓜诗人作诗。再见！")
            break

        # A very loose verification is performed here.
        chars_count = len(seed_chars)
        if seed_chars in sw.poem_pattern_list:
            # Internal test only
            poem = sw.write_poem(seed_chars)
            print(poem.replace('_nl_', '\n'))

        sys.stdout.write("请输入诗歌的类型简称（[JJ5],[JJ7],[LS5],[LS7]）：> ")
        sys.stdout.flush()
        seed_chars = sys.stdin.readline().strip()