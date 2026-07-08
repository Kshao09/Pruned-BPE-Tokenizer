# Copyright 2018 Bo Shao. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import tensorflow as tf
import tensorflow_addons as tfa

from opencc import OpenCC

from addons.ner.nerutils import Config
from addons.ner.nermodel import NERModel
from addons.rules.knowledgebase import KnowledgeBase
from chatbot.datautil import cut_text_line, emo_words, get_readable_out_text, to_dense_text

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


exclude_list = [
    '`', '@', '#', '$', '%', '^', '&', '*', '-', '=', '_', '+',
    '|', '\\', '{', '}', '[', ']', '\'', '"', '<', '>', '/',
    '、', '：', '（', '）', '～', '《', '》', '......', '...'
]
exclude_list.extend(emo_words)


class NerPredictor(object):
    def __init__(self, data_dir, model_file='ner'):
        self.config = Config()

        self.opencc = OpenCC('t2s')

        self.word_to_idx = self.config.vocab_x
        self.UNK_ID = self.word_to_idx.get(self.config.UNK_TOKEN)
        self.idx_to_tag = {idx: tag for tag, idx in self.config.vocab_y.items()}

        self.model = NERModel(config=self.config)

        # Restore model weights
        print("# Restoring model weights for NER model ...")
        result_dir = os.path.join(data_dir, 'Result')

        ckpt = tf.train.Checkpoint(model=self.model)
        ckpt.restore(os.path.join(result_dir, model_file)).expect_partial()

    def predict(self, sentence, trim=False):  # the input is already loose text
        words = cut_text_line(to_dense_text(sentence), self.opencc, split_oov=False)
        if trim:
            words = [item for item in words if item not in exclude_list]
        word_ids = list(map(lambda word: self.word_to_idx.get(word, self.UNK_ID), words))
        pred_words = []
        for i in range(2):
            pred_ids = self._model_predict([word_ids])
            pred_words = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]
            if pred_words[0] == '_tag_':
                break
            elif i == 1:
                pred_words[0] = '_tag_'
            else:
                print("NER predictor: Going to predict the second time ...")
        return pred_words, words

    def _model_predict(self, sentences):
        """
        Args:
            sentences: list of sentences
        Returns:
            viterbi_sequences: list of label_ids for each sentence
        """
        sequence_lengths = [len(sentence) for sentence in sentences]

        # Get tag scores and transition params of CRF
        viterbi_sequences = []
        logits = self._model_call(sentences, sequence_lengths)

        # Iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tfa.text.viterbi_decode(logit, self.model.transition_params)
            viterbi_sequences += [viterbi_seq]
        return viterbi_sequences

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    ])
    def _model_call(self, sentences, sequence_lengths):
        return self.model(sentences, sequence_lengths=sequence_lengths)

    def run_first_time(self):
        self.predict_usernames("我叫小明")

    def predict_usernames(self, sentence, is_final=False):
        if is_final:
            pred_words, words = self.predict(self.config.FINALNAME_PREFIX + ' ' + sentence, trim=True)
        else:
            pred_words, words = self.predict(self.config.USERNAME_PREFIX + ' ' + sentence, trim=True)
        xing_list, ming_list, final_list = [], [], []
        title = ''
        neged_xing, neged_ming = False, False
        for i in range(len(pred_words)):
            if pred_words[i] == '姓' and not neged_xing:
                xing_list.append(words[i])
            elif pred_words[i] in ['兄', '姐'] and not neged_ming:
                ming_list.append(words[i])
                title = pred_words[i]
            elif pred_words[i] in ['称']:
                final_list.append(words[i])
            elif pred_words[i] == '非':
                if len(xing_list):
                    neged_xing = True
                if len(ming_list):
                    neged_ming = True

        xing, ming, final = '', '', ''
        if len(xing_list):
            xing = get_readable_out_text(xing_list)
        if len(ming_list):
            ming = get_readable_out_text(ming_list)
        if len(final_list):
            final = get_readable_out_text(final_list)

        return xing, ming, title, final

    def predict_celebname(self, sentence):
        pred_words, words = self.predict(self.config.CELEBNAME_PREFIX + ' ' + sentence, trim=True)
        name_list = []
        for i in range(len(pred_words)):
            if pred_words[i] == '名':
                name_list.append(words[i])

        return get_readable_out_text(name_list)

    def predict_nationname(self, sentence):  # 仅支持中文实体
        pred_words, words = self.predict(self.config.NATIONNAME_PREFIX + ' ' + sentence, trim=True)
        nation_list, cap_list = [], []
        for i in range(len(pred_words)):
            if pred_words[i] == '国':
                nation_list.append(words[i])
            elif pred_words[i] == '都':
                cap_list.append(words[i])

        return ''.join(nation_list), ''.join(cap_list)

    def predict_capname(self, sentence):  # 仅支持中文实体
        pred_words, words = self.predict(self.config.CAPNAME_PREFIX + ' ' + sentence, trim=True)
        name_list = []
        for i in range(len(pred_words)):
            if pred_words[i] == '都':
                name_list.append(words[i])

        return ''.join(name_list)

    def predict_shenghui_name(self, sentence):  # 仅支持中文实体
        pred_words, words = self.predict(self.config.SHENGHUI_PREFIX + ' ' + sentence, trim=True)
        sheng_list, shi_list = [], []
        for i in range(len(pred_words)):
            if pred_words[i] == '省':
                sheng_list.append(words[i])
            elif pred_words[i] == '市':
                shi_list.append(words[i])

        return ''.join(sheng_list), ''.join(shi_list)

    def predict_ss_jiancheng_name(self, sentence):  # 仅支持中文实体
        pred_words, words = self.predict(self.config.SS_JIANCHENG_PREFIX + ' ' + sentence, trim=True)
        di_list, jc_list = [], []
        for i in range(len(pred_words)):
            if pred_words[i] == '地':
                di_list.append(words[i])
            elif pred_words[i] == '简':
                jc_list.append(words[i])

        return ''.join(di_list), ''.join(jc_list)

    def predict_meaning(self, sentence):
        pred_words, words = self.predict(self.config.MEANING_PREFIX + ' ' + sentence)
        name_list = []
        for i in range(len(pred_words)):
            if pred_words[i] == 'mean_et':
                name_list.append(words[i])

        return get_readable_out_text(name_list)

    def predict_whatis(self, sentence):
        pred_words, words = self.predict(self.config.WHATIS_PREFIX + ' ' + sentence)
        name_list = []
        for i in range(len(pred_words)):
            if pred_words[i] == 'whis_et':
                name_list.append(words[i])

        return get_readable_out_text(name_list)

    def predict_whowhatis(self, sentence):
        pred_words, words = self.predict(self.config.WHOWHATIS_PREFIX + ' ' + sentence)
        name_list = []
        for i in range(len(pred_words)):
            if pred_words[i] == 'wwis_et':
                name_list.append(words[i])

        return get_readable_out_text(name_list)

    def predict_niobj(self, sentence):
        pred_words, words = self.predict(self.config.NIOBJ_PREFIX + ' ' + sentence)
        name_list = []
        for i in range(len(pred_words)):
            if pred_words[i] == 'ni_et':
                name_list.append(words[i])

        return get_readable_out_text(name_list)

    def predict_celebname_4attr(self, sentence):
        pred_words, words = self.predict(self.config.CELEB_ATTR_PREFIX + ' ' + sentence, trim=True)
        name_list = []
        for i in range(len(pred_words)):
            if pred_words[i] == 'ceat_et':
                name_list.append(words[i])

        return get_readable_out_text(name_list)

    def predict_celebname_4work(self, sentence):
        pred_words, words = self.predict(self.config.CELEB_WORK_PREFIX + ' ' + sentence, trim=True)
        name_list = []
        for i in range(len(pred_words)):
            if pred_words[i] == 'cewk_et':
                name_list.append(words[i])

        return get_readable_out_text(name_list)

    def predict_workname_4attr(self, sentence):
        pred_words, words = self.predict(self.config.WORK_ATTR_PREFIX + ' ' + sentence, trim=True)
        name_list = []
        for i in range(len(pred_words)):
            if pred_words[i] == 'wkat_et':
                name_list.append(words[i])

        return get_readable_out_text(name_list)

    def predict_poem_line(self, sentence):  # 只可能是中文实体
        pred_words, words = self.predict(self.config.POEM_LINE_PREFIX + ' ' + sentence)
        name_list = []
        for i in range(len(pred_words)):
            if pred_words[i] == '诗':
                name_list.append(words[i])

        return ''.join(name_list)

    def predict_jielong_chengyu(self, sentence):  # 只可能是中文实体
        pred_words, words = self.predict(self.config.CHENGYU_JIELONG_PREFIX + ' ' + sentence)
        name_list = []
        for i in range(len(pred_words)):
            if pred_words[i] == 'chgyu_et':
                name_list.append(words[i])

        return ''.join(name_list)

    def predict_timezone_area(self, sentence):  # 只支持中文实体
        pred_words, words = self.predict(self.config.TIMEZONE_AREA_PREFIX + ' ' + sentence)
        name_list = []
        for i in range(len(pred_words)):
            if pred_words[i] == 'time_et':
                name_list.append(words[i])

        return ''.join(name_list)

    def predict_weather_city(self, sentence):  # 只支持中文实体
        pred_words, words = self.predict(self.config.WEATHER_CITY_PREFIX + ' ' + sentence)
        name_list = []
        for i in range(len(pred_words)):
            if pred_words[i] == 'weact_et':
                name_list.append(words[i])

        return ''.join(name_list)

    def predict_hugo_name(self, sentence):  # 只支持中文实体
        hugo_cr = KnowledgeBase.extract_hugo_criteria(sentence)
        if hugo_cr is not None:
            pred_words, words = self.predict(self.config.HUGO_MOST_PREFIX + ' ' + sentence)
            ar_list, ob_list = [], []
            for i in range(len(pred_words)):
                if pred_words[i] == 'hugo_ar':
                    ar_list.append(words[i])
                elif pred_words[i] == 'hugo_ob':
                    ob_list.append(words[i])
            return "{}+{}+{}".format(''.join(ar_list), hugo_cr, ''.join(ob_list)), True
        else:
            pred_words, words = self.predict(self.config.HUGO_NAME_PREFIX + ' ' + sentence)
            name_list = []
            for i in range(len(pred_words)):
                if pred_words[i] == 'hugob_et':
                    name_list.append(words[i])
            return ''.join(name_list), False

    def predict_org_leader_entries(self, sentence):
        pred_words, words = self.predict(self.config.ORG_LEADER_PREFIX + ' ' + sentence, trim=True)
        org_list, pos_list = [], []
        for i in range(len(pred_words)):
            if pred_words[i] == 'org_et':
                org_list.append(words[i])
            elif pred_words[i] == '职':
                pos_list.append(words[i])

        return get_readable_out_text(org_list), get_readable_out_text(pos_list)

    def predict_animal_name(self, sentence):  # 只支持中文实体
        pred_words, words = self.predict(self.config.ANIMAL_ATTR_PREFIX + ' ' + sentence)
        name_list = []
        for i in range(len(pred_words)):
            if pred_words[i] == 'aniat_et':
                name_list.append(words[i])

        return ''.join(name_list)

    def predict_attr_ctxt_ents(self, sentence):  # 仅支持中文实体
        pred_words, words = self.predict(self.config.ATTR_CTXT_PREFIX + ' ' + sentence, trim=True)
        time_list, subject_list, action_list, object_list = [], [], [], []
        for i in range(len(pred_words)):
            if pred_words[i] == 'tim_et':
                time_list.append(words[i])
            elif pred_words[i] == 'sbj_et':
                subject_list.append(words[i])
            elif pred_words[i] == 'act_et':
                action_list.append(words[i])
            elif pred_words[i] == 'obj_et':
                object_list.append(words[i])

        return ''.join(time_list), ''.join(subject_list), ''.join(action_list), ''.join(object_list)

    def predict_cn_en_trans(self, sentence):  # 只支持中文实体
        pred_words, words = self.predict(self.config.CN_EN_TRANS_PREFIX + ' ' + sentence)
        name_list = []
        for i in range(len(pred_words)):
            if pred_words[i] == 'tracn_et':
                name_list.append(words[i])

        return ''.join(name_list)


if __name__ == "__main__":
    import sys
    from settings import PROJECT_ROOT

    data_dir = os.path.join(PROJECT_ROOT, 'addons', 'ner', 'Data')
    nerp = NerPredictor(data_dir=data_dir)
    nerp.run_first_time()

    ii = 0
    app_cnt = 15
    sys.stdout.write("请输入一个句子（用户）：> ")
    sys.stdout.flush()
    quest = sys.stdin.readline().strip()
    while quest:
        if quest == 'exit':
            print("感谢您使用。再见！")
            break

        ii += 1
        if ii % app_cnt == 1:
            xing, ming, title, final = nerp.predict_usernames(quest)
            print("xing = {}, ming = {}, title = {}, final = {}".format(xing, ming, title, final))
            # For the next ii
            sys.stdout.write("请输入一个句子（是谁）：> ")
        elif ii % app_cnt == 2:
            name = nerp.predict_celebname(quest)
            print("{}".format(name))
            # For the next ii
            sys.stdout.write("请输入一个句子（国家）：> ")
        elif ii % app_cnt == 3:
            name = nerp.predict_nationname(quest)
            print("{}".format(name))
            # For the next ii
            sys.stdout.write("请输入一个句子（首都）：> ")
        elif ii % app_cnt == 4:
            name = nerp.predict_capname(quest)
            print("{}".format(name))
            # For the next ii
            sys.stdout.write("请输入一个句子（什么意思？）：> ")
        elif ii % app_cnt == 5:
            name = nerp.predict_meaning(quest)
            print("{}".format(name))
            # For the next ii
            sys.stdout.write("请输入一个句子（什么是？）：> ")
        elif ii % app_cnt == 6:
            name = nerp.predict_whatis(quest)
            print("{}".format(name))
            # For the next ii
            sys.stdout.write("请输入一个句子（知道听说过吗？）：> ")
        elif ii % app_cnt == 7:
            name = nerp.predict_whowhatis(quest)
            print("{}".format(name))
            # For the next ii
            sys.stdout.write("请输入一个句子（呢？）：> ")
        elif ii % app_cnt == 8:
            name = nerp.predict_niobj(quest)
            print("{}".format(name))
            # For the next ii
            sys.stdout.write("请输入一个句子（诗行？）：> ")
        elif ii % app_cnt == 9:
            name = nerp.predict_poem_line(quest)
            print("{}".format(name))
            # For the next ii
            sys.stdout.write("请输入一个句子（接龙成语？）：> ")
        elif ii % app_cnt == 10:
            name = nerp.predict_jielong_chengyu(quest)
            print("{}".format(name))
            # For the next ii
            sys.stdout.write("请输入一个句子（地区时间）：> ")
        elif ii % app_cnt == 11:
            name = nerp.predict_timezone_area(quest)
            print("{}".format(name))
            # For the next ii
            sys.stdout.write("请输入一个句子（城市天气）：> ")
        elif ii % app_cnt == 12:
            name = nerp.predict_weather_city(quest)
            print("{}".format(name))
            # For the next ii
            sys.stdout.write("请输入一个句子（巨型物体度量）：> ")
        elif ii % app_cnt == 13:
            name, _ = nerp.predict_hugo_name(quest)
            print("{}".format(name))
            # For the next ii
            sys.stdout.write("请输入一个句子（例如“刘德华的老婆是谁”）：> ")
        elif ii % app_cnt == 14:
            name = nerp.predict_celebname_4attr(quest)
            print("{}".format(name))
            # For the next ii
            sys.stdout.write("请输入一个句子（例如“莫言都写过哪些书”）：> ")
        else:
            name = nerp.predict_celebname_4work(quest)
            print("{}".format(name))
            # For the next ii
            sys.stdout.write("请输入一个句子（用户）：> ")

        sys.stdout.flush()
        quest = sys.stdin.readline().strip()
