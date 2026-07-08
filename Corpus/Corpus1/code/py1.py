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
"""This class is only used at inference time."""
import os
import re
import random
import datetime as dt
from collections import namedtuple
from googletrans import Translator

from chatbot.datautil import get_formatted_text_line, to_dense_text
from addons.rules.weatherutils import CityKeyDesc
from addons.rules.wikiscraper import get_first_paragraph, get_person_entry_dict, get_thing_entry_dict, WORK_CTY_LIST

STORIES_FILE = "stories.txt"
JOKES_FILE = "jokes.txt"
DUANZI_FILE = "duanzi.txt"
POEMS_FILE = "poems.txt"
LYRICS_FILE = "lyrics.txt"
CHENGYU_FILE = "chengyu.txt"
TIMEZONE_CITIES_FILE = "timezone_cities.txt"
WEATHER_CITIES_FILE = "weather_cities.txt"
SPEC_UNAME_FILE = "spec_uname.txt"
CELEB_INFO_FILE = "celeb_info.txt"
NATION_CAP_FILE = "nation_cap.txt"
PROV_CITY_FILE = "prov_city_info.txt"
ENTRY_MEANING_FILE = "entry_meaning.txt"
ENTRY_WHATIS_FILE = "entry_whatis.txt"
NIOBJ_INFO_FILE = "niobj_info.txt"
# KEY_VAL_PAIR_FILE = "key_val_pairs.txt"
CELEB_BASE_DATA_FILE = "celeb_base_data.txt"
CELEB_ATTR_DATA_FILE = "celeb_attr_data.txt"
THING_ALIAS_WIKI_FILE = "thing_alias_wiki.txt"
HUGE_OBJ_DATA_FILE = "huge_obj_data.txt"
WORK_ATTR_DATA_FILE = "work_attr_data.txt"
ORG_LEADER_DATA_FILE = "org_leader_data.txt"
ANIMAL_ATTR_FILE = "animal_attr.txt"
HIST_EVENTS_FILE = "hist_events.txt"
SPORT_EVENT_DATA_FILE = "sport_event_data.txt"
CN_EN_TRANS_FILE = "cn_en_trans.txt"
SP_ANS_FILE = "special_ans.txt"
PRELOAD_PAIR_FILE = "preload_pairs.txt"

COMMENT_LINE_STT = "#=="
CONVERSATION_SEP = "==="

CELEB_ATTR_MAP = {
    '生日': '生日',
    '年龄': '年龄',
    '性别': '性别',
    '姓氏': '姓氏',
    '字号': '字号',
    '名字': '名字',
    '原名': '原名',
    '本名': '原名',
    '笔名': '笔名',
    '国籍': '国籍',
    '籍贯': '籍贯',
    '户籍': '户籍',
    '身高': '身高',
    '高度': '身高',
    '体重': '体重',
    '重量': '体重',
    '逝世': '逝世日期',
}

CELEB_MAIN_RELATION_SET = {'男友', '女友', '前任',
                           '前夫', '前老公', '老公', '丈夫', '先生',
                           '前妻', '前老婆', '老婆', '妻子', '媳妇', '太太',
                           '配偶', '父亲', '爸爸', '母亲', '妈妈'}

CELEB_RELATION_MAP = {
    '男友': '男友',
    '女友': '女友',
    '前任': '前任',
    '前夫': '前任',
    '前老公': '前任',
    '老公': '老公',
    '丈夫': '老公',
    '先生': '老公',
    '前妻': '前任',
    '前老婆': '前任',
    '老婆': '老婆',
    '妻子': '老婆',
    '媳妇': '老婆',
    '太太': '老婆',
    '配偶': '配偶',
    '父亲': '父亲',
    '爸爸': '父亲',
    '母亲': '母亲',
    '妈妈': '母亲',
    '其他': '其他',
    '祖父': '祖父',
    '祖母': '祖母',
    '岳父': '岳父',
    '岳母': '岳母',
    '老丈人': '岳父',
    '丈人': '岳父',
    '岳丈': '岳父',
    '丈母娘': '岳母',
    '公爹': '公爹',
    '公公': '公爹',
    '婆母': '婆母',
    '婆婆': '婆母',
    '儿子': '儿子',
    '女儿': '女儿',
    '子女': '子女',
    '儿女': '子女',
    '小孩': '子女',
    '孩子': '子女',
    '哥哥': '哥哥',
    '弟弟': '弟弟',
    '姐姐': '姐姐',
    '妹妹': '妹妹',
    '师父': '师父',
    '弟子': '弟子',
}

CELEB_WORK_MAP = {
    '诗歌': '诗歌',
    '小说': '小说',
    '武侠小说': '小说',
    '言情小说': '小说',
    '科幻小说': '小说',
    '侦探小说': '小说',
    '代表小说': '小说',
    '书籍': '书籍',
    '代表书籍': '书籍',
    '音乐专辑': '音乐专辑',
    '代表专辑': '音乐专辑',
    '音乐': '歌曲',
    '歌曲': '歌曲',
    '代表音乐': '歌曲',
    '代表歌曲': '歌曲',
    '电影': '电影',
    '电视剧': '电视剧',
    '代表作': '代表作',
}

CELEB_ATTR_CONVERT_MAP = {**CELEB_ATTR_MAP, **CELEB_RELATION_MAP, **CELEB_WORK_MAP}

CELEB_ATTR_CATEG_DICT = {
    '生日': '生日是哪天',
    '年龄': '年龄多大',
    '性别': '是男是女',
    '姓氏': '姓氏是什么',
    '字号': '字什么，号什么',
    '名字': '叫什么名字',
    '原名': '原名叫什么',
    '笔名': '笔名叫什么',
    '国籍': '是哪国人',
    '籍贯': '是什么地方人',
    '户籍': '是什么地方人',
    '身高': '身高是多少',
    '体重': '体重是多少',
    '男友': '男友是谁',
    '女友': '女友是谁',
    '前任': '前任是谁',
    '前夫': '前夫是谁',
    '老公': '老公是谁',
    '前妻': '前妻是谁',
    '老婆': '老婆是谁',
    '配偶': '配偶是谁',
    '父亲': '父亲是谁',
    '母亲': '母亲是谁',
    '祖父': '祖父是谁',
    '祖母': '祖母是谁',
    '岳父': '岳父是谁',
    '岳母': '岳母是谁',
    '公爹': '公爹是谁',
    '婆母': '婆母是谁',
    '儿子': '儿子是谁',
    '女儿': '女儿是谁',
    '子女': '子女是谁',
    '哥哥': '哥哥是谁',
    '弟弟': '弟弟是谁',
    '姐姐': '姐姐是谁',
    '妹妹': '妹妹是谁',
    '师父': '师父是谁',
    '弟子': '弟子是谁',
    '逝世日期': '逝世日期是哪天',
    '诗歌': '写过哪些诗歌',
    '小说': '写过哪些小说',
    '书籍': '写过哪些书籍',
    '音乐专辑': '有哪些音乐专辑',
    '歌曲': '有哪些歌曲作品',
    '电影': '演过哪些电影',
    '电视剧': '演过哪些电视剧',
    '代表作': '有哪些代表作品',
}

HUGE_CR_QUALS = "长短高深宽窄大小重多少"

ANIMAL_ATTR_MAP = {
    '动植物': '是动物还是植物',
    '有腿': '是否有腿',
    '有翅': '是否有翅膀',
    '有尾': '是否有尾巴',
    '有眼': '是否有眼睛',
    '有耳': '是否有耳朵',
    '会飞': '是否会飞',
    '会游': '是否会游泳',
    '会爬树': '是否会爬树',
    '会跑': '是否会跑',
    '会走': '是否会行走',
}


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class KnowledgeBase(object, metaclass=Singleton):
    def __init__(self, opencc_t2s=None):
        self.opencc_t2s = opencc_t2s

        self.story_stat = {}
        self.stories = {}
        self.jokes = []
        self.duanzi = []
        self.poems = {}
        self.poems_no_exp = {}  # 无解释的诗，只能被用户指定诗句等方式提取，不会主动去背诵
        self.poem_dynasties = {}
        self.poem_writers = {}
        self.poem_lines_list = []
        self.poem_lines_dict = {}
        self.lyrics = {}
        self.chengyu_dict = {}   # 键:成语第一字; 值:字符串列表，每个字符串为以该字开头的成语的后三字
        self.timezone_cities_dict = {}
        self.weather_cities_dict = {}
        self.spec_uname_dict = {}
        self.celeb_dict = {}
        self.nation_cap_dict, self.cap_nation_dict = {}, {}
        self.prov_city_dict, self.city_prov_dict = {}, {}
        self.state_shoufu_dict, self.shoufu_state_dict = {}, {}
        self.prov_city_jc_dict, self.jc_prov_dict, self.jc_city_dict = {}, {}, {}
        self.entry_meaning_dict = {}
        self.entry_whatis_dict = {}
        self.niobj_info_dict = {}
        # self.key_val_dict = {}
        self.celeb_alias_dict = {}
        self.chinese_xing_set, self.celeb_male_set, self.celeb_female_set = set(), set(), set()
        self.attr_celeb_dict, self.celeb_attr_cache_dict = {}, {}
        self.whatis_alias_dict = {}
        self.huge_obj_data, self.thing_attr_cache_dict = {}, {}
        self.work_attr_data, self.work_alias_dict = {}, {}
        self.org_leader_data = {}
        self.animal_alias_dict, self.attr_animal_dict = {}, {}
        self.animal_set, self.plant_set, self.non_animal_non_plant_set = set(), set(), set()
        self.hist_events_dict = {}
        self.sport_event_data_dict = {}
        self.cn_en_trans_dict = {}
        self.sp_ans1_list, self.sp_ans2_list, self.sp_ans3_list, self.sp_ans4_list = [], [], [], []
        self.preload_pairs = {}
        # A global cache to store the weather query results
        # key=eng_city_desc, val=CityWeather
        self.weather_info_cache = {}
        # sets to keep the scraping history if a key search was not found. Then we do not waste time to retry
        self.wiki_first_para_neg_set, self.wiki_person_entry_neg_set, self.wiki_thing_entry_neg_set = set(), set(), set()

    def load_knbase(self, knbase_dir):
        """
        Args:
             knbase_dir: Name of the KnowledgeBase folder. The file names inside are fixed.
        """
        self._load_stories(knbase_dir)
        self._load_jokes(knbase_dir)
        self._load_duanzi(knbase_dir)
        self._load_poems(knbase_dir)
        self._load_lyrics(knbase_dir)
        self._load_chengyu(knbase_dir)
        self._load_timezone_cities(knbase_dir)
        self._load_weather_cities(knbase_dir)
        self._load_spec_uname(knbase_dir)
        self._load_celeb_info(knbase_dir)
        self._load_nation_cap(knbase_dir)
        self._load_prov_city_info(knbase_dir)
        self._load_entry_meaning(knbase_dir)
        self._load_entry_whatis(knbase_dir)
        self._load_niobj_info(knbase_dir)
        # self._load_key_val_pairs(knbase_dir)
        self._load_celeb_base_data(knbase_dir)
        self._load_celeb_attr_data(knbase_dir)
        self._load_thing_alias_wiki(knbase_dir)
        self._load_huge_obj_data(knbase_dir)
        self._load_work_attr_data(knbase_dir)
        self._load_org_leader_data(knbase_dir)
        self._load_animal_attr_data(knbase_dir)
        self._load_hist_events(knbase_dir)
        self._load_sport_event_data(knbase_dir)
        self._load_cn_en_trans_data(knbase_dir)
        self._load_sp_ans(knbase_dir)
        self._load_preload_pairs(knbase_dir)

    def _load_stories(self, knbase_dir):
        stories_file_name = os.path.join(knbase_dir, STORIES_FILE)

        story_id = 0
        with open(stories_file_name, 'r', encoding='utf-8') as stories_f:
            s_cat, s_content = '', []
            for line in stories_f:
                ln = line.strip()
                if not ln or ln.startswith('#'):
                    continue
                if ln.startswith('_CAT:'):
                    if s_cat != '' and len(s_content) > 0:
                        story_id += 1
                        self.stories[story_id] = Story(cat=s_cat, content=s_content)
                        current_stat = self.story_stat.get(s_cat, [])
                        current_stat.append(story_id)
                        self.story_stat[s_cat] = current_stat
                        s_name, s_content = '', []
                    s_cat = ln[5:].strip().lower()
                elif ln.startswith('_CONTENT:'):
                    s_content.append(ln[9:].strip())
                else:
                    s_content.append(ln.strip())

            if s_cat != '' and len(s_content) > 0:  # The last one
                story_id += 1
                self.stories[story_id] = Story(cat=s_cat, content=s_content)
                current_stat = self.story_stat.get(s_cat, [])
                current_stat.append(story_id)
                self.story_stat[s_cat] = current_stat

    def _load_jokes(self, knbase_dir):
        jokes_file_name = os.path.join(knbase_dir, JOKES_FILE)

        with open(jokes_file_name, 'r', encoding='utf-8') as jokes_f:
            j_content = ''
            for line in jokes_f:
                ln = line.strip()
                if not ln or ln.startswith('#'):
                    continue
                if ln.startswith('_CONTENT:'):
                    if j_content != '':
                        self.jokes.append(j_content)
                    j_content = ln[9:].strip()
                else:
                    j_content += '_np_' + ln.strip()

            if j_content != '':  # The last one
                self.jokes.append(j_content)

    def _load_duanzi(self, knbase_dir):
        duanzi_file_name = os.path.join(knbase_dir, DUANZI_FILE)

        with open(duanzi_file_name, 'r', encoding='utf-8') as duanzi_f:
            d_content = ''
            for line in duanzi_f:
                ln = line.strip()
                if not ln or ln.startswith('#'):
                    continue
                if ln.startswith('_CONTENT:'):
                    if d_content != '':
                        self.duanzi.append(d_content)
                    d_content = ln[9:].strip()
                else:
                    d_content += '_np_' + ln.strip()

            if d_content != '':  # The last one
                self.duanzi.append(d_content)

    def _load_poems(self, knbase_dir):
        poems_file_name = os.path.join(knbase_dir, POEMS_FILE)

        poem_id = 0
        with open(poems_file_name, 'r', encoding='utf-8') as poems_f:
            p_title, p_writer, p_content, p_cont_exp, p_explanation = '', '', '', '', ''
            last_item = None
            for line in poems_f:
                ln = line.strip()
                if not ln or ln.startswith('#'):
                    continue
                if ln.startswith('_TITLE:'):
                    if p_title != '' and p_writer != '' and p_content != '':
                        poem_id += 1
                        self._add_poem(poem_id, p_title, p_writer, p_content, p_cont_exp, p_explanation)
                        p_title, p_writer, p_content, p_cont_exp, p_explanation = '', '', '', '', ''
                    p_title = ln[7:].strip().lower()
                elif ln.startswith('_WRITER:'):
                    p_writer = ln[8:].strip()
                elif ln.startswith('_CONTENT:'):
                    cont_ln = ln[9:].strip()
                    if cont_ln.startswith('_exp_'):
                        p_cont_exp, p_content = cont_ln[5:].strip(), ''
                    else:
                        p_content = cont_ln
                    last_item = 'CONTENT'
                elif ln.startswith('_EXPLANATION:'):
                    p_explanation = ln[13:].strip()
                    last_item = 'EXPLANATION'
                else:
                    if last_item == 'CONTENT':
                        p_content += '_nl_' + ln.strip()
                    elif last_item == 'EXPLANATION':
                        p_explanation += '_nl_' + ln.strip()

            if p_title != '' and p_writer != '' and p_content != '':  # The last one
                poem_id += 1
                self._add_poem(poem_id, p_title, p_writer, p_content, p_cont_exp, p_explanation)

    def _load_lyrics(self, knbase_dir):
        lyrics_file_name = os.path.join(knbase_dir, LYRICS_FILE)

        lyric_id = 0
        with open(lyrics_file_name, 'r', encoding='utf-8') as lyrics_f:
            l_title, l_writer, l_composer, l_singer, l_content = '', '', '', '', ''
            last_item = None
            for line in lyrics_f:
                ln = line.strip()
                if not ln and last_item == 'CONTENT':
                    l_content += '_nl_'
                    continue
                elif not ln or ln.startswith('#'):
                    continue
                if ln.startswith('_TITLE:'):
                    if l_title != '' and l_content != '':
                        lyric_id += 1
                        self.lyrics[lyric_id] = Lyric(title=l_title, writer=l_writer, composer=l_composer,
                                                      singer=l_singer, content=l_content)
                        l_title, l_writer, l_composer, l_singer, l_content = '', '', '', '', ''
                    l_title = ln[7:].strip().lower()
                elif ln.startswith('_WRITER:'):
                    l_writer = ln[8:].strip()
                elif ln.startswith('_COMPOSER:'):
                    l_composer = ln[10:].strip()
                elif ln.startswith('_SINGER:'):
                    l_singer = ln[8:].strip()
                elif ln.startswith('_CONTENT:'):
                    l_content = ln[9:].strip()
                    last_item = 'CONTENT'
                elif last_item == 'CONTENT':
                    l_content += '_nl_' + ln.strip()

            if l_title != '' and l_content != '':  # The last one
                lyric_id += 1
                self.lyrics[lyric_id] = Lyric(title=l_title, writer=l_writer, composer=l_composer,
                                              singer=l_singer, content=l_content)

    def _load_chengyu(self, knbase_dir):
        chengyu_file_name = os.path.join(knbase_dir, CHENGYU_FILE)

        with open(chengyu_file_name, 'r', encoding='utf-8') as chengyu_f:
            for line in chengyu_f:
                ln = line.strip()
                if not ln:
                    continue
                ky = ln[:1]
                vl = ln[1:]
                if ky in self.chengyu_dict:
                    cy_list = self.chengyu_dict[ky]
                    cy_list.append(vl)
                else:
                    self.chengyu_dict[ky] = [vl]

    def _load_timezone_cities(self, knbase_dir):
        timezone_cities_file_name = os.path.join(knbase_dir, TIMEZONE_CITIES_FILE)

        with open(timezone_cities_file_name, 'r', encoding='utf-8') as timezone_cities_f:
            for line in timezone_cities_f:
                ln = line.strip()
                if not ln or ln.startswith('#'):
                    continue
                cities = ln.split("：：")
                c_zhs = cities[0].strip().split('|')
                c_eng = cities[1].strip()

                for cz in c_zhs:
                    self.timezone_cities_dict[cz] = CityKeyDesc(c_eng, c_zhs[0])

    def _load_weather_cities(self, knbase_dir):
        weather_cities_file_name = os.path.join(knbase_dir, WEATHER_CITIES_FILE)

        with open(weather_cities_file_name, 'r', encoding='utf-8') as weather_cities_f:
            for line in weather_cities_f:
                ln = line.strip()
                if not ln or ln.startswith('#'):
                    continue
                cities = ln.split("：：")
                c_zhs = cities[0].strip().split('|')
                c_eng = cities[1].strip()

                for cz in c_zhs:
                    self.weather_cities_dict[cz] = CityKeyDesc(c_eng, c_zhs[0])

    def _load_spec_uname(self, knbase_dir):
        spec_uname_file_name = os.path.join(knbase_dir, SPEC_UNAME_FILE)

        with open(spec_uname_file_name, 'r', encoding='utf-8') as spec_uname_f:
            for line in spec_uname_f:
                ln = line.strip()
                if not ln or ln.startswith('#') or ln == '===':
                    continue
                spec_uname_info = ln.split("：：")
                spec_unames = spec_uname_info[0].strip().split('|')
                uname_desc = spec_uname_info[1].strip()

                for uname in spec_unames:
                    self.spec_uname_dict[uname.lower()] = uname_desc

    def _load_celeb_info(self, knbase_dir):
        celeb_info_file_name = os.path.join(knbase_dir, CELEB_INFO_FILE)

        with open(celeb_info_file_name, 'r', encoding='utf-8') as celeb_info_f:
            for line in celeb_info_f:
                ln = line.strip()
                if not ln or ln.startswith('#') or ln == '===':
                    continue
                celeb_info = ln.split("：：")
                celeb_names = celeb_info[0].strip().split('|')
                celeb_desc = celeb_info[1].strip()

                def_name = celeb_names[0].lower()
                for cn in celeb_names:
                    assert cn.lower not in self.celeb_dict
                    new_desc = celeb_desc.replace('_CN_', cn).replace('_RPCN_', cn)
                    new_desc = new_desc.replace('_func_ta1_to_be_replaced',
                                                '_func_set_pronoun_para0_flw_para1_ta1 [{}]'.format(def_name))
                    new_desc = new_desc.replace('_func_ta2_to_be_replaced',
                                                '_func_set_pronoun_para0_flw_para1_ta2 [{}]'.format(def_name))
                    new_desc = new_desc.replace('_func_tamen1_to_be_replaced',
                                                '_func_set_pronoun_para0_flw_para1_tamen1 [{}]'.format(def_name))
                    new_desc = new_desc.replace('_func_tamen2_to_be_replaced',
                                                '_func_set_pronoun_para0_flw_para1_tamen2 [{}]'.format(def_name))
                    self.celeb_dict[cn.lower()] = new_desc

    def _load_nation_cap(self, knbase_dir):
        nation_cap_file_name = os.path.join(knbase_dir, NATION_CAP_FILE)

        with open(nation_cap_file_name, 'r', encoding='utf-8') as nation_cap_f:
            for line in nation_cap_f:
                ln = line.strip()
                if not ln or ln.startswith('#') or ln == '===':
                    continue
                nation_cap = ln.split("：：")
                nation_names = nation_cap[0].strip().split('|')
                cap_name = nation_cap[1].strip()

                for nn in nation_names:
                    self.nation_cap_dict[nn] = cap_name
                self.cap_nation_dict[cap_name] = nation_names[0]

    def _load_prov_city_info(self, knbase_dir):
        prov_city_file_name = os.path.join(knbase_dir, PROV_CITY_FILE)

        cur_cat = None
        with open(prov_city_file_name, 'r', encoding='utf-8') as prov_city_f:
            for line in prov_city_f:
                ln = line.strip()
                if not ln or ln.startswith('==='):
                    continue
                if ln.startswith('#='):
                    cur_cat = ln[2:]
                else:
                    kv_pair = ln.split("：：")
                    keys = kv_pair[0].strip().split('|')
                    val_txt = kv_pair[1].strip()

                    if cur_cat == '省会':
                        for kk in keys:
                            self.prov_city_dict[kk] = val_txt
                        self.city_prov_dict[val_txt] = keys[0]
                    elif cur_cat == '美国首府':
                        vals = val_txt.split('|')
                        for kk in keys:
                            self.state_shoufu_dict[kk] = vals[0]
                        for vv in vals:
                            self.shoufu_state_dict[vv] = keys[0]
                    elif cur_cat == '省市简称':
                        for kk in keys:
                            self.prov_city_jc_dict[kk] = val_txt
                    elif cur_cat == '简称省份':
                        for kk in keys:
                            self.jc_prov_dict[kk] = val_txt
                    elif cur_cat == '简称城市':
                        for kk in keys:
                            self.jc_city_dict[kk] = val_txt
                    else:
                        raise ValueError("Unknown category value: {}.".format(cur_cat))

    def _load_entry_meaning(self, knbase_dir):
        entry_meaning_file_name = os.path.join(knbase_dir, ENTRY_MEANING_FILE)

        with open(entry_meaning_file_name, 'r', encoding='utf-8') as entry_meaning_f:
            for line in entry_meaning_f:
                ln = line.strip()
                if not ln or ln.startswith('#') or ln == '===':
                    continue
                entry_meaning = ln.split("：：")
                entries = entry_meaning[0].strip().split('|')
                entry_desc = entry_meaning[1].strip()

                def_entry = entries[0].lower()
                for et in entries:
                    new_desc = entry_desc.replace('_EN_', et)
                    new_desc = new_desc.replace('_func_ta1_to_be_replaced',
                                                '_func_set_pronoun_para0_flw_para1_ta1 [{}]'.format(def_entry))
                    new_desc = new_desc.replace('_func_ta2_to_be_replaced',
                                                '_func_set_pronoun_para0_flw_para1_ta2 [{}]'.format(def_entry))
                    new_desc = new_desc.replace('_func_ta3_to_be_replaced',
                                                '_func_set_pronoun_para0_flw_para1_ta3 [{}]'.format(def_entry))
                    self.entry_meaning_dict[et.lower()] = new_desc

    def _load_entry_whatis(self, knbase_dir):
        entry_whatis_file_name = os.path.join(knbase_dir, ENTRY_WHATIS_FILE)

        with open(entry_whatis_file_name, 'r', encoding='utf-8') as entry_whatis_f:
            for line in entry_whatis_f:
                ln = line.strip()
                if not ln or ln.startswith('#') or ln == '===':
                    continue
                entry_whatis = ln.split("：：")
                entries = entry_whatis[0].strip().split('|')
                entry_desc = entry_whatis[1].strip()

                def_entry = entries[0].lower()
                assert def_entry not in self.entry_whatis_dict
                new_desc = entry_desc.replace('_EN_', def_entry)
                if new_desc.find('_KEPT0_') < 0:
                    func_idx = new_desc.find('_func_')
                    if func_idx > 0:
                        new_desc = "_CASE0_{}_CASE1_ {}".format(new_desc[:func_idx].strip(), new_desc[func_idx:].strip())
                    else:
                        new_desc = "_CASE0_{}_CASE1_".format(new_desc.strip())
                new_desc = new_desc.replace('_func_ta1_to_be_replaced',
                                            '_func_set_pronoun_para0_flw_para1_ta1 [{}]'.format(def_entry))
                new_desc = new_desc.replace('_func_ta2_to_be_replaced',
                                            '_func_set_pronoun_para0_flw_para1_ta2 [{}]'.format(def_entry))
                new_desc = new_desc.replace('_func_ta3_to_be_replaced',
                                            '_func_set_pronoun_para0_flw_para1_ta3 [{}]'.format(def_entry))
                self.entry_whatis_dict[def_entry] = new_desc

                if len(entries) > 1:
                    for idx, et in enumerate(entries, start=0):
                        if idx == 0:
                            continue
                        kk = et.lower()
                        assert kk not in self.whatis_alias_dict
                        self.whatis_alias_dict[kk] = def_entry

    def _load_niobj_info(self, knbase_dir):
        niobj_info_file_name = os.path.join(knbase_dir, NIOBJ_INFO_FILE)

        with open(niobj_info_file_name, 'r', encoding='utf-8') as niobj_info_f:
            for line in niobj_info_f:
                ln = line.strip()
                if not ln or ln.startswith('#') or ln == '===':
                    continue
                niobj_info = ln.split("：：")
                niobj = niobj_info[0].strip()
                niobj_desc = niobj_info[1].strip()

                new_desc = niobj_desc.replace('_func_ta1_to_be_replaced',
                                              '_func_set_pronoun_para0_flw_para1_ta1 [{}]'.format(niobj))
                new_desc = new_desc.replace('_func_ta2_to_be_replaced',
                                            '_func_set_pronoun_para0_flw_para1_ta2 [{}]'.format(niobj))
                new_desc = new_desc.replace('_func_ta3_to_be_replaced',
                                            '_func_set_pronoun_para0_flw_para1_ta3 [{}]'.format(niobj))
                self.niobj_info_dict[niobj.lower()] = new_desc

    # def _load_key_val_pairs(self, knbase_dir):
    #     key_val_pair_file_name = os.path.join(knbase_dir, KEY_VAL_PAIR_FILE)
    #
    #     with open(key_val_pair_file_name, 'r', encoding='utf-8') as key_val_pair_f:
    #         for line in key_val_pair_f:
    #             ln = line.strip()
    #             if not ln or ln.startswith('#') or ln == '===':
    #                 continue
    #             kv_pair = ln.split("：：")
    #             kks = kv_pair[0].strip().split('|')
    #             vv = kv_pair[1].strip()
    #
    #             for kk in kks:
    #                 self.key_val_dict[kk.lower()] = vv

    def _load_celeb_base_data(self, knbase_dir):
        celeb_base_data_file_name = os.path.join(knbase_dir, CELEB_BASE_DATA_FILE)

        cur_base = None
        with open(celeb_base_data_file_name, 'r', encoding='utf-8') as celeb_base_data_f:
            for line in celeb_base_data_f:
                ln = line.strip()
                if not ln or ln.startswith('===') or ln.startswith('#=='):
                    continue
                ln = ln.lower()
                if ln.startswith('##'):
                    cur_base = ln[2:]
                elif cur_base == '别名':
                    kv_pair = ln.split("：：")
                    celeb_names = kv_pair[0].strip().split('|')
                    for cn in celeb_names:
                        # assert cn not in self.celeb_alias_dict
                        self.celeb_alias_dict[cn] = kv_pair[1].strip()
                elif cur_base in ['百家姓', '男性', '女性']:
                    ln = re.sub(r'\s+', ' ', re.sub(r'\t+', ' ', ln))
                    if cur_base == '百家姓':
                        self.chinese_xing_set.update(ln.split())
                    elif cur_base == '男性':
                        self.celeb_male_set.update(ln.split())
                    else:
                        self.celeb_female_set.update(ln.split())

    def _load_celeb_attr_data(self, knbase_dir):
        celeb_attr_data_file_name = os.path.join(knbase_dir, CELEB_ATTR_DATA_FILE)

        cur_attr, cur_dict = None, {}
        with open(celeb_attr_data_file_name, 'r', encoding='utf-8') as celeb_attr_data_f:
            for line in celeb_attr_data_f:
                ln = line.strip()
                if not ln or ln.startswith('==='):
                    continue
                # ln = ln.lower()
                if ln.startswith('##'):
                    cur_attr, cur_dict = ln[2:], {}
                elif ln.startswith('#='):
                    if cur_attr and cur_dict:
                        self.attr_celeb_dict[cur_attr] = cur_dict
                    cur_attr, cur_dict = ln[2:], {}
                else:
                    kv_pair = ln.split("：：")
                    celeb_names = kv_pair[0].lower().strip().split('|')
                    for cn in celeb_names:
                        assert cn not in cur_dict
                        cur_dict[cn] = kv_pair[1].strip()

        if cur_attr and cur_dict:
            self.attr_celeb_dict[cur_attr] = cur_dict

    def _load_thing_alias_wiki(self, knbase_dir):
        thing_alias_wiki_file_name = os.path.join(knbase_dir, THING_ALIAS_WIKI_FILE)

        with open(thing_alias_wiki_file_name, 'r', encoding='utf-8') as thing_alias_wiki_f:
            for line in thing_alias_wiki_f:
                ln = line.strip()
                if not ln or ln.startswith('==='):
                    continue
                elif ln.startswith('#='):
                    continue
                else:
                    kv_pair = ln.split("：：")
                    kks = kv_pair[0].strip().split('|')
                    vv = kv_pair[1].strip()
                    for kk in kks:
                        kk = kk.lower()
                        assert kk not in self.whatis_alias_dict
                        self.whatis_alias_dict[kk] = vv

    def _load_huge_obj_data(self, knbase_dir):
        huge_obj_data_file_name = os.path.join(knbase_dir, HUGE_OBJ_DATA_FILE)

        cur_cat, cur_dict = None, {}
        with open(huge_obj_data_file_name, 'r', encoding='utf-8') as huge_obj_data_f:
            for line in huge_obj_data_f:
                ln = line.strip()
                if not ln or ln.startswith('==='):
                    continue
                elif ln.startswith('#='):
                    if cur_cat and cur_dict:
                        self.huge_obj_data[cur_cat] = cur_dict
                    cur_cat, cur_dict = ln[2:], {}
                else:
                    kv_pair = ln.split("：：")
                    kks = kv_pair[0].strip().split('|')
                    vv = kv_pair[1].strip()
                    for kk in kks:
                        cur_dict[kk] = vv.replace('_OBJ_', kk)

        if cur_cat and cur_dict:
            self.huge_obj_data[cur_cat] = cur_dict

    def _load_work_attr_data(self, knbase_dir):
        work_attr_data_file_name = os.path.join(knbase_dir, WORK_ATTR_DATA_FILE)

        cur_attr, cur_dict = None, {}
        with open(work_attr_data_file_name, 'r', encoding='utf-8') as work_attr_data_f:
            for line in work_attr_data_f:
                ln = line.strip()
                if not ln or ln.startswith('==='):
                    continue
                if ln.startswith('#='):
                    if cur_attr and cur_dict:
                        self.work_attr_data[cur_attr] = cur_dict
                    cur_attr, cur_dict = ln[2:], {}
                elif cur_attr == '别称':
                    kv_pair = ln.split("：：")
                    work_names = kv_pair[0].strip().split('|')
                    for wn in work_names:
                        assert wn not in self.work_alias_dict
                        self.work_alias_dict[wn] = kv_pair[1].strip()
                else:
                    kv_pair = ln.split("：：")
                    work_name = kv_pair[0].lower().strip()
                    assert work_name not in cur_dict
                    cur_dict[work_name] = kv_pair[1].strip()

        if cur_attr and cur_dict:
            self.work_attr_data[cur_attr] = cur_dict

    def _load_org_leader_data(self, knbase_dir):
        org_leader_data_file_name = os.path.join(knbase_dir, ORG_LEADER_DATA_FILE)

        cur_cat, cur_dict = None, {}
        with open(org_leader_data_file_name, 'r', encoding='utf-8') as org_leader_data_f:
            for line in org_leader_data_f:
                ln = line.strip()
                if not ln or ln.startswith('==='):
                    continue
                elif ln.startswith('#='):
                    if cur_cat and cur_dict:
                        self.org_leader_data[cur_cat] = cur_dict
                    cur_cat, cur_dict = ln[2:], {}
                else:
                    kv_pair = ln.split("：：")
                    kks = kv_pair[0].strip().split('|')
                    vv = kv_pair[1].strip()
                    for kk in kks:
                        cur_dict[kk] = vv

            if cur_cat and cur_dict:
                self.org_leader_data[cur_cat] = cur_dict

    def _load_animal_attr_data(self, knbase_dir):
        animal_attr_file_name = os.path.join(knbase_dir, ANIMAL_ATTR_FILE)

        cur_base = None
        cur_attr, cur_dict = None, {}
        with open(animal_attr_file_name, 'r', encoding='utf-8') as animal_attr_data_f:
            for line in animal_attr_data_f:
                ln = line.strip()
                if not ln or ln.startswith('==='):
                    continue
                if ln.startswith('##'):
                    cur_base = ln[2:4]
                elif ln.startswith('#='):
                    cur_base = None
                    if cur_attr and cur_dict:
                        self.attr_animal_dict[cur_attr] = cur_dict
                    col_idx = ln.find('：')
                    cur_attr = ln[2:col_idx].strip() if col_idx > 0 else ln[2:].strip()
                    cur_dict = {}
                elif cur_base == '别名':
                    kv_pair = ln.split("：：")
                    ani_names = kv_pair[0].strip().split('|')
                    for ani in ani_names:
                        assert ani not in self.animal_alias_dict
                        self.animal_alias_dict[ani] = kv_pair[1].strip()
                elif cur_base in ['动物', '植物', '非非']:
                    ln = re.sub(r'\s+', ' ', re.sub(r'\t+', ' ', ln))
                    if cur_base == '动物':
                        self.animal_set.update(ln.split())
                    elif cur_base == '植物':
                        self.plant_set.update(ln.split())
                    else:
                        self.non_animal_non_plant_set.update(ln.split())
                else:
                    kv_pair = ln.split("：：")
                    assert kv_pair[0].strip() not in cur_dict
                    cur_dict[kv_pair[0].strip()] = kv_pair[1].strip()

        if cur_attr and cur_dict:
            self.attr_animal_dict[cur_attr] = cur_dict

    def _load_hist_events(self, knbase_dir):
        hist_events_file_name = os.path.join(knbase_dir, HIST_EVENTS_FILE)

        with open(hist_events_file_name, 'r', encoding='utf-8') as hist_events_f:
            for line in hist_events_f:
                ln = line.strip()
                if not ln or ln.startswith('#') or ln == '===':
                    continue
                kv_pair = ln.split("：：")
                kks = kv_pair[0].strip().split('|')
                vv = kv_pair[1].strip()

                for kk in kks:
                    self.hist_events_dict[kk.lower()] = vv

    def _load_sport_event_data(self, knbase_dir):
        sport_event_data_file_name = os.path.join(knbase_dir, SPORT_EVENT_DATA_FILE)

        cur_cat, cur_list = None, []
        with open(sport_event_data_file_name, 'r', encoding='utf-8') as sport_event_data_f:
            for line in sport_event_data_f:
                ln = line.strip()
                if not ln or ln.startswith('#=='):
                    continue
                if ln.startswith("##"):
                    if cur_cat and cur_list:
                        self.sport_event_data_dict[cur_cat] = cur_list
                        cur_list = []
                    cur_cat = ln[2:].strip()
                else:
                    ln = re.sub(r'\s+', ' ', re.sub(r'\t+', ' ', ln))
                    ents = ln.split()
                    last_ent = ents[-1].strip()
                    index, year = int(ents[0].strip()), int(ents[1].strip())
                    if last_ent == 'pending':
                        sp_event = SportEvent(index=index, year=year, place=ents[2].strip(),
                                              start_date=ents[3].strip(), end_date=ents[4].strip(),
                                              first=None, second=None, third=None, fourth=None,
                                              status=2, cancelled_cause=None)
                    elif last_ent.endswith('取消'):
                        sp_event = SportEvent(index=index, year=year, place=ents[2].strip(),
                                              start_date=None, end_date=None, first=None, second=None, third=None,
                                              fourth=None, status=0, cancelled_cause=last_ent)
                    else:
                        sp_event = SportEvent(index=index, year=year, place=ents[2].strip(),
                                              start_date=ents[3].strip(), end_date=ents[4].strip(),
                                              first=ents[5].strip(), second=ents[6].strip(), third=ents[7].strip(),
                                              fourth=ents[8].strip(), status=1, cancelled_cause=None)
                    cur_list.append(sp_event)

            if cur_cat and cur_list:
                self.sport_event_data_dict[cur_cat] = cur_list

    def _load_cn_en_trans_data(self, knbase_dir):
        cn_en_trans_file_name = os.path.join(knbase_dir, CN_EN_TRANS_FILE)

        with open(cn_en_trans_file_name, 'r', encoding='utf-8') as cn_en_trans_f:
            for line in cn_en_trans_f:
                ln = line.strip()
                if not ln or ln.startswith('#') or ln == '===':
                    continue
                trans_info = ln.split("：：")
                cn_names = trans_info[0].strip().split('|')
                en_trans = trans_info[1].strip()

                for cn in cn_names:
                    assert cn not in self.cn_en_trans_dict
                    self.cn_en_trans_dict[cn] = en_trans

    def _load_sp_ans(self, knbase_dir):
        sp_ans_file_name = os.path.join(knbase_dir, SP_ANS_FILE)

        with open(sp_ans_file_name, 'r', encoding='utf-8') as sp_ans_f:
            for line in sp_ans_f:
                ln = line.strip()
                if not ln or ln.startswith('#'):
                    continue
                if ln.startswith('1'):
                    self.sp_ans1_list.append(ln[2:].strip())
                elif ln.startswith('2'):
                    self.sp_ans2_list.append(ln[2:].strip())
                elif ln.startswith('3'):
                    self.sp_ans3_list.append(ln[2:].strip())
                elif ln.startswith('4'):
                    self.sp_ans4_list.append(ln[2:].strip())

    def _load_preload_pairs(self, knbase_dir):
        preload_pair_file_name = os.path.join(knbase_dir, PRELOAD_PAIR_FILE)

        conversations = []
        with open(preload_pair_file_name, 'r', encoding='utf-8') as preload_pair_f:
            samples = []
            for line in preload_pair_f:
                l = line.strip()
                if not l or l.startswith(COMMENT_LINE_STT):
                    continue

                if l == CONVERSATION_SEP:
                    if len(samples):
                        assert len(samples) % 2 == 0
                        conversations.append(samples)
                    samples = []
                else:
                    samples.append(l)

            if len(samples):  # Add the last one
                assert len(samples) % 2 == 0
                conversations.append(samples)

        for conversation in conversations:
            for i in range(0, len(conversation) - 1, 2):
                source = get_formatted_text_line(conversation[i], self.opencc_t2s, dense=True)
                target = get_formatted_text_line(conversation[i+1], self.opencc_t2s, dense=True)
                assert '_func_' not in source
                if source.find('||') > 0:
                    if source.startswith('_kv_same_rep_'):
                        source = source.replace('_kv_same_rep_', '').strip()
                        to_be_replaced = True
                    else:
                        to_be_replaced = False
                    srcs = source.split('||')
                    for src in srcs:
                        if len(src) > 2 and src[-1] in ['。', '？', '！']:
                            src = src[:-1].strip()
                        src = get_formatted_text_line(src, self.opencc_t2s, dense=True)
                        assert src not in self.preload_pairs
                        if to_be_replaced:
                            self.preload_pairs[src] = target.replace('rep__src', src)
                        else:
                            self.preload_pairs[src] = target
                else:
                    if len(source) > 2 and source[-1] in ['。', '？', '！']:
                        source = source[:-1].strip()
                    assert source not in self.preload_pairs
                    self.preload_pairs[source] = target

    def _add_poem(self, poem_id, p_title, p_writer, p_content, p_cont_exp, p_explanation):
        assert p_title != '' and p_writer != '' and p_writer[0] != '['
        # kk, _ = self.get_poem_id_by_title(p_title)
        # if kk != 0:
        #     print("Poem is already there: {}".format(p_title))
        if p_explanation != '':
            has_exp = True
            self.poems[poem_id] = Poem(title=p_title, writer=p_writer, content=p_cont_exp+p_content,
                                       explanation=p_explanation)
            p_writer, dynasty = self.parse_poem_writer(p_writer)
            if dynasty:
                if dynasty in self.poem_dynasties:
                    self.poem_dynasties[dynasty].append(poem_id)
                else:
                    self.poem_dynasties[dynasty] = [poem_id]

            if p_writer in self.poem_writers:
                self.poem_writers[p_writer].append(poem_id)
            else:
                self.poem_writers[p_writer] = [poem_id]
        else:
            has_exp = False
            self.poems_no_exp[poem_id] = Poem(title=p_title, writer=p_writer,
                                              content=p_cont_exp+p_content, explanation='')

        if has_exp:
            self.poem_lines_list.append("PHID={}".format(poem_id))
        else:
            self.poem_lines_list.append("PNID={}".format(poem_id))
        ln_cnt = len(self.poem_lines_list)

        p_cnt = re.sub(r'[；。！]', 'SP', re.sub(r'_nl_', '', p_content))
        lines = p_cnt.strip().split('SP')
        for line in lines:
            pln = line.strip()
            if pln:
                chars = list(pln.strip())
                chars_cnt = len(chars)
                tmp = []
                for ii, ch in enumerate(chars, start=1):
                    # When a sep is encountered, we check how many we have so far, and how many are left
                    if ch in '，？' and len(tmp) >= 5 and chars_cnt-ii >= 5:
                        ln = ''.join(tmp)
                        self.poem_lines_list.append(ln)
                        self.poem_lines_dict[ln] = ln_cnt
                        ln_cnt += 1
                        tmp = []
                    else:
                        tmp.append(ch)
                if tmp:
                    ln = ''.join(tmp)
                    if ln[-1] in '，？':
                        ln = ln[:-1]
                    self.poem_lines_list.append(ln)
                    self.poem_lines_dict[ln] = ln_cnt
                    ln_cnt += 1

    """
    # functions to access knowledgebase data for chat_session
    """
    def get_preload_pair_value(self, input_text, chat_sess, to_dense=True):
        new_input = input_text
        if to_dense:
            new_input = to_dense_text(new_input, remove_end_punc=True)
        if new_input in self.preload_pairs:
            return self.preload_pairs[new_input].replace('_gg_name_', chat_sess.bot_name)
        return None

    def pick_a_sp_answer(self, chat_sess, option: int):
        assert 1 <= option <= 4
        if option == 1:
            return random.choice(self.sp_ans1_list).replace('_gg_name_', chat_sess.bot_name)
        elif option == 2:
            return random.choice(self.sp_ans2_list).replace('_gg_name_', chat_sess.bot_name)
        elif option == 3:
            return random.choice(self.sp_ans3_list).replace('_gg_name_', chat_sess.bot_name)
        else:
            return random.choice(self.sp_ans4_list).replace('_gg_name_', chat_sess.bot_name)

    def get_resp_from_input_by_poem_context(self, input_text, chat_sess):
        new_input = to_dense_text(input_text, remove_end_punc=True)

        last_topic = chat_sess.last_topic
        if last_topic and last_topic.title == 'POEM' and last_topic.value:
            poem_id, ph_type = int(last_topic.value[5:]), last_topic.value[:4]
            poem = None
            if ph_type == 'PHID':
                poem = self.poems[poem_id]
            elif ph_type == 'PNID':
                poem = self.poems_no_exp[poem_id]
            if poem:
                writer, title = poem.writer, poem.title

                p_writer, dynasty = self.parse_poem_writer(writer)
                if dynasty:
                    if p_writer == '无名氏':
                        writer_title = "{}无名氏的《{}》".format(dynasty, title)
                    else:
                        ex_text = '' if len(dynasty) > 2 else '诗人'
                        writer_title = '{}{}{}的《{}》'.format(dynasty, ex_text, p_writer, title)
                elif title == '无题' and writer == '网络':
                    writer_title = '这首未具名的网路诗歌'
                else:
                    writer_title = '{}的《{}》'.format(writer, title)

                if chat_sess.poem_line_list:
                    if new_input in chat_sess.poem_line_list:
                        chat_sess.keep_topic += 1
                        chat_sess.keep_context = True
                        return "刚刚提到了这句，它是{}中的一句诗。".format(writer_title)

                    this_idx = self.poem_lines_dict.get(new_input, -1)
                    if this_idx <= 0:
                        half_line = self._get_half_line_by_full_line(new_input)
                        if half_line:
                            this_idx = self.poem_lines_dict.get(half_line, -1)
                    poem_idx, idx = -1, this_idx
                    while idx > 0:
                        idx -= 1
                        prev_line = self.poem_lines_list[idx]
                        if prev_line == "{}={}".format(ph_type, poem_id):
                            poem_idx = idx
                            break
                    if poem_idx >= 0:
                        cached_line = chat_sess.poem_line_list[0]
                        first_idx = self.poem_lines_dict.get(cached_line, -1)
                        if first_idx <= 0:
                            half_line = self._get_half_line_by_full_line(cached_line)
                            if half_line:
                                first_idx = self.poem_lines_dict.get(half_line, -1)
                        if 0 < first_idx < this_idx:
                            chat_sess.poem_line_list.append(new_input)
                        elif first_idx > this_idx:
                            chat_sess.poem_line_list.insert(0, new_input)
                        chat_sess.keep_topic += 1
                        chat_sess.keep_context = True
                        return "{}也是{}中的一句诗。".format(new_input, writer_title)

            return None

    """
    # poem and lyric related functions
    """
    def get_poem_id_from_line(self, poem_line):
        # poem_line can be a line in a poem, or the title of a poem
        # Return poem_id, dict_type (PHID or PNID)
        if poem_line:
            poem_id, ph_type = self._get_poem_id_by_half_line(poem_line)
            if poem_id > 0 and ph_type:
                return poem_id, ph_type
            for k1, v1 in self.poems.items():
                if v1.title == poem_line:
                    return int(k1), 'PHID'
            for k2, v2 in self.poems_no_exp.items():
                if v2.title == poem_line:
                    return int(k2),  'PNID'
                half_line = self._get_half_line_by_full_line(poem_line)
                if half_line:
                    return self._get_poem_id_by_half_line(half_line)
        return 0, ''

    @staticmethod
    def _get_half_line_by_full_line(poem_line):
        half_line = ''
        line_len = len(poem_line)
        if 10 <= line_len <= 15:
            if re.search(r'[，？！]', poem_line):
                parts = re.sub('[？！]', '，', poem_line).split('，')
                if parts and len(parts) == 2 and len(parts[0]) == len(parts[1]):
                    half_line = parts[0]
            elif line_len == 10:
                half_line = poem_line[:5]
            elif line_len == 14:
                half_line = poem_line[:7]
        return half_line

    def _get_poem_id_by_half_line(self, half_line):
        idx = self.poem_lines_dict.get(half_line, -1)
        while idx > 0:
            idx -= 1
            prev_line = str(self.poem_lines_list[idx])
            if prev_line.startswith('PHID'):
                return int(prev_line[5:]), 'PHID'
            elif prev_line.startswith('PNID'):
                return int(prev_line[5:]), 'PNID'
        return 0, ''

    def get_poem_id_by_first_line(self, first_line):
        idx = self.poem_lines_dict.get(first_line, -1)
        if idx > 0:
            idx -= 1
            prev_line = str(self.poem_lines_list[idx])
            if prev_line.startswith('PHID'):
                return int(prev_line[5:]), 'PHID'
            elif prev_line.startswith('PNID'):
                return int(prev_line[5:]), 'PNID'
        return 0, ''

    def get_poem_id_by_title(self, poem_title):
        if poem_title:
            for k, v in self.poems.items():
                if v.title == poem_title:
                    return k, 'PHID'
            for k, v in self.poems_no_exp.items():
                if v.title == poem_title:
                    return k, 'PNID'
        return 0, ''

    def get_lyric_id_by_title(self, lyric_title):
        if lyric_title:
            for k, v in self.lyrics.items():
                if v.title == lyric_title:
                    return k
        return 0

    @staticmethod
    def parse_poem_writer(pm_writer):
        if pm_writer.endswith(']'):
            s_idx = pm_writer.find('[')
            dynasty = pm_writer[s_idx+1:-1]
            pm_writer = pm_writer[:s_idx]
            return pm_writer, dynasty
        return pm_writer, None

    """
    # translate Chinese to English, detect question language
    """
    @staticmethod
    def translate_to_english(cn_txt: str):
        en_txt = ''
        try:
            translator = Translator(service_urls=['translate.google.com'])
            en_txt = translator.translate(cn_txt).text.strip()
        except Exception as ex:
            print("ex = {}".format(ex))
        if en_txt:
            en_words = en_txt.split()
            if len(en_words) >= 2 and en_txt[-1] not in '.?!':
                if en_words[0].lower() in ['who', 'what', 'when', 'where', 'why', 'how',
                                           'is', 'am', 'are', 'do', 'did', 'have'] or \
                        en_words[0].lower().startswith(("who'", "what'", "when'", "where'", "why'", "how'")):
                    en_txt += '?'
        return en_txt

    @staticmethod
    def detect_language_is_english(cn_txt: str):
        try:
            translator = Translator(service_urls=['translate.google.com'])
            if translator.detect(cn_txt).lang == 'en':
                return True
        except Exception as ex:
            print("ex = {}".format(ex))
        return False

    """
    # get celeb / whatis data from local and/or wiki
    """
    @staticmethod
    def remove_celeb_title(celeb_name):
        return re.sub('^(著名|知名|杰出)?的?'
                      '([大男女]?((物理|[科数文哲化])学家|文豪|伟人|才子|诗人|词人|作家|导演|编剧)|'
                      '[男女]?(艺术家|歌唱家|运动员|艺人|演员|歌手))(?!$)',
                      '', celeb_name)

    @staticmethod
    def _remove_work_name_punc(work_name):
        new_name = work_name.replace('·', '')  # 念奴娇·赤壁怀古 => 念奴娇赤壁怀古
        # 删除书名号
        if len(re.findall(r'[《<]', new_name)) == 1 and len(re.findall(r'[》>]', new_name)) == 1 \
                and new_name[-1] in '》>':
            # 一个作品（书籍、小说，影视、歌曲等）名
            if re.search(r'^(电视剧|电影|小说|诗歌|歌曲)?[《<]', new_name):
                return re.sub(r'[《》<>]', '', new_name)
        return new_name

    @staticmethod
    def _remove_work_cat(work_name):
        return re.sub('^(小说|诗歌|歌曲|电影|电视(连续)?剧|古装剧)(?!$)', '', work_name)

    def parse_loc_text_capital(self, loc_text):
        cap_idx = loc_text.find('首都')
        if cap_idx >= 0 and len(loc_text) >= 4:
            if loc_text.endswith('首都'):
                before_cap = loc_text[:-2]
                if before_cap in self.nation_cap_dict:
                    return self.nation_cap_dict[before_cap]
                elif before_cap.endswith('的') and len(before_cap) >= 3:
                    before_cap = before_cap[:-1]
                    if before_cap in self.nation_cap_dict:
                        return self.nation_cap_dict[before_cap]
            else:
                after_cap = loc_text[cap_idx+2:]
                if len(after_cap) >= 2:
                    return after_cap
        return loc_text

    def _get_converted_obj_key(self, obj_key):
        low_key = obj_key.lower()
        if low_key in self.whatis_alias_dict:  # Handle case like '美国德州' and '吉林市'
            return self.whatis_alias_dict[low_key]
        # step 1: deal with case of '首都', if it contains
        new_key = self.parse_loc_text_capital(low_key)
        # step 2: remove nation name, just a few.
        # In case a new nation name needs to be added below, you have to make sure that the whole obj_key cannot
        # be a long nation name, which is the same as the replaced one
        if not low_key.endswith('国') and not low_key.endswith('联邦'):
            new_key = re.sub('^(俄罗斯|加拿大|意大利|埃及|[中美米英法德加]国|日本国?)的?(?!$)', '', new_key)

        new_key = re.sub(r'省|市|自治区|行政区', '',
                         re.sub(r'壮族自治区|回族自治区|维吾尔自治区|维族自治区|维吾尔族自治区|特别行政区', '', new_key))
        if new_key in self.whatis_alias_dict:
            new_key = self.whatis_alias_dict[new_key]

        return new_key

    @staticmethod
    def _get_attr_info_by_key_list(kv_dict, key_list):
        for new_key in key_list:
            if new_key in kv_dict:
                return new_key, kv_dict[new_key]
        return None, None

    @staticmethod
    def _get_celeb_low_pri_return(celeb_desc, return_low_pri):
        low_pri = False
        if celeb_desc.startswith('_low_pri_'):
            low_pri = True
            celeb_desc = celeb_desc[9:].strip()
        if return_low_pri:
            return celeb_desc, low_pri
        else:
            return celeb_desc

    def is_a_person_name(self, ent_name):
        new_name = ent_name.lower()
        if new_name in self.celeb_alias_dict or new_name in self.celeb_dict:
            return True
        elif new_name in self.celeb_male_set or new_name in self.celeb_female_set:
            return True
        elif self.get_celeb_info_from_local(ent_name):
            return True
        elif self.get_first_paragraph_from_wiki(ent_name)[0]:
            return True
        return False

    def is_a_thing_name(self, ent_name):
        new_name = ent_name.lower()
        if new_name in self.whatis_alias_dict:
            return True
        elif self.get_entry_whatis_from_local(ent_name):
            return True
        else:
            for kk, vv in self.huge_obj_data.items():
                if new_name in vv:
                    return True
        isp, wiki_text = self.get_first_paragraph_from_wiki(ent_name)
        if not isp and wiki_text:
            return True
        return False

    def get_celeb_info_from_local(self, celeb_name, return_low_pri=False):
        new_name = celeb_name.lower()
        if new_name in self.celeb_dict:
            return self._get_celeb_low_pri_return(self.celeb_dict[new_name], return_low_pri)
        if new_name in self.celeb_alias_dict:
            new_name = self.celeb_alias_dict[new_name]
            if new_name in self.celeb_dict:
                return self._get_celeb_low_pri_return(self.celeb_dict[new_name], return_low_pri)
        if len(new_name) >= 3:
            new_name2 = re.sub(r'的(?=(ceo|董事长|总裁|首席执行官|老板|.*(主席|总统)))', '', new_name)
            new_name2 = self.remove_celeb_title(new_name2)
            new_name2 = self.celeb_alias_dict.get(new_name2) or new_name2
            if new_name2 != new_name:
                celeb_desc = self.celeb_dict.get(new_name2)
                if celeb_desc:
                    return self._get_celeb_low_pri_return(celeb_desc, return_low_pri)
        if return_low_pri:
            return None, False
        else:
            return None

    def get_entry_whatis_from_local(self, entry_name):
        new_name = entry_name.lower()
        # Step 1: try the original, such as 小说《失乐园》
        if new_name in self.entry_whatis_dict:
            return self.entry_whatis_dict[new_name]
        # Step 2: try the one with 书名号 removed, such as 小说失乐园, 念奴娇·赤壁怀古 => 念奴娇赤壁怀古
        new_name = self._remove_work_name_punc(new_name)
        if new_name in self.entry_whatis_dict:
            return self.entry_whatis_dict[new_name]
        # Step 3: try the one with alias converted
        if new_name in self.whatis_alias_dict:
            new_name = self.whatis_alias_dict[new_name]
            if new_name in self.entry_whatis_dict:
                return self.entry_whatis_dict[new_name]
        # Step 4: replace with owner's alias, e.g. 苏东坡的水调歌头 => 苏轼的水调歌头
        if len(new_name) >= 4 and '的' in new_name:
            de_idx = new_name.find('的')
            fst_part, sec_part = new_name[:de_idx], new_name[de_idx+1:]
            new_part = self.celeb_alias_dict.get(fst_part) or self.whatis_alias_dict.get(fst_part)
            if new_part:
                new_name = "{}的{}".format(new_part, sec_part)
                if new_name in self.entry_whatis_dict:
                    return self.entry_whatis_dict[new_name]
        # Step 5: try the one with work category removed, such as 失乐园
        if len(new_name) >= 3:
            new_name2 = self._remove_work_cat(new_name)
            if new_name2.endswith('的内容'):
                new_name2 = new_name2[:-3]
            if new_name2 != new_name:
                whis_desc = self.entry_whatis_dict.get(new_name2)
                if whis_desc:
                    return whis_desc
        return None

    def get_first_paragraph_from_wiki(self, key):
        ori_key = key.lower()
        if ori_key in self.wiki_first_para_neg_set:
            return False, None

        search_key = ori_key
        if search_key in self.celeb_alias_dict:
            search_key = self.celeb_alias_dict[search_key]
        if search_key in self.whatis_alias_dict:
            search_key = self.whatis_alias_dict[search_key]
        _, isp, wp = get_first_paragraph(search_key)

        if not wp and len(search_key) >= 3:
            # for case of celeb name
            new_key = self.remove_celeb_title(search_key)
            if new_key in self.celeb_alias_dict:
                new_key = self.celeb_alias_dict[new_key]
            if new_key != search_key:
                _, isp, wp = get_first_paragraph(new_key)
                search_key = new_key
            # for case of 作品名称，比如歌名，电影名
            elif search_key[:2] in ['小说', '诗歌', '歌曲', '电影'] or search_key.startswith('电视剧'):
                if search_key.startswith('电视剧'):
                    new_key = search_key[3:]
                    extra = search_key[:3]
                else:
                    new_key = search_key[2:]
                    extra = search_key[:2]
                if new_key[0] == '《' and new_key[-1] == '》':
                    new_key = new_key[1:-1]
                _, isp, wp = get_first_paragraph('{}_({})'.format(new_key, extra))
                search_key = new_key
                if not wp:
                    _, isp, wp = get_first_paragraph(search_key)
        if not wp:
            self.wiki_first_para_neg_set.add(ori_key)
            return False, None

        lower_key = search_key.lower()
        print("lower_key = {}".format(lower_key))
        if isp:  # is a person entry
            gender = self._get_celeb_gender(lower_key)
            if gender == '男':
                wp = wp.replace('_func_set_pronoun_para0_flw_para1_ta1_ta2', '_func_set_pronoun_para0_flw_para1_ta1')
            elif gender == '女':
                wp = wp.replace('_func_set_pronoun_para0_flw_para1_ta1_ta2', '_func_set_pronoun_para0_flw_para1_ta2')
            self.celeb_dict[lower_key] = wp
        else:
            self.entry_whatis_dict[lower_key] = wp
        return isp, wp

    def get_person_entry_dict_from_wiki(self, person_name):
        celeb_dict = self.celeb_attr_cache_dict.get(person_name)
        if not celeb_dict and person_name not in self.wiki_person_entry_neg_set:  # connect to wiki now
            celeb_dict = get_person_entry_dict(person_name)
            if celeb_dict:
                self.celeb_attr_cache_dict[person_name] = celeb_dict
            else:
                self.wiki_person_entry_neg_set.add(person_name)
        return celeb_dict

    def get_thing_entry_dict_from_wiki(self, obj_key, key_conversion=False):
        new_key = self._get_converted_obj_key(obj_key) if key_conversion else obj_key

        obj_attr_dict = self.thing_attr_cache_dict.get(new_key)
        if not obj_attr_dict and new_key not in self.wiki_thing_entry_neg_set:  # connect to wiki now
            obj_attr_dict = get_thing_entry_dict(new_key)
            if obj_attr_dict:
                self.thing_attr_cache_dict[new_key] = obj_attr_dict  # add to memory for the whole system up-time
            else:
                self.wiki_thing_entry_neg_set.add(new_key)
        return obj_attr_dict, new_key

    def get_work_entry_dict_from_wiki(self, work_key_list):
        # new_key = self.work_alias_dict[work_key] if work_key in self.work_alias_dict else work_key
        for work_key in work_key_list:
            obj_attr_dict = self.thing_attr_cache_dict.get(work_key)
            if not obj_attr_dict and work_key not in self.wiki_thing_entry_neg_set:  # connect to wiki now
                obj_attr_dict = get_thing_entry_dict(work_key)
                if obj_attr_dict:
                    self.thing_attr_cache_dict[work_key] = obj_attr_dict  # add to memory for the whole system up-time
                else:
                    self.wiki_thing_entry_neg_set.add(work_key)
            if obj_attr_dict:
                return obj_attr_dict
        return None

    """
    # get celeb attr data
    """
    def _get_celeb_gender(self, celeb_name):
        new_name = self.celeb_alias_dict.get(celeb_name) or celeb_name

        if new_name in self.celeb_female_set:
            return '女'
        elif new_name in self.celeb_male_set:
            return '男'
        return None

    def get_celeb_attr_info_text(self, celeb_name, attr_key, in_question):
        new_key = CELEB_ATTR_CONVERT_MAP.get(attr_key)
        if not new_key:
            return None

        if new_key in ['名字', '原名', '笔名']:
            dense_quest = to_dense_text(in_question)
            whose_bm = False
            if new_key in ['原名', '笔名'] and re.search(r'(哪[个位](作家|作者|[文诗词]?人)?|谁)[的之]笔名', dense_quest):
                new_key = '原名'  # does not matter if it was '原名'
                whose_bm = True
            from_file, ret_val, ret_key = self._get_celeb_attr_info_from_all(celeb_name, new_key)
            if ret_val:
                if from_file and ret_val.endswith('。'):
                    return ret_val
                elif from_file:
                    extra = random.choice(["这我知道，", "这我记得，", "这我当然知道，", ""])
                    if new_key == '名字':
                        return random.choice(["{}{}名叫{}。".format(extra, celeb_name, ret_val),
                                              "{}{}的名字叫{}呀。".format(extra, celeb_name, ret_val), ])
                    elif new_key == '原名' and whose_bm:
                        return "{}{}是{}的笔名。".format(extra, celeb_name, ret_val)
                    else:
                        return "{}{}的{}叫{}。".format(extra, celeb_name, new_key, ret_val)
                else:
                    return "{}的{}：{}（信息提取自维基百科）".format(celeb_name, ret_key, ret_val)
            elif new_key in ['名字', '原名']:
                return "{}的{}就是{}啊，哈哈哈{{捂脸}}{{捂脸}}".format(celeb_name, new_key, celeb_name)

        new_name = self.celeb_alias_dict.get(celeb_name) or celeb_name

        if new_key == '姓氏':
            ret_val = self.attr_celeb_dict['姓氏'].get(new_name)
            if ret_val:
                if ret_val.endswith('。'):
                    return ret_val
                else:
                    extra = random.choice(["这还用问？", "这我知道，", ""])
                    return "{}{}姓{}。".format(extra, new_name, ret_val)
            else:
                xing_name = self.attr_celeb_dict['名字'].get(new_name) or new_name
                name_len = len(xing_name)
                if name_len in [3, 4] and xing_name[:2] in self.chinese_xing_set:
                    if xing_name == new_name:
                        return random.choice(["{}应该是复姓{}。".format(xing_name, xing_name[:2]),
                                              "我觉得{}是复姓{}吧。".format(xing_name, xing_name[:2]), ])
                    else:
                        return "{}{}，所以是复姓{}。".format(new_name, xing_name, xing_name[:2])
                elif name_len in [2, 3] and xing_name[:1] in self.chinese_xing_set:
                    if xing_name == new_name:
                        return random.choice(["{}姓{}啊。".format(xing_name, xing_name[:1]),
                                              "我觉得{}是姓{}吧。".format(xing_name, xing_name[:1]), ])
                    else:
                        return "{}{}，所以是姓{}。".format(new_name, xing_name, xing_name[:1])
            return random.choice(["晕，关注名人的姓氏干嘛？好八卦哟。",
                                  "矮油，八卦名人的姓氏多没意思啊，在下真的不擅长呢。", ])
        if new_key == '字号':
            ret_val = self.attr_celeb_dict['字号'].get(new_name)
            if ret_val:
                if ret_val.endswith('。'):
                    return ret_val
                else:
                    extra = random.choice(["这我记得，", "这我知道，", ""])
                    return "{}{}{}。".format(extra, new_name, ret_val)
            return random.choice(["晕，关注名人的字号干嘛？好八卦哟。",
                                  "矮油，我真不熟悉名人的字号呢。毕竟这爱好也太高雅了吧，哈哈哈。", ])

        gender = None
        if new_name in self.celeb_female_set:
            gender = '女'
        elif new_name in self.celeb_male_set:
            gender = '男'

        if new_key == '年龄':
            _, s_date, _ = self._get_celeb_attr_info_from_all(new_name, '生日')
            _, e_date, _ = self._get_celeb_attr_info_from_all(new_name, '逝世日期')
            if s_date:
                s_year = self._extract_person_bir_dec_year(s_date)
                if s_year > 0:
                    if e_date:
                        e_year = self._extract_person_bir_dec_year(e_date)
                        if e_year > 0:
                            age = e_year - s_year
                            return "这能算出来，{}享年{}岁（生于{}，逝世于{}）。".format(new_name, age, s_date, e_date)
                        else:
                            return "这都是过去的人了吧，只知道{}的生日是{}。".format(new_name, s_date)
                    else:
                        cur_year = dt.datetime.now().year
                        age = cur_year - s_year
                        if age < 100:
                            extra = random.choice(["这不难算出来，", "这我知道，", ""])
                            return "{}{}今年应该是{}岁了吧（生日：{}）。".format(extra, new_name, age, s_date)
                        else:
                            return "{}的近况我不了解，只知道其生日是{}。".format(new_name, s_date)
            if gender == '女':
                return "矮油，关心美女的年龄呀，这可不是我的强项呢！"
            elif gender == '男':
                return "晕，我也不关心{}啊，所以他的年龄我还真的记不住呢。".format(new_name)
            # else:
            #     return "矮油，这年龄我哪记得住呀？现在的小学考试题怎么这么偏呢？"
        elif new_key == '性别':
            gender_dict = self.attr_celeb_dict['性别']
            if new_name in gender_dict:
                return gender_dict[new_name]
            elif gender in ['男', '女']:
                return random.choice(["这我知道，{}是{}的。".format(new_name, gender),
                                      "这我记得，{}是{}的。".format(new_name, gender),
                                      "{}我熟悉，是{}的呀。".format(new_name, gender),
                                      "{}当然是{}的。".format(new_name, gender),])
            else:
                return random.choice(["绑回家检查一下你就知道啦，哈哈哈。",
                                      "领回家研究研究你就知道啦，哈哈哈。"])

        extra = random.choice(["这信息很容易查到，", "这我知道，", "这在网上不难查到，", "这我记得，", ""])
        from_file, ret_val, ret_key = self._get_celeb_attr_info_from_all(new_name, new_key)
        if ret_val:
            if from_file and ret_val.endswith('。'):
                return ret_val
            elif from_file and new_key == '户籍':
                return "{}{}{}。".format(extra, new_name, ret_val)
            elif from_file:
                return "{}{}的{}是{}。".format(extra, new_name, ret_key, ret_val)
            elif new_key == '逝世日期':
                _, d_place, _ = self._get_celeb_attr_info_from_all(new_name, 'DEATH_PLACE')
                if d_place:
                    return "{}于{}在{}逝世（信息提取自维基百科）".format(new_name, ret_val, d_place)
                else:
                    return "{}的{}：{}（信息提取自维基百科）".format(new_name, ret_key, ret_val)
            else:
                return "{}的{}：{}（信息提取自维基百科）".format(new_name, ret_key, ret_val)
        elif new_key == '户籍':
            _, nat, _ = self._get_celeb_attr_info_from_all(new_name, '国籍')
            _, gmq, _ = self._get_celeb_attr_info_from_all(new_name, '公民权')
            _, b_place, _ = self._get_celeb_attr_info_from_all(new_name, 'BIRTH_PLACE')
            _, w_place, _ = self._get_celeb_attr_info_from_all(new_name, '出道地点')
            _, loc, _ = self._get_celeb_attr_info_from_all(new_name, '籍贯')
            _, l_place, _ = self._get_celeb_attr_info_from_all(new_name, '居住地')
            _, b_date, _ = self._get_celeb_attr_info_from_all(new_name, '生日')
            if (nat or gmq) and b_place:
                base_text = None
                b_place = b_place.replace('英属香港', '香港')
                b_year, b_month = 0, 0
                if b_date:
                    b_year, b_month = self._extract_person_bir_dec_year_month(b_date)

                if nat == '中华人民共和国':
                    base_text = "{}来自中国内地，出生于{}".format(new_name, b_place)
                elif nat in ['中华人民共和国（香港）', '香港（中华人民共和国）']:
                    if b_place.find('香港') >= 0:
                        base_text = "{}来自中国香港，出生于{}".format(new_name, b_place)
                    else:
                        base_text = "{}出生于{}".format(new_name, b_place)
                elif (nat and re.search(r'中华民国|大清([（，]|$)', nat)) or \
                        (gmq and re.search(r'中华民国|大清([（，]|$)', gmq)):
                    if b_place.startswith('大清'):
                        base_text = "{}生于中国清代，出生地是{}".format(new_name, re.sub(r'^大清', '', b_place))
                    elif b_place.startswith('中华民国'):
                        if 0 < b_year < 1949 or (b_year == 1949 and 0 < b_month <= 9):
                            base_text = "{}生于民国时期，出生地是{}".format(new_name, re.sub(r'^中华民国', '', b_place))
                    if not base_text:
                        base_text = "{}出生于{}".format(new_name, b_place)
                elif nat:
                    base_text = "{}的国籍是{}，出生于{}".format(new_name, nat, b_place)

                if base_text:
                    if w_place and w_place.find('中华民国') < 0:
                        w_place = w_place.replace('英属香港', '香港')
                        if nat:
                            w_place = w_place.replace(nat, '')
                        if w_place and w_place not in b_place:  # present only when it is different
                            base_text += "，出道地点是{}".format(w_place)

                    if loc and loc.find('中华民国') < 0:
                        loc = loc.replace('英属香港', '香港')
                        if loc not in b_place: # present only when it is different
                            base_text += "，其籍贯为{}".format(loc)
                    elif l_place and l_place.find('中华民国') < 0:
                        l_place = l_place.replace('英属香港', '香港')
                        if nat:
                            l_place = l_place.replace(nat, '')
                        if l_place and l_place not in b_place:  # present only when it is different
                            base_text += "，其居住地为{}".format(l_place)

                if not base_text and gmq:
                    base_text = "{}出生于{}，曾拥有如下国家或地区的公民权：{}".format(new_name, b_place, gmq)
                if base_text:
                    return base_text + "（信息提取自维基百科）"
            elif b_place:
                if gender == '女':
                    pron_call = '她'
                elif gender == '男':
                    pron_call = '他'
                else:
                    pron_call = '其'
                base_text = "对{}我了解不多，只知道{}出生于{}。".format(new_name, pron_call, b_place)
                return base_text
            return "晕，这不是为难我嘛，我没有{}的出生地或所在地信息呢。".format(new_name)
        return None

    def get_celeb_relation_info_text(self, celeb_name, attr_key, in_quest):
        new_key = CELEB_ATTR_CONVERT_MAP.get(attr_key)
        if not new_key:
            return None, None

        new_name = self.celeb_alias_dict.get(celeb_name) or celeb_name

        gender = None
        if new_name in self.celeb_female_set:
            gender = '女'
            if new_key == '配偶':
                new_key = '老公'
        elif new_name in self.celeb_male_set:
            gender = '男'
            if new_key == '配偶':
                new_key = '老婆'

        extra = random.choice(["这信息很容易查到，", "这我知道，", "这在网上不难查到，", "这我记得，", ""])
        name_in_quest = get_formatted_text_line(celeb_name, self.opencc_t2s, dense=False)
        owners = r'({}|[他她])\s([的之]\s|有\s(没\s(有\s)?)?)?'.format(name_in_quest)

        # Step 1: Handle other relations first using regex
        if new_key in ['老公', '老婆', '爸爸', '父亲', '妈妈', '母亲']:
            if re.search(r'{}祖\s父'.format(owners), in_quest):
                new_key = '祖父'
            elif re.search(r'{}祖\s母'.format(owners), in_quest):
                new_key = '祖母'
            if re.search(r'{}(岳\s[父丈]|(老\s)?丈\s人)'.format(owners), in_quest):
                new_key = '岳父'
            elif re.search(r'{}(岳\s母|丈\s母\s娘)'.format(owners), in_quest):
                new_key = '岳母'
            elif re.search(r'{}(公\s[公爹])'.format(owners), in_quest):
                new_key = '公爹'
            elif re.search(r'{}(婆\s[婆母])'.format(owners), in_quest):
                new_key = '婆母'
        elif new_key == '其他':
            # The first four are for prediction errors
            if re.search(r'{}(父\s亲|爸\s爸|老\s爹)'.format(owners), in_quest):
                new_key = '父亲'
            elif re.search(r'{}(母\s亲|妈\s妈|老\s娘)'.format(owners), in_quest):
                new_key = '母亲'
            elif re.search(r'{}(老\s公|丈\s夫|先\s生)'.format(owners), in_quest):
                new_key = '老公'
            elif re.search(r'{}(老\s婆|妻\s子|媳\s妇|太\s太)'.format(owners), in_quest):
                new_key = '老婆'
            elif re.search(r'{}(爱\s人|配\s偶)'.format(owners), in_quest):
                new_key = '配偶'
            elif re.search(r'{}(祖\s父|爷\s爷)'.format(owners), in_quest):
                new_key = '祖父'
            elif re.search(r'{}(祖\s母|奶\s奶)'.format(owners), in_quest):
                new_key = '祖母'
            elif re.search(r'{}(岳\s[父丈]|(老\s)?丈\s人)'.format(owners), in_quest):
                new_key = '岳父'
            elif re.search(r'{}(岳\s母|丈\s母\s娘)'.format(owners), in_quest):
                new_key = '岳母'
            elif re.search(r'{}(公\s[公爹])'.format(owners), in_quest):
                new_key = '公爹'
            elif re.search(r'{}(婆\s[婆母])'.format(owners), in_quest):
                new_key = '婆母'
            elif re.search(r'({}|几\s个\s)(儿\s子)'.format(owners), in_quest):
                new_key = '儿子'
            elif re.search(r'({}|几\s个\s)(女\s儿)'.format(owners), in_quest):
                new_key = '女儿'
            elif re.search(r'({}|几\s个\s)(小\s孩|孩\s子|[儿子]\s女)'.format(owners), in_quest):
                new_key = '子女'
            elif re.search(r'({}|几\s个\s)(哥\s哥|兄\s长)'.format(owners), in_quest):
                new_key = '哥哥'
            elif re.search(r'({}|几\s个\s)([兄弟]\s弟)'.format(owners), in_quest):
                new_key = '弟弟'
            elif re.search(r'({}|几\s个\s)(姐\s姐)'.format(owners), in_quest):
                new_key = '姐姐'
            elif re.search(r'({}|几\s个\s)(妹\s妹)'.format(owners), in_quest):
                new_key = '妹妹'
            elif re.search(r'{}(师\s[父傅]|恩\s师)'.format(owners), in_quest):
                new_key = '师父'
            elif re.search(r'({}|(几\s个\s|哪\s些\s).*)(徒\s[弟儿]|弟\s子)'.format(owners), in_quest) \
                    and not re.search(r'((谁\s|哪\s[个位]\s)(人\s)?)的\s(徒\s[弟儿]|弟\s子)', in_quest):
                new_key = '弟子'

            print("##### new_name = #{}#, new_key = #{}#".format(new_name, new_key))

        if new_key in ['祖父', '祖母', '岳父', '岳母', '公爹', '婆母', '哥哥', '弟弟', '姐姐', '妹妹',
                       '儿子', '女儿', '子女', '师父', '弟子']:
            tmp_dict = self.attr_celeb_dict.get(new_key)
            if tmp_dict:
                if new_name in tmp_dict:
                    ret_val = tmp_dict[new_name]
                    if ret_val.endswith('。'):
                        return ret_val, new_key
                    else:
                        return "{}{}的{}是{}。".format(extra, new_name, new_key, ret_val), new_key
            elif new_key == '子女':
                son_dict, dau_dict = self.attr_celeb_dict['儿子'], self.attr_celeb_dict['女儿']
                v1, v2 = son_dict.get(new_name), dau_dict.get(new_name)
                # NO PERIOD (。) type of data is supported in the v1 and v2
                if v1 and v2:
                    return "{}{}的儿子：{}；女儿：{}。".format(extra, new_name, v1, v2), new_key
                elif v1:
                    return "{}{}的儿子：{}。".format(extra, new_name, v1), new_key
                elif v2:
                    return "{}{}的女儿：{}。".format(extra, new_name, v2), new_key

            if new_key in ['岳父', '岳母', '公爹', '婆母', '哥哥', '弟弟', '姐姐', '妹妹', '儿子', '女儿', '子女']:
                celeb_dict = self.get_person_entry_dict_from_wiki(new_name)
                if celeb_dict:
                    if new_key == '岳父':
                        _, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['岳父', '妻之父', '元配之父'])
                    elif new_key == '岳母':
                        _, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['岳母', '妻之母', '元配之母'])
                    elif new_key == '公爹':
                        _, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['公爹', '夫之父'])
                    elif new_key == '婆母':
                        _, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['婆母', '夫之母'])
                    elif new_key == '哥哥':
                        new_key, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['兄', '兄长', '兄弟', '弟兄'])
                        if ret_val and new_key == '兄':
                            new_key = '哥哥'
                    elif new_key == '弟弟':
                        new_key, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['弟', '弟弟', '兄弟', '弟兄'])
                        if ret_val and new_key == '弟':
                            new_key = '弟弟'
                    elif new_key == '姐姐':
                        new_key, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['姐', '姐姐', '姐妹', '姊妹'])
                        if ret_val and new_key == '姐':
                            new_key = '姐姐'
                    elif new_key == '妹妹':
                        new_key, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['妹', '妹妹', '姐妹', '姊妹'])
                        if ret_val and new_key == '妹':
                            new_key = '妹妹'
                    else:
                        if new_key == '儿子':
                            _, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['儿子', '子'])
                        elif new_key == '女儿':
                            _, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['女儿', '女'])
                        else:  # cases of '儿女', '子女', initialize it to None
                            ret_val = None
                        if not ret_val:
                            new_key, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['儿女', '子女'])
                            if ret_val and ret_val[0] in '123456789一二两三四五六七八九':
                                new_key, ret_val = None, None
                            elif not ret_val:
                                v1 = celeb_dict.get('子') or celeb_dict.get('儿子')
                                v2 = celeb_dict.get('女') or celeb_dict.get('女儿')
                                if v1 and v2:
                                    return "{}的儿子：{}；女儿：{}（信息提取自维基百科）".format(new_name, v1, v2), new_key
                                elif v1:
                                    return "{}的儿子：{}（信息提取自维基百科）".format(new_name, v1), new_key
                                elif v2:
                                    return "{}的女儿：{}（信息提取自维基百科）".format(new_name, v2), new_key
                    if ret_val:
                        return "{}的{}：{}（信息提取自维基百科）".format(new_name, new_key, ret_val), new_key
            return random.choice(["干嘛要关注名人的家庭隐私呢？这我真不擅长啊。",
                                  "为啥要这么关注名人的家庭关系呢？这我真不擅长呢。"]), new_key

        # Step 2: Handle other special cases
        if (new_key in ['男友', '老公'] and gender == '男') or (new_key in ['女友', '老婆'] and gender == '女'):
            wife_pat = re.compile(r'谁\s嫁\s[给了]|{}\s(娶\s了|(的\s)?(媳\s妇|太\s太))'.format(name_in_quest))
            hsbd_pat = re.compile(r'谁\s娶\s了|{}\s(嫁\s[给了]|(的\s)?(先\s生|丈\s夫))'.format(name_in_quest))
            if new_key == '老公' and re.search(wife_pat, in_quest):
                new_key = '老婆'  # fix model prediction error
            elif new_key == '老婆' and re.search(hsbd_pat, in_quest):
                new_key = '老公'
            else:
                return random.choice([
                    "我所了解的{}是{}的，也不应该有{}呀，哈哈哈。".format(new_name, gender, new_key),
                    "你是跟我开玩笑吧，{}应该是{}人呀，也有{}吗？".format(new_name, gender, new_key),
                    "现在的{}人也开始有{}了吗？这倒真是新鲜事呢。".format(gender, new_key),
                ]), new_key
        elif new_key == '前任':
            wife_pat = re.compile(r'{}\s(的\s)?(前\s(任\s)?(妻\s子?|老\s婆|媳\s妇|太\s太))'.format(name_in_quest))
            hsbd_pat = re.compile(r'{}\s(的\s)?(前\s(任\s)?((丈\s)?夫|老\s公|先\s生))'.format(name_in_quest))
            if re.search(wife_pat, in_quest):
                new_key = '前妻'
            elif re.search(hsbd_pat, in_quest):
                new_key = '前夫'
            elif gender == '男':
                new_key = '前妻'
            elif gender == '女':
                new_key = '前夫'
            if (new_key == '前夫' and gender == '男') or (new_key == '前妻' and gender == '女'):
                return random.choice([
                    "我所了解的{}是{}的，也不应该有{}呀，哈哈哈。".format(new_name, gender, new_key),
                    "你是跟我开玩笑吧，{}应该是{}人呀，也有{}吗？".format(new_name, gender, new_key),
                    "现在的{}人也开始有{}了吗？这倒真是新鲜事呢。".format(gender, new_key),
                ]), new_key

        # Step 3: Handle most general cases
        from_file, ret_val, ret_key = self._get_celeb_attr_info_from_all(new_name, new_key)
        if not ret_val:
            if new_key in ['男友', '前夫']:
                from_file, ret_val, ret_key = self._get_celeb_attr_info_from_all(new_name, '老公')
            elif new_key == '老公':
                from_file, ret_val, ret_key = self._get_celeb_attr_info_from_all(new_name, '前夫')
            elif new_key in ['女友', '前妻']:
                from_file, ret_val, ret_key = self._get_celeb_attr_info_from_all(new_name, '老婆')
            elif new_key == '老婆':
                from_file, ret_val, ret_key = self._get_celeb_attr_info_from_all(new_name, '前妻')

        if ret_val:
            if from_file and ret_val.endswith('。'):
                return ret_val, new_key
            elif new_key in ['男友', '前夫', '老公', '女友', '前妻', '老婆']:
                if (new_key in ['男友', '前夫'] and ret_key == '老公') or new_key == '男友':
                    return "{}有没{}我不清楚，不过我知道她的{}：{}".format(new_name, new_key, ret_key, ret_val), new_key
                elif new_key == '老公' and ret_key == '前夫':
                    return "{}的现任老公我不清楚，不过我知道她前夫是{}。".format(new_name, ret_val), new_key
                elif (new_key in ['女友', '前妻'] and ret_key == '老婆') or new_key == '女友':
                    return "{}有没{}我不清楚，不过我知道他的{}：{}".format(new_name, new_key, ret_key, ret_val), new_key
                elif new_key == '老婆' and ret_key == '前妻':
                    return "{}的现任老婆我不清楚，不过我知道他前妻是{}。".format(new_name, ret_val), new_key
            if from_file:
                return "{}{}的{}是{}。".format(extra, new_name, ret_key, ret_val), new_key
            else:
                return "{}的{}：{}（信息提取自维基百科）".format(new_name, ret_key, ret_val), new_key
        return None, new_key

    def get_celeb_work_info_text(self, celeb_name, attr_key, in_quest):
        new_key = CELEB_ATTR_CONVERT_MAP.get(attr_key)
        if new_key:
            new_name = self.celeb_alias_dict.get(celeb_name) or celeb_name
            from_file, ret_val = self._get_celeb_work_info_from_all(new_name, new_key, in_quest)
            if ret_val:
                ret_text, chatted = '', False
                ran_text = random.choice(['还有所耳闻', '还有些了解', '还是比较关注的', '还算熟悉', '还是知道的'])
                if new_key in ['诗歌', '小说', '书籍']:
                    if re.search(r'[读念看]\s过', in_quest):
                        ret_text = random.choice([
                            "我不爱学习的，更别说读课外书了。不过{}我{}。".format(new_name, ran_text),
                            "我不爱读书，但{}的{}我倒真看过，所以还有些了解。".format(new_name, new_key), ])
                        chatted = True
                elif new_key == '歌曲':
                    if re.search(r'听\s过', in_quest):
                        ret_text = random.choice([
                            "我喜欢听歌，但很少有时间仔细欣赏。当然{}我{}。".format(new_name, ran_text),
                            "{}的歌我还真听过一些，所以也算有些了解。".format(new_name), ])
                        chatted = True
                elif new_key in ['电影', '电视剧']:
                    if re.search(r'看\s过', in_quest):
                        ret_text = random.choice([
                            "我喜欢看影视剧，但苦于太忙，所以看得很少。不过{}我{}。".format(new_name, ran_text),
                            "{}的影视剧我还真看过一些，所以算比较熟悉。".format(new_name), ])
                        chatted = True
                if from_file and ret_val.endswith('。'):
                    if chatted and ret_val.startswith(new_name):
                        name_len = len(new_name)
                        if new_name in self.celeb_female_set:
                            ret_val = '她' + ret_val[name_len:]
                        elif new_name in self.celeb_male_set:
                            ret_val = '他' + ret_val[name_len:]
                    ret_text += ret_val
                elif chatted:
                    if ret_val.startswith(new_name):
                        name_len = len(new_name)
                        if new_name in self.celeb_female_set:
                            ret_val = '她' + ret_val[name_len:]
                        elif new_name in self.celeb_male_set:
                            ret_val = '他' + ret_val[name_len:]
                    ret_text += ret_val + '。'
                elif from_file:
                    ret_text += ret_val + '。'
                else:
                    ret_text = ret_val + "（信息提取自维基百科）"
                return ret_text
            else:
                if new_key in ['诗歌', '小说', '书籍'] and re.search(r'[读念看]\s过', in_quest):
                    return "我本就不爱学习，再加上忙，所以真的很少读书呢。"
                elif new_key == '歌曲' and re.search(r'听\s过', in_quest):
                    return "我是喜欢听歌，可惜太忙了，没时间欣赏呢。"
                elif new_key in ['电影', '电视剧'] and re.search(r'看\s过', in_quest):
                    return "我是很喜欢看{}，但实在太忙，所以真没时间看呢。".format(new_key)
        return None

    def _get_celeb_work_info_from_all(self, new_name, new_key, in_quest):
        # From file
        if new_key == '代表作':
            f_dict = self._get_celeb_work_from_all_file_cats(
                ['代表作', '书籍', '诗歌', '小说', '音乐专辑', '歌曲', '电影', '电视剧'], new_name)
            f_count = len(f_dict)
            if f_count > 1:
                out_text = "{}的代表作较多，其中".format(new_name)
                ii = 0
                for kk, vv in f_dict.items():
                    if ii > 0:
                        out_text += "；"
                    out_text += "{}有{}".format(kk, vv)
                    ii += 1
                return True, out_text
            elif f_count == 1:
                f_cat = list(f_dict.keys())[0]
                f_val = f_dict.get(f_cat)
                if f_val.endswith('。'):
                    out_text = f_val
                elif f_cat == '书籍':
                    out_text = "{}的代表作有{}".format(new_name, f_val)
                elif f_cat == '音乐专辑':
                    out_text = "{}的作品主要为音乐，著名的专辑有{}".format(new_name, f_val)
                else:
                    out_text = "{}的代表作主要为{}，著名的有{}".format(new_name, f_cat, f_val)
                return True, out_text
        elif new_key == '书籍':
            attr_dict = self.attr_celeb_dict['书籍']
            ret_val = attr_dict.get(new_name)
            if ret_val:
                out_text = ret_val if ret_val.endswith('。') else "{}的著名书籍有{}".format(new_name, ret_val)
                return True, out_text
            attr_dict = self.attr_celeb_dict['小说']
            ret_val = attr_dict.get(new_name)
            if ret_val:
                out_text = ret_val if ret_val.endswith('。') else "{}的著名小说有{}".format(new_name, ret_val)
                return True, out_text
        elif new_key == '歌曲':
            f_dict = self._get_celeb_work_from_all_file_cats(['音乐专辑', '歌曲'], new_name)
            v1_text = f_dict.get('音乐专辑')
            v2_text = f_dict.get('歌曲')

            if v1_text and v2_text and re.search(r'音\s乐', in_quest) and v1_text[-1] != '。' and v2_text[-1] != '。':
                out_text = "{}的代表作包括专辑：{}；歌曲：{}".format(new_name, v1_text, v2_text)
                return True, out_text
            elif v2_text:
                out_text = v2_text if v2_text.endswith('。') else "{}的歌曲代表作有{}".format(new_name, v2_text)
                return True, out_text
            elif v1_text:
                out_text = v1_text if v1_text.endswith('。') else "{}的著名专辑有{}".format(new_name, v1_text)
                return True, out_text
        else:
            attr_dict = self.attr_celeb_dict.get(new_key)
            if attr_dict:
                ret_val = attr_dict.get(new_name)
                if ret_val:
                    if ret_val.endswith('。'):
                        out_text = ret_val
                    elif new_key == '音乐专辑':
                        out_text = "{}的著名专辑有{}".format(new_name, ret_val)
                    else:
                        out_text = "{}的{}代表作有{}".format(new_name, new_key, ret_val)
                    return True, out_text

        # From cache, then wiki
        celeb_dict = self.get_person_entry_dict_from_wiki(new_name)
        if celeb_dict:
            if new_key in ['诗歌', '小说']:
                if new_key == '诗歌':
                    f_dict = self._get_celeb_work_from_all_wiki_cats(celeb_dict, ['诗歌', '诗集', '诗歌集', '诗词集'])
                else:
                    f_dict = self._get_celeb_work_from_all_wiki_cats(celeb_dict, ['小说', '小说集'])
                f_count = len(f_dict)
                if f_count > 1:
                    out_text = "{}著有很多{}，代表作包括".format(new_key, new_name)
                    ii = 0
                    for kk, vv in f_dict.items():
                        if ii > 0:
                            out_text += "；"
                        out_text += "{}{}".format(kk, vv)
                        ii += 1
                    return False, out_text
                elif f_count == 1:
                    f_cat = list(f_dict.keys())[0]
                    if f_cat == new_key:
                        out_text = "{}创作过很多{}，代表作有{}".format(new_name, new_key, f_dict.get(f_cat))
                    else:  # '诗集', '诗歌集', '诗词集'
                        out_text = "{}创作过很多{}，著有{}{}".format(new_name, new_key, f_cat, f_dict.get(f_cat))
                    return False, out_text
                elif new_key == '小说':
                    ti_cai = celeb_dict.get('体裁')
                    ret_val = celeb_dict.get('代表作品')
                    if ti_cai and ret_val:
                        ret_val = self._replace_last_comma_for_work_name(ret_val)
                        period_mat = re.search(r'，', ti_cai)  # 如有顿号，已经在wikiscraper.py中被替换
                        if not period_mat and ti_cai.endswith('小说'):
                            out_text = "{}创作过很多{}，代表作有{}".format(new_name, ti_cai, ret_val)
                            return False, out_text
                        if period_mat:
                            ti_cai_list = ti_cai.split('，')
                            if ti_cai_list[0].strip().endswith('小说'):
                                ti_cai = '及'.join(ti_cai.rsplit('，', 1))
                                out_text = "{}创作过很多{}，代表作有{}".format(new_name, ti_cai, ret_val)
                                return False, out_text
            elif new_key == '书籍':
                f_dict = self._get_celeb_work_from_all_wiki_cats(
                    celeb_dict, ['小说', '小说集', '散文', '散文集', '诗歌', '诗集', '诗歌集', '诗词集'])
                f_count = len(f_dict)
                if f_count > 1:
                    out_text = "{}著有很多文学作品，代表作包括".format(new_name)
                    ii = 0
                    for kk, vv in f_dict.items():
                        if ii > 0:
                            out_text += "；"
                        out_text += "{}{}".format(kk, vv)
                        ii += 1
                    return False, out_text
                elif f_count == 1:
                    f_cat = list(f_dict.keys())[0]
                    if f_cat in ['小说', '小说集']:
                        out_text = "{}的主要创作体裁为小说，".format(new_name)
                    elif f_cat in ['散文', '散文集']:
                        out_text = "{}的主要创作体裁为散文，".format(new_name)
                    else:  # found_cat in ['诗歌', '诗集', '诗歌集', '诗词集']
                        out_text = "{}的主要作品形式为诗歌，".format(new_name)
                    if f_cat.endswith('集'):
                        out_text += "著有{}{}".format(f_cat, f_dict.get(f_cat))
                    else:
                        out_text += "代表作有{}".format(f_dict.get(f_cat))
                    return False, out_text
                else:
                    ti_cai = celeb_dict.get('体裁')
                    ret_val = celeb_dict.get('代表作品')
                    if ti_cai and ret_val:
                        if re.search(r'，', ti_cai):
                            ti_cai = '及'.join(ti_cai.rsplit('，', 1))
                            out_text = "{}的作品体裁较多，涉及{}".format(new_name, ti_cai)
                        else:
                            out_text = "{}的作品主要为{}".format(new_name, ti_cai)
                        out_text += "，代表作有{}".format(self._replace_last_comma_for_work_name(ret_val))
                        return False, out_text
            elif new_key == '歌曲':
                f_dict = self._get_celeb_work_from_all_wiki_cats(
                    celeb_dict, ['音乐专辑', '专辑', '主要歌曲', '歌曲', '音乐作品', '音乐'])
                v1_text = f_dict.get('音乐专辑') or f_dict.get('专辑')
                v2_text = f_dict.get('主要歌曲') or f_dict.get('歌曲')
                v3_text = f_dict.get('音乐作品') or f_dict.get('音乐')
                type_cnt = 0
                if v1_text:
                    type_cnt += 1
                if v2_text:
                    type_cnt += 1
                if v3_text:
                    type_cnt += 1

                out_text = ''
                if type_cnt > 1:
                    if v1_text:
                        out_text = "{}的代表作包括专辑：{}".format(new_name, v1_text)
                    if v2_text:
                        if out_text:
                            out_text += "；歌曲：{}".format(v2_text)
                        else:
                            out_text = "{}的代表作包括歌曲：{}".format(new_name, v1_text)
                    if v3_text:
                        out_text += "；其他音乐：{}".format(v2_text)
                    return False, out_text
                elif type_cnt == 1:
                    if v1_text:
                        out_text = "{}的主要音乐专辑有{}".format(new_name, v1_text)
                    elif v2_text:
                        out_text = "{}创作或演唱的歌曲有{}等".format(new_name, v2_text)
                    else:
                        out_text = "{}的音乐代表作有{}".format(new_name, v3_text)
                    return False, out_text
                else:
                    title_role = celeb_dict.get('TITLE_ROLE')
                    if title_role and title_role in ['歌手', '男歌手', '女歌手']:
                        ret_val = celeb_dict.get('代表作品')
                        if ret_val:
                            ret_val = self._replace_last_comma_for_work_name(ret_val)
                            out_text = "{}演唱的歌曲有{}等".format(new_name, ret_val)
                            return False, out_text
            elif new_key == '电影':
                f_dict = self._get_celeb_work_from_all_wiki_cats(
                    celeb_dict, ['电影作品', '主要电影', '电影', '影视作品', '主要影视', '影视', '微电影'])
                vv_text = f_dict.get('微电影')
                v1_text = f_dict.get('电影作品') or f_dict.get('主要电影') or f_dict.get('电影')
                v2_text = f_dict.get('影视作品') or f_dict.get('主要影视') or f_dict.get('影视')
                if v1_text or v2_text:
                    if v1_text:
                        out_text = "{}的电影代表作有{}".format(new_name, v1_text)
                    else:
                        out_text = "{}的影视代表作有{}".format(new_name, v2_text)
                    if vv_text:
                        out_text = "；另有微电影作品{}".format(vv_text)
                    return False, out_text
                else:
                    title_role = celeb_dict.get('TITLE_ROLE')
                    if title_role and title_role in ['演员', '男演员', '女演员']:
                        ret_val = celeb_dict.get('代表作品')
                        if ret_val:
                            ret_val = self._replace_last_comma_for_work_name(ret_val)
                            out_text = "{}主演的影视剧有{}等".format(new_name, ret_val)
                            return False, out_text
            elif new_key == '电视剧':
                f_dict = self._get_celeb_work_from_all_wiki_cats(
                    celeb_dict, ['电视剧作品', '主要电视剧', '电视剧', '电视节目', '剧集', '影视作品', '主要影视', '影视'])
                v1_text = f_dict.get('电视剧作品') or f_dict.get('主要电视剧') or f_dict.get('电视剧') \
                          or f_dict.get('电视节目') or f_dict.get('剧集')
                v2_text = f_dict.get('影视作品') or f_dict.get('主要影视') or f_dict.get('影视')
                if v1_text or v2_text:
                    if v1_text:
                        out_text = "{}的电视剧代表作有{}".format(new_name, v1_text)
                    else:
                        out_text = "{}的影视代表作有{}".format(new_name, v2_text)
                    return False, out_text
                else:
                    title_role = celeb_dict.get('TITLE_ROLE')
                    if title_role and title_role in ['演员', '男演员', '女演员']:
                        ret_val = celeb_dict.get('代表作品')
                        if ret_val:
                            ret_val = self._replace_last_comma_for_work_name(ret_val)
                            out_text = "{}主演的影视剧有{}等".format(new_name, ret_val)
                            return False, out_text
            elif new_key == '代表作':
                ret_val = celeb_dict.get('代表作品')
                if ret_val:
                    out_text = "{}的代表作有{}".format(new_name, self._replace_last_comma_for_work_name(ret_val))
                    return False, out_text
                else:
                    f_dict = self._get_celeb_work_from_all_wiki_cats(celeb_dict, WORK_CTY_LIST)
                    f_count = len(f_dict)
                    if f_count > 1:
                        out_text = "{}著有很多的代表作品，包括".format(new_name)
                        ii = 0
                        for kk, vv in f_dict.items():
                            if ii > 0:
                                out_text += "；"
                            out_text += "{}{}".format(kk, vv)
                            ii += 1
                        return False, out_text
                    elif f_count == 1:
                        f_cat = list(f_dict.keys())[0]
                        out_text = "{}的代表作主要为{}，包括{}".format(new_name, f_cat, f_dict.get(f_cat))
                        return False, out_text
            elif new_key == '音乐专辑':
                ret_val = celeb_dict.get('代表音乐专辑') or celeb_dict.get('代表专辑')
                if ret_val:
                    out_text = "{}的著名专辑有{}".format(new_name, self._replace_last_comma_for_work_name(ret_val))
                    return False, out_text
        return False, None

    def _get_celeb_attr_info_from_all(self, new_name, new_key):
        # True means from file, otherwise from wiki
        if not new_key or new_key == '年龄':  # age is calculated
            return False, None, None

        ret_val = None
        # From file
        if new_key == '配偶':
            for tmp_key in ['老公', '老婆', '前夫', '前妻']:
                ret_val = self.attr_celeb_dict[tmp_key].get(new_name)
                if ret_val:
                    return True, ret_val, tmp_key

        attr_dict = self.attr_celeb_dict.get(new_key)
        if attr_dict:
            ret_val = attr_dict.get(new_name)
        if ret_val:
            return True, ret_val, new_key

        if new_key in ['名字', '原名', '男友', '女友']:  # 维基百科里没有清晰的名字及男女朋友信息
            if new_key == '名字':
                ret_val = self.attr_celeb_dict['原名'].get(new_name)
                if ret_val:
                    return True, ret_val, '原名'
            elif new_key == '原名':
                ret_val = self.attr_celeb_dict['名字'].get(new_name)
                if ret_val:
                    return True, ret_val, '名字'
            return False, None, None

        # From cache, then wiki
        ret_key = new_key
        celeb_dict = self.get_person_entry_dict_from_wiki(new_name)
        if celeb_dict:
            if new_key == '生日':
                _, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['出生', '生日', '出生日期'])
                if ret_val:
                    ret_val = self._extract_person_bir_dec_date(ret_val)
            elif new_key == '身高':
                _, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['身高', '登录身高'])
            elif new_key == '体重':
                _, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['体重', '登录体重'])
            elif new_key in ['前夫', '老公', '前妻', '老婆', '配偶']:
                if new_key in ['前夫', '老公']:
                    _, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['丈夫', '夫', '配偶'])
                elif new_key in ['前妻', '老婆']:
                    _, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['妻子', '妻', '配偶'])
                    if new_key == '老婆' and not ret_val:
                        k1, v1 = self._get_attr_info_by_key_list(celeb_dict, ['元配', '正室'])
                        k2, v2 = self._get_attr_info_by_key_list(celeb_dict, ['继室', '侧室', '妾室'])
                        if v1 and k1 and v2 and k2:
                            v1 = v1.replace('(', '（').replace(')', '）')
                            v2 = v2.replace('(', '（').replace(')', '）')
                            ret_val = re.sub(r'(?<=[年月日])）（', '，', "{}（{}），{}（{}）".format(v1, k1, v2, k2))
                        elif v1 and k1:
                            v1 = v1.replace('(', '（').replace(')', '）')
                            ret_val = re.sub(r'(?<=[年月日])）（', '，', "{}（{}）".format(v1, k1))
                        elif v2 and k2:
                            v2 = v2.replace('(', '（').replace(')', '）')
                            ret_val = re.sub(r'(?<=[年月日])）（', '，', "{}（{}）".format(v2, k2))
                else:  # new_key == '配偶'
                    ret_key, ret_val = self._get_attr_info_by_key_list(celeb_dict, ['配偶', '丈夫', '夫', '妻子', '妻'])
                    if ret_key and ret_val:
                        if ret_key in ['丈夫', '夫']:
                            ret_key = '老公'
                        elif ret_key in ['妻子', '妻']:
                            ret_key = '老婆'
                    if ret_key is None:
                        ret_key = '配偶'
                if ret_val is None:
                    val = celeb_dict.get("婚年")
                    if val and re.search(r'（与.*）', val):
                        ret_val = val
                elif ret_val == '已婚':
                    ret_val = None
                if ret_val:
                    ret_key = self._classify_spouse_text(ret_val, ret_key)
            elif new_key in ['父亲', '母亲']:
                find_key, ret_val = self._get_attr_info_by_key_list(celeb_dict, [new_key, '父母'])
                if ret_val and find_key == '父母':
                    ret_val = self._extract_person_parent(ret_val, new_key)
                elif not ret_val and new_key == '母亲':
                    v1, v2 = celeb_dict.get('嫡母'), celeb_dict.get('继母')
                    if v1 and v2:
                        v1 = v1.replace('(', '（').replace(')', '）')
                        v2 = v2.replace('(', '（').replace(')', '）')
                        ret_val = re.sub(r'(?<=[年月日])）（', '，', "{}（嫡母），{}（继母）".format(v1, v2))
                    elif v1:
                        v1 = v1.replace('(', '（').replace(')', '）')
                        ret_val = re.sub(r'(?<=[年月日])）（', '，', "{}（嫡母）".format(v1))
                    elif v2:
                        v2 = v2.replace('(', '（').replace(')', '）')
                        ret_val = re.sub(r'(?<=[年月日])）（', '，', "{}（继母）".format(v2))
            elif new_key == '逝世日期':
                _, ret_val = self._get_attr_info_by_key_list(celeb_dict, [new_key, '逝世'])
                if ret_val:
                    ret_val = self._extract_person_bir_dec_date(ret_val)
            else:
                ret_val = celeb_dict.get(new_key)

        return False, ret_val, ret_key

    @staticmethod
    def _get_csv_entry_count(csv_text):
        entries = csv_text.split('，')
        return len(entries)

    @staticmethod
    def _replace_last_comma_for_work_name(work_name):
        return '》及《'.join(work_name.rsplit('》，《', 1))

    def _get_celeb_work_from_all_file_cats(self, cat_list, new_name):
        found_dict = {}
        for cat_key in cat_list:
            tmp_dict = self.attr_celeb_dict.get(cat_key)
            if tmp_dict:
                ret_val = tmp_dict.get(new_name)
                if ret_val:
                    found_dict[cat_key] = ret_val
                    if ret_val.endswith('。'):
                        break
        return found_dict

    @staticmethod
    def _get_celeb_work_from_all_wiki_cats(data_dict, cat_list):
        found_dict = {}
        for cat in cat_list:
            val_text = data_dict.get('代表' + cat)
            if val_text:
                if KnowledgeBase._get_csv_entry_count(val_text) == 1:
                    found_dict[cat] = val_text
                else:
                    found_dict[cat] = KnowledgeBase._replace_last_comma_for_work_name(val_text)
        return found_dict

    @staticmethod
    def _classify_spouse_text(spouse_text, new_key):
        if new_key not in ['男友', '前夫', '老公', '女友', '前妻', '老婆']:
            return new_key
        # spouse_text from wiki
        if re.search(r'（.+），.+（.+）', spouse_text):
            return '几任老公' if new_key in ['男友', '前夫', '老公'] else '几任老婆'
        elif re.search(r'（.*离[婚异]）$', spouse_text):
            return '前夫' if new_key in ['男友', '前夫', '老公'] else '前妻'
        elif re.search(r'（.*结婚）$', spouse_text):
            return '老公' if new_key in ['男友', '前夫', '老公'] else '老婆'
        return new_key

    @staticmethod
    def _extract_person_bir_dec_date(bir_dec_text):
        date_mat = re.search(r'(公元)?前?\d{1,4}年\s?(\d{1,2}月)?\s?(\d{1,2}日)?', bir_dec_text)
        if date_mat:
            ds, de = date_mat.start(), date_mat.end()
            return bir_dec_text[ds:de].replace(' ', '')
        return None

    @staticmethod
    def _extract_person_bir_dec_year(se_date):
        year_mat = re.search(r'(公元)?前?\d{1,4}年', se_date)
        if year_mat:
            ys, ye = year_mat.start(), year_mat.end()
            year_txt = se_date[ys:ye].replace('年', '')
            if year_txt.startswith('公元前'):
                year_int = -1 * int(year_txt.replace('公元前', ''))
            else:
                year_int = int(year_txt.replace('公元', ''))
            return year_int
        return 0

    @staticmethod
    def _extract_person_bir_dec_year_month(se_date):
        year_mat = re.search(r'(公元)?前?\d{1,4}年', se_date)
        month_mat = re.search(r'\d{1,2}月', se_date)
        if year_mat and month_mat:
            year_txt = se_date[year_mat.start():year_mat.end()].replace('年', '')
            if year_txt.startswith('公元前'):
                year_int = -1 * int(year_txt.replace('公元前', ''))
            else:
                year_int = int(year_txt.replace('公元', ''))
            month_int = int(se_date[month_mat.start():month_mat.end()].replace('月', ''))
            return year_int, month_int
        return 0, 0

    @staticmethod
    def _extract_person_parent(parents_text, attr):
        assert attr in ['父亲', '母亲']
        parents_text = parents_text.replace('(父)', '（父）').replace('(母)', '（母）')
        parents_text = parents_text.replace('（父亲）', '（父）').replace('（母亲）', '（母）')
        parents_text = parents_text.replace('父：', '父亲：').replace('母：', '母亲：')
        f1_idx, m1_idx = parents_text.find('（父）'), parents_text.find('（母）')
        f2_idx, m2_idx = parents_text.find('父亲：'), parents_text.find('母亲：')
        if f1_idx > 0 and m1_idx > 0:
            p_text = parents_text.replace('，', '')
            # has to get the indexes again
            f_idx, m_idx = p_text.find('（父）'), p_text.find('（母）')
            if m_idx > f_idx:
                father = p_text[:f_idx]
                mother = p_text[f_idx+3:m_idx]
            else:
                mother = p_text[:m_idx]
                father = p_text[m_idx+3:f_idx]
            return father.strip() if attr == '父亲' else mother.strip()
        elif f2_idx >= 0 and m2_idx >= 0:
            p_text = parents_text.replace('母亲：', '，') if m2_idx > f2_idx else parents_text.replace('父亲：', '，')
            p_list = re.sub(r'，+', '，', p_text).split('，')
            if len(p_list) == 2:
                if p_list[0].find('母亲：') >= 0:
                    mother = p_list[0].replace('母亲：', '')
                    father = p_list[1]
                else:
                    father = p_list[0].replace('父亲：', '')
                    mother = p_list[1]
                return father.strip() if attr == '父亲' else mother.strip()
        elif '，' in parents_text:
            p_list = parents_text.split('，')
            if len(p_list) == 2:
                if p_list[0].find('（母）') > 0:
                    mother = p_list[0].replace('（母）', '')
                    father = p_list[1].replace('（父）', '')
                else:  # the place below may not be needed
                    father = p_list[0].replace('（父）', '')
                    mother = p_list[1].replace('（母）', '')
                return father.strip() if attr == '父亲' else mother.strip()
        elif f1_idx > 0 or f2_idx >= 0:
            father = parents_text.replace('（父）', '').replace('父亲：', '')
            return father.strip() if attr == '父亲' else None
        elif m1_idx > 0 or m2_idx >= 0:
            mother = parents_text.replace('（母）', '').replace('母亲：', '')
            return mother.strip() if attr == '母亲' else None
        return None

    """
    # get huge obj data functions and area attr functions
    """
    def get_huge_obj_data_by_key(self, cat_key, obj_key, from_most):
        # step 1: replace the number in digits, if any
        if obj_key in self.whatis_alias_dict:  # Handle case like '001型航空母舰'
            tmp = self.whatis_alias_dict[obj_key]
        else:
            tmp = KnowledgeBase._replace_digit_num_if_any(obj_key)
        # step 2: deal with the area alias for from_most case
        if from_most and '+' in tmp:
            tmp_parts = tmp.split('+')
            if len(tmp_parts) == 3:
                p0, p1, p2 = tmp_parts[0], tmp_parts[1], tmp_parts[2]
                if p0 in self.whatis_alias_dict:
                    p0 = self.whatis_alias_dict[p0]
                    if p0 == '中华人民共和国':
                        p0 = '中国'
                if p1.endswith('的'):
                    p1 = p1[:-1]
                if p2 in ['高楼', '厦', '大厦', '建筑', '建筑物']:
                    p2 = '楼'
                tmp = "{}{}{}".format(p0, p1, p2)
            else:
                tmp = tmp.replace('+', '')

        # step 3: add missing 大, 高, 长, 宽, 深
        rank1 = r'[一二三四五六七八九十]'
        tmp = re.sub(r'(?<=第{})河'.format(rank1), '大河', tmp)
        tmp = re.sub(r'(?<=第{})峰'.format(rank1), '高峰', re.sub(r'(?<=第{})山'.format(rank1), '高山', tmp))
        tmp = re.sub(r'(?<=第{})楼'.format(rank1), '高楼', tmp)
        tmp = re.sub(r'(?<=第{})岛'.format(rank1), '大岛', tmp)
        tmp = re.sub(r'(?<=面积第{})(?![大小])'.format(rank1), '大', tmp)
        tmp = re.sub(r'(?<=人口第{})(?![多少])'.format(rank1), '多', tmp)
        tmp = re.sub(r'(?<=(海拔|高度)第{})(?![高])'.format(rank1), '高', tmp)
        tmp = re.sub(r'(?<=长度第{})(?![长大])'.format(rank1), '长', tmp)
        tmp = re.sub(r'(?<=宽度第{})(?![宽])'.format(rank1), '宽', tmp)
        tmp = re.sub(r'(?<=深度第{})(?![深])'.format(rank1), '深', tmp)
        # step 4: remove redundant words, and standardize the expression
        rank2 = r'[最二三四五六七八九十]'
        tmp = re.sub(r'的?(国土|领土)', '', tmp)
        tmp = re.sub(r'(面积|海拔|长度|宽度|高度|深度)(?=[最第])', '', tmp)
        tmp = re.sub(r'最{2,}', '最', re.sub(r'(地球上?|世界上)(?=[最第])', '世界', tmp))
        tmp = re.sub(r'(?<=[{}])的'.format(HUGE_CR_QUALS), '', re.sub(r'第一(?=[{}])'.format(HUGE_CR_QUALS), '最', tmp))
        tmp = re.sub(r'(?<={}[长高深宽大])河(?=$)'.format(rank2), '河流', tmp)
        tmp = re.sub(r'(?<={}高)(高峰|[山峰])(?=$)'.format(rank2), '山峰', tmp)
        tmp = re.sub(r'(?<={}[大小])国(?=$)'.format(rank2), '国家', tmp)
        tmp = re.sub(r'(?<={}[大小])城(?=$)'.format(rank2), '城市', tmp)
        tmp = re.sub(r'湖泊', '湖', re.sub(r'岛屿', '岛', tmp))
        hugo_key = re.sub(r'省级行政区(域)?', '省', re.sub(r'省份', '省', tmp))
        print("++++++++++ obj_key = {}, hugo_key = {}".format(obj_key, hugo_key))
        # step 5: retrieve the information, and prepare the output
        if cat_key == '身高':  # deal with two prediction error cases
            cat_key = '高度'
        elif cat_key == '体重':
            cat_key = '重量'

        if cat_key == '高度':
            obj_dict = {**self.huge_obj_data['高度'], **self.huge_obj_data['海拔']}
        elif cat_key == '面积':
            obj_dict = {**self.huge_obj_data['面积'], **self.huge_obj_data['表面积']}
        elif cat_key == '重量':
            obj_dict = {**self.huge_obj_data['重量'], **self.huge_obj_data['质量']}
        else:
            obj_dict = self.huge_obj_data.get(cat_key)
        def_dict = self.huge_obj_data['名称']

        desc_txt = None
        if obj_dict:
            desc_txt = obj_dict.get(hugo_key) or obj_dict.get(obj_key)
        if not desc_txt:
            desc_txt = def_dict.get(hugo_key)
        if not desc_txt and obj_dict:
            if cat_key in ['高度', '海拔']:
                hugo_key = re.sub(r'的?最高(的?地方|[山高]?峰|[处点])$', '', hugo_key)
            elif cat_key == '深度':
                hugo_key = re.sub(r'的?最深(的?地方|处)$', '', hugo_key)
            elif cat_key == '宽度':
                hugo_key = re.sub(r'的?最宽(的?地方|处)$', '', hugo_key)
            desc_txt = obj_dict.get(hugo_key)  # this retrieval may be useless in most cases
        if not desc_txt and obj_dict and not from_most and cat_key != '名称':
            hugo_key = self._get_converted_obj_key(hugo_key)
            desc_txt = obj_dict.get(hugo_key)
        if not desc_txt and cat_key == '体积':
            banj_dict, zhij_dict = self.huge_obj_data['半径'], self.huge_obj_data['直径']
            if hugo_key in zhij_dict:
                desc_txt = "我没有其体积数据，但找到了它的直径信息。{}".format(zhij_dict[hugo_key])
            elif hugo_key in banj_dict:
                desc_txt = "我没有其体积数据，但找到了它的半径信息。{}".format(banj_dict[hugo_key])

        if desc_txt:
            cv_mat = re.search(r'_context_vals\s*\[.*\]', desc_txt)
            if cv_mat:
                ms, me = cv_mat.start(), cv_mat.end()
                ret_text = desc_txt[:ms].strip()
                ss = desc_txt.find('[', ms)
                cv_info = desc_txt[ss+1:me-1]
                if cat_key == '高度':
                    cv_info = cv_info.replace('海拔是多少', '高度是多少')
                elif cat_key == '面积':
                    cv_info = cv_info.replace('表面积是多少', '面积是多少')
                elif cat_key == '重量':
                    cv_info = cv_info.replace('质量是多少', '重量是多少')
            else:
                ret_text = desc_txt
                cv_info = None
            return ret_text, cv_info
        elif not from_most and cat_key in ['面积', '人口', '海拔', '高度', '长度', '宽度', '深度']:
            # now trying to retrieve from wiki online
            key_conversion = True if cat_key in ['面积', '人口', '海拔'] else False
            obj_attr_dict, ret_key = self.get_thing_entry_dict_from_wiki(hugo_key, key_conversion)
            if obj_attr_dict:
                if cat_key in ['面积', '人口']:
                    val_text = obj_attr_dict.get(cat_key)
                    if val_text:
                        mark1 = obj_attr_dict.get("{}注一".format(cat_key), '')
                        mark2 = obj_attr_dict.get("{}注二".format(cat_key), '')
                        if mark1:
                            mark1 = "（{}）".format(mark1)
                        if mark2:
                            mark2 = "（{}）".format(mark2)
                        if mark2 == '（总计）':
                            ret_text = "{}{}的总{}：{}".format(ret_key, mark1, cat_key, val_text)
                        else:
                            ret_text = "{}{}的{}{}：{}".format(ret_key, mark1, cat_key, mark2, val_text)
                        ret_text += "（信息提取自维基百科）"
                        cv_info = "{}=={}".format(ret_key, '面积有多大' if cat_key == '面积' else '人口有多少')
                        return ret_text, cv_info
                else:
                    val_text = obj_attr_dict.get(cat_key)
                    if val_text:
                        ret_text = "{}的{}：{}（信息提取自维基百科）".format(ret_key, cat_key, val_text)
                        return ret_text, "{}=={}是多少".format(ret_key, cat_key)
                    else:
                        key_list = ["最小{}".format(cat_key), "最大{}".format(cat_key), "平均{}".format(cat_key)]
                        all_val = ''
                        for tmp_key in key_list:
                            tmp_val = obj_attr_dict.get(tmp_key)
                            if tmp_val:
                                if all_val != '':
                                    all_val += '；'
                                all_val += "{}：{}".format(tmp_key, tmp_val)
                        if all_val:
                            ret_text = "{}的{}（信息提取自维基百科）".format(ret_key, all_val)
                            return ret_text, "{}=={}是多少".format(ret_key, cat_key)
        return None, None

    def get_area_attr_by_key(self, cat_key, obj_key):
        area_key = self.whatis_alias_dict[obj_key] if obj_key in self.whatis_alias_dict else obj_key
        obj_dict = self.huge_obj_data.get(cat_key)

        desc_txt = None
        if obj_dict:
            if area_key in obj_dict:
                desc_txt = obj_dict[area_key]
            else:
                area_key = self._get_converted_obj_key(area_key)
                desc_txt = obj_dict.get(area_key)
        if not desc_txt and cat_key == '所在国':
            extra = random.choice(["这我知道，", "这我记得，", "这我当然知道，", "", ""])
            if area_key in self.cap_nation_dict:
                desc_txt = "{}{}在{}，它是该国的首都呀。".format(extra, area_key, self.cap_nation_dict[area_key])
            elif re.search(r'[城市省]$', area_key) and area_key[:-1] in self.cap_nation_dict:
                desc_txt = "{}{}在{}，它是该国的首都啊。".format(extra, area_key, self.cap_nation_dict[area_key[:-1]])
            elif area_key in self.timezone_cities_dict and self.timezone_cities_dict[area_key].eng_city_key == 'CN':
                desc_txt = "{}{}位于并属于中国呀。".format(extra, area_key)
            elif re.search(r'[城市省]$', area_key) and area_key[:-1] in self.timezone_cities_dict and \
                    self.timezone_cities_dict[area_key[:-1]].eng_city_key == 'CN':
                desc_txt = "{}{}位于并属于中国啊。".format(extra, area_key)
        if desc_txt:
            return desc_txt, area_key

        obj_attr_dict, ret_key = self.get_thing_entry_dict_from_wiki(area_key, key_conversion=False)
        if obj_attr_dict:
            if cat_key == '所在国':
                ret_text = ''
                vt1 = obj_attr_dict.get('国家') or obj_attr_dict.get('所属国') or \
                      obj_attr_dict.get('主权国家') or obj_attr_dict.get('国')
                vt2 = obj_attr_dict.get('位置')
                if vt1:
                    ret_text = "{}的所在国或所属国：{}（信息提取自维基百科）".format(ret_key, vt1)
                elif vt2:
                    ret_text = "{}的位置：{}（信息提取自维基百科）".format(ret_key, vt2)
                if ret_text:
                    return ret_text, ret_key
            elif cat_key in ['所在省', '所在州']:
                ret_text = ''
                vt1, vt2 = obj_attr_dict.get('省') or obj_attr_dict.get('省份'), obj_attr_dict.get('自治区')
                vt3 = obj_attr_dict.get('州') or obj_attr_dict.get('州份') or obj_attr_dict.get('所属州')
                if vt1:
                    ret_text = "{}的所在省份：{}（信息提取自维基百科）".format(ret_key, vt1)
                elif vt2:
                    ret_text = "{}的所在自治区：{}（信息提取自维基百科）".format(ret_key, vt2)
                elif vt3:
                    ret_text = "{}的所在州：{}（信息提取自维基百科）".format(ret_key, vt3)
                if ret_text:
                    return ret_text, ret_key
            elif cat_key == '行政区':
                vt1, vt2 = obj_attr_dict.get('行政区划'), obj_attr_dict.get('下级行政区')
                if vt1 or vt2:
                    if vt1:
                        ret_text = "{}的行政区划：{}（信息提取自维基百科）".format(ret_key, vt1)
                    else:
                        ret_text = "{}的下级行政区：{}（信息提取自维基百科）".format(ret_key, vt2)
                    return ret_text, ret_key
            # no else conditions
        return None, None

    @staticmethod
    def find_matched_hugo_niobj(ni_ents, last_topic_val):
        # step 1: replace the number in digits, if any
        tmp = KnowledgeBase._replace_digit_num_if_any(ni_ents)
        # step 2: compile the patterns
        rank = r'[一二三四五六七八九十]'
        hugo_niobj_cat1 = re.compile(r'(面积|人口|海拔|长度|宽度|高度|深度)?最[{}][的之]?'.format(HUGE_CR_QUALS))
        hugo_niobj_cat2 = re.compile(r'(面积|人口|海拔|长度|宽度|高度|深度)?第{}[{}]?[的之]?'.format(rank, HUGE_CR_QUALS))
        # step 3: prepare the output
        if re.fullmatch(hugo_niobj_cat1, tmp) or re.fullmatch(hugo_niobj_cat2, tmp):
            # 中国=最长=河流=长度
            prev_hugo_info = last_topic_val.split("=")

            if tmp.endswith('的'):
                tmp = tmp[:-1]
            if tmp[-1] not in HUGE_CR_QUALS:
                tmp += prev_hugo_info[1][-1]

            cat_key = prev_hugo_info[3]
            hugo_name = "{}{}{}".format(prev_hugo_info[0], tmp, prev_hugo_info[2])
            # print("********** cat_key = {}, hugo_name = {}".format(cat_key, hugo_name))
            return cat_key, hugo_name
        return None, None

    @staticmethod
    def extract_hugo_criteria(in_sent):
        # step 1: replace the number in digits, if any
        tmp = KnowledgeBase._replace_digit_num_if_any(in_sent)
        # step 2: compile the patterns
        rank = r'[一二三四五六七八九十]'
        cr_conds = r'面 积|人 口|海 拔|长 度|宽 度|高 度|深 度'
        cr_quals = r'{}'.format(HUGE_CR_QUALS)
        cr_units = r'面 积|人 口|海 拔|长 度|宽 度|高 度|深 度|半 径|直 径|周 长|体 积|重 量'
        hugo_cr_cat1 = re.compile(r'(({0})\s)?最\s[{1}](\s[的之])?(?!\s({2}))'.format(cr_conds, cr_quals, cr_units))
        hugo_cr_cat2 = re.compile(r'(({0})\s)?第\s{1}(\s[{2}])?(\s[的之])?(?!\s({3}))'.format(
            cr_conds, rank, cr_quals, cr_units))
        cr_mat1 = re.search(hugo_cr_cat1, tmp)
        if cr_mat1:
            return ''.join(tmp[cr_mat1.start():cr_mat1.end()].split())
        cr_mat2 = re.search(hugo_cr_cat2, tmp)
        if cr_mat2:
            return ''.join(tmp[cr_mat2.start():cr_mat2.end()].split())
        return None

    @staticmethod
    def _replace_digit_num_if_any(hugo_text):
        d_num_cat = re.compile(r'(10|[1-9])(?!\d)')
        d_num_mat = re.search(d_num_cat, hugo_text)
        if d_num_mat:
            ss, ee = d_num_mat.start(), d_num_mat.end()
            num_chars = '一二三四五六七八九十'
            hanzi_num = num_chars[int(hugo_text[ss:ee]) - 1]
            out_text = hugo_text[:ss] + hanzi_num + hugo_text[ee:]
        else:
            out_text = hugo_text
        return out_text

    """
    # get work attr function
    """
    def get_work_attr_info(self, work_name, attr_key):
        print("work_name = {}, attr_key = {}".format(work_name, attr_key))
        new_work = self.work_alias_dict[work_name] if work_name in self.work_alias_dict else work_name
        ret_text = ''
        if attr_key in self.work_attr_data:
            work_attr_dict = self.work_attr_data[attr_key]
            tmp_work = self._remove_work_cat(new_work) if new_work not in work_attr_dict else new_work
            if tmp_work in work_attr_dict:
                ret_text = work_attr_dict[tmp_work]
            elif tmp_work in self.work_alias_dict:
                tmp_work = self.work_alias_dict[tmp_work]
                if tmp_work in work_attr_dict:
                    ret_text = work_attr_dict[tmp_work]
            if not ret_text and attr_key in ['作者', '编剧', '导演', '主演']:
                obj_attr_dict = self.get_work_entry_dict_from_wiki([new_work, tmp_work])
                if obj_attr_dict and attr_key in obj_attr_dict:
                    ret_key = obj_attr_dict[attr_key]
                    if re.search('^(小说|诗歌|歌曲|电影|电视(连续)?剧|古装剧)', new_work):
                        with_cat = True
                        ret_text = "{}的{}是：".format(new_work, attr_key)
                    else:
                        with_cat = False
                        ret_text = "《{}》的{}是：".format(new_work, attr_key)
                    if '，' in ret_key:
                        ret_text += "{}（信息提取自维基百科）".format(ret_key)
                        if with_cat:
                            ret_text += " _fc_tamen1 [{}的{}们]".format(new_work, attr_key)
                        else:
                            ret_text += " _fc_tamen1 [《{}》的{}们]".format(new_work, attr_key)
                    else:
                        new_name = self.celeb_alias_dict.get(ret_key.lower()) or ret_key
                        ret_text += "{}（信息提取自维基百科）".format(new_name)
                        if new_name in self.celeb_female_set:
                            ret_text += ' _fc_ta2 [{}]'.format(new_name)
                        elif new_name in self.celeb_male_set:
                            ret_text += ' _fc_ta1 [{}]'.format(new_name)
                        else:
                            ret_text += ' _fc_ta1_ta2 [{}]'.format(new_name)
        return new_work, ret_text

    """
    # get object leader function
    """
    def get_org_leader_info(self, org_name, pos):
        ret_text = ''
        low_org = org_name.lower()
        new_org = self.whatis_alias_dict.get(low_org) or low_org
        new_pos = re.sub(r'^(最高|国家|现任)(?!$)', '', pos).upper()
        if new_org in self.org_leader_data:
            ret_val = ''
            extra = random.choice(["这我知道，", "这我记得，", "这我当然知道，", "", ""])
            org_dict = self.org_leader_data[new_org]
            if pos in org_dict:
                ret_val = org_dict[pos]
            elif new_pos in org_dict:
                ret_val = org_dict[new_pos]
            if ret_val:
                ret_text = ret_val if ret_val.endswith('。') else "{}{}{}：{}。".format(extra, new_org, pos, ret_val)
            elif new_pos in ['老板', '大老板', '负责人', '一把手']:
                k0, n0 = self._get_attr_info_by_key_list(org_dict, ['董事长', '所有权者', '创办人'])
                k1, n1 = self._get_attr_info_by_key_list(org_dict, ['首席执行官', '执行长', '总经理'])
                if k0 and k1:
                    ret_text = "{}{}的{}：{}；{}：{}。".format(extra, new_org, k0, n0, k1, n1)
                elif k0:
                    ret_text = "{}{}的{}：{}。".format(extra, new_org, k0, n0)
                elif k1:
                    ret_text = "{}的{}：{}。".format(new_org, k1, n1)
            elif new_pos in ['总裁', '老总', '总经理', 'CEO']:
                k1, n1 = self._get_attr_info_by_key_list(org_dict, ['首席执行官', '执行长', '总经理'])
                if k1 and n1:
                    ret_text = "{}{}的{}：{}。".format(extra, new_org, k1, n1)
            if ret_text:
                return ret_text, new_org

        ob_at_dict, new_org = self.get_thing_entry_dict_from_wiki(new_org)
        if not ob_at_dict:
            ob_at_dict, new_org = self.get_thing_entry_dict_from_wiki(low_org, key_conversion=True)
        if ob_at_dict:
            if pos in ob_at_dict:
                de = '' if pos in ['省委书记', '省长', '市委书记', '市长', '行长', '局长'] else '的'
                ret_text = "{}{}{}：{}".format(new_org, de, pos, ob_at_dict[pos])
            elif new_pos in ob_at_dict:
                de = '' if new_pos in ['省委书记', '省长', '市委书记', '市长', '行长', '局长'] else '的'
                ret_text = "{}{}{}：{}".format(new_org, de, new_pos, ob_at_dict[new_pos])
            else:
                if new_pos in ['领导人', '掌舵人', '领袖', '统帅', '元首', '首脑']:
                    k0, n0 = self._get_attr_info_by_key_list(ob_at_dict, ['君主', '国王', '女王', '天皇', '埃米尔'])
                    k1, n1 = self._get_attr_info_by_key_list(ob_at_dict, ['总督'])
                    k2, n2 = self._get_attr_info_by_key_list(ob_at_dict, ['总统', '首相', '总理', '内阁总理大臣'])
                    if k0 and n0:
                        ret_text = "{}的{}：{}".format(new_org, k0, n0)
                    if k1 and n1:
                        if ret_text:
                            ret_text += "；{}：{}".format(k1, n1)
                        else:
                            ret_text = "{}的{}：{}".format(new_org, k1, n1)
                    if k2 and n2:
                        if ret_text:
                            ret_text += "；{}：{}".format(k2, n2)
                        else:
                            ret_text = "{}的{}：{}".format(new_org, k2, n2)
                elif new_pos in ['君主', '国王', '女王']:
                    k0, n0 = self._get_attr_info_by_key_list(ob_at_dict, ['君主', '国王', '女王'])
                    if k0 and n0:
                        ret_text = "{}的{}：{}".format(new_org, k0, n0)
                elif new_pos in ['总统', '首相', '总理']:  # only when itself cannot be found
                    k2, n2 = self._get_attr_info_by_key_list(ob_at_dict, ['总统', '首相', '总理', '内阁总理大臣'])
                    if k2 and n2:
                        ret_text = "{}的{}：{}".format(new_org, k2, n2)
                elif new_pos in ['教皇', '教宗', '主教', '大主教']:
                    kk, nn = self._get_attr_info_by_key_list(ob_at_dict, ['教区主教'])
                    if kk and nn:
                        ret_text = "{}的{}：{}".format(new_org, kk, nn)
                elif new_pos in ['行政长官', '特首']:  # 香港和澳门的特首
                    for kk, vv in ob_at_dict.items():
                        if re.search(r'行政长官(?= |$)', kk):
                            ret_text = "{}的行政长官：{}".format(new_org, vv)
                elif new_pos == '联席主席':
                    for kk, vv in ob_at_dict.items():
                        if re.search(r'主席$', kk):
                            ret_text = "{}的{}：{}".format(new_org, kk, vv)
                elif re.search(r'局长$', new_pos):
                    for kk, vv in ob_at_dict.items():
                        if re.search(r'首长$', kk):
                            ret_text = "{}的{}为{}".format(new_org, kk, vv)
                elif re.search(r'行长$', new_pos):  # 当美联储为称为美国央行时
                    kk, nn = self._get_attr_info_by_key_list(ob_at_dict, ['主席'])
                    if kk and nn:
                        ret_text = "{}{}：{}".format(new_org, kk, nn)
                elif new_pos in ['教练', '主教练']:
                    kk, nn = self._get_attr_info_by_key_list(ob_at_dict, ['主教练', '教练'])
                    if kk and nn:
                        ret_text = "{}{}：{}".format(new_org, kk, nn)
                elif new_pos in ['省党委书记', '区党委书记', '市党委书记', '党委书记', '省委书记', '区委书记']:
                    if new_pos == '党委书记':
                        kk, nn = self._get_attr_info_by_key_list(ob_at_dict, ['省委书记', '自治区党委书记', '市委书记'])
                    elif new_pos == '市党委书记':
                        kk, nn = self._get_attr_info_by_key_list(ob_at_dict, ['市委书记'])
                    else:
                        kk, nn = self._get_attr_info_by_key_list(ob_at_dict, ['省委书记', '自治区党委书记'])
                    if kk and nn:
                        ret_text = "{}的{}：{}".format(new_org, kk, nn)
                elif new_pos in ['省长', '区长', '区主席', '主席']:  # '联席主席' 是比如足球俱乐部的职位
                    kk, nn = self._get_attr_info_by_key_list(ob_at_dict, ['省长', '自治区主席', '联席主席'])
                    if kk and nn:
                        ret_text = "{}的{}：{}".format(new_org, kk, nn)
                elif new_pos in ['老板', '大老板', '负责人', '一把手', '董事长', '首席执行官', '执行长', '总裁', '老总',
                                 '总经理', 'CEO']:
                    k0, n0 = self._get_attr_info_by_key_list(ob_at_dict, ['董事长', '党委书记', '创办人', '创始人',
                                                                          '创立者'])
                    k1, n1 = self._get_attr_info_by_key_list(ob_at_dict, ['首席执行官', '执行长', '总经理', '行长',
                                                                          '机构首长'])
                    _, nn = self._get_attr_info_by_key_list(ob_at_dict, ['代表人物'])
                    if new_pos in ['老板', '大老板', '负责人', '一把手']:
                        if k0 and k1:
                            ret_text = "{}的{}：{}；{}：{}".format(new_org, k0, n0, k1, n1)
                        elif k0:
                            ret_text = "{}的{}：{}".format(new_org, k0, n0)
                        elif k1:
                            ret_text = "{}的{}：{}".format(new_org, k1, n1)
                    elif new_pos in ['总裁', '老总', '总经理', 'CEO']:
                        if k1 and n1:
                            ret_text = "{}的{}：{}".format(new_org, k1, n1)
                        elif k0 and n0:
                            ret_text = "{}的{}我不清楚，不过我知道它的{}：{}".format(new_org, new_pos, k0, n0)
                    if not ret_text and nn:
                        ret_text = "{}的主要高层为{}".format(new_org, nn)

            if ret_text:
                ret_text += "（信息提取自维基百科）"

        return ret_text, new_org

    """
    # get animal attr info function
    """
    def get_animal_attr_info_text(self, animal_name, attr_key):
        ret_text = ''
        new_name = self.animal_alias_dict.get(animal_name) or animal_name
        # {'动植物', '有腿', '有翅', '有尾', '有眼', '有耳', '会飞', '会游', '会跑', '会走'}
        if attr_key == '动植物':
            extra = random.choice(["这我知道，", "这是简单常识，", "这还用说，", ""])
            spec_dict = self.attr_animal_dict['特殊']
            if new_name in spec_dict:
                ret_text = spec_dict[new_name]
            elif new_name in self.animal_set:
                ret_text = "{}{}是动物。 _func_ta3_tamen3".format(extra, animal_name)
            elif new_name in self.plant_set:
                ret_text = "{}{}是植物。 _func_ta3_tamen3".format(extra, animal_name)
            elif new_name in self.non_animal_non_plant_set:
                ret_text = "{}{}既不是动物，也不是植物。 _func_ta3_tamen3".format(extra, animal_name)
        elif attr_key in self.attr_animal_dict:
            attr_dict = self.attr_animal_dict[attr_key]
            if new_name in attr_dict:
                ret_text = attr_dict[new_name]
            elif new_name in self.plant_set or new_name in self.non_animal_non_plant_set:
                base_text = '是植物' if new_name in self.plant_set else '不是动物'
                if attr_key == '有腿':
                    desc = random.choice(['应该没有腿吧', '没听说它有腿啊'])
                    ret_text = "{}{}，{}。".format(animal_name, base_text, desc)
                elif attr_key == '有翅':
                    desc = random.choice(['应该没有翅膀吧', '没听说它有翅膀啊'])
                    ret_text = "{}{}，{}。".format(animal_name, base_text, desc)
                elif attr_key == '有尾':
                    desc = random.choice(['应该没有尾巴吧', '没听说它有尾巴啊'])
                    ret_text = "{}{}，{}。".format(animal_name, base_text, desc)
                elif attr_key == '有眼':
                    desc = random.choice(['应该没有眼睛吧', '没听说它有眼睛啊'])
                    ret_text = "{}{}，{}。".format(animal_name, base_text, desc)
                elif attr_key == '有耳':
                    desc = random.choice(['应该没有耳朵吧', '没听说它有耳朵啊'])
                    ret_text = "{}{}，{}。".format(animal_name, base_text, desc)
                elif attr_key == '会飞':
                    desc = random.choice(['应该不会飞吧', '没听说它会飞啊'])
                    ret_text = "{}{}，{}。".format(animal_name, base_text, desc)
                elif attr_key == '会游':
                    desc = random.choice(['应该不会游泳吧', '没听说它会游泳啊'])
                    ret_text = "{}{}，{}。".format(animal_name, base_text, desc)
                elif attr_key == '会爬树':
                    desc = random.choice(['应该不会爬树吧', '没听说它会爬树啊'])
                    ret_text = "{}{}，{}。".format(animal_name, base_text, desc)
                elif attr_key == '会跑':
                    desc = random.choice(['应该不会奔跑吧', '没听说它会奔跑啊'])
                    ret_text = "{}{}，{}。".format(animal_name, base_text, desc)
                elif attr_key == '会走':
                    desc = random.choice(['应该不会行走吧', '没听说它会行走啊'])
                    ret_text = "{}{}，{}。".format(animal_name, base_text, desc)

        return ret_text

    """
    # get sport event info function
    """
    def get_sport_event_info(self, cat, typ, spt_yr, abs_jc, rel_jc):
        ret_text = ''
        if cat in self.sport_event_data_dict:
            cur_list = self.sport_event_data_dict[cat]
            cur_event = None
            today = dt.date.today()
            cur_year = today.year

            if rel_jc in [-2, -1, 0, 1, 2]:
                for spt_evt in cur_list:
                    if spt_evt.year <= cur_year:
                        abs_jc = spt_evt.index
                    else:
                        break
                if abs_jc > 0:
                    abs_jc += rel_jc
            for spt_evt in cur_list:
                if spt_evt.year == spt_yr > 0 or spt_evt.index == abs_jc > 0:
                    cur_event = spt_evt
                    break
            if not cur_event and spt_yr > 0:
                prev_year, next_year = 0, 0
                for spt_evt in cur_list:
                    if spt_yr - 3 <= spt_evt.year <= spt_yr + 3:
                        if spt_evt.year < spt_yr:
                            prev_year = spt_evt.year
                        else:
                            next_year = spt_evt.year
                ret_text = "据我所知，{}年并没有举办{}啊。".format(spt_yr, cat)
                if prev_year > 0 and next_year > 0:
                    ret_text += "相距最近的有{}年的以及{}年的。".format(prev_year, next_year)
                elif prev_year > 0:
                    ret_text += "相距最近的有{}年的。".format(prev_year)
                elif next_year > 0:
                    ret_text += "相距最近的有{}年的。".format(next_year)
                return ret_text

            if cur_event:
                if spt_yr > 0:
                    jc_txt = '{}年'.format(spt_yr)
                elif rel_jc in [-2, -1, 0, 1, 2]:
                    jc_txt = ['上上届', '上一届', '这一届', '下一届', '下下届'][rel_jc+2]
                    if cur_event.year != cur_year and typ != '年份':
                        jc_txt += '（{}年）'.format(cur_event.year)
                else:  # abs_jc > 0
                    if typ == '年份':
                        jc_txt = '第{}届'.format(abs_jc)
                    else:
                        jc_txt = '第{}届（{}年）'.format(abs_jc, cur_event.year)
                jc_txt += cat

                extra = random.choice(["这信息很容易查到，", "这我知道，", "这在网上不难查到，", "这我记得，", ""])

                if cur_event.status == 0:
                    if typ == '年份':
                        ret_text = "{}{}原计划于{}年在{}举办，后{}。".format(extra, jc_txt, cur_event.year,
                                                                  cur_event.place, cur_event.cancelled_cause)
                    else:
                        ret_text = "{}{}原计划在{}举办，后{}。".format(extra, jc_txt, cur_event.place,
                                                              cur_event.cancelled_cause)
                    return ret_text

                sm, sd = self._get_month_day_from_date_text(cur_event.start_date)
                em, ed = self._get_month_day_from_date_text(cur_event.end_date)
                if typ == '年份':
                    if cur_event.status == 1 or int(cur_event.year) < cur_year:
                        ret_text = "{}{}举办于{}年。".format(extra, jc_txt, cur_event.year)
                    elif cur_event.status == 2:
                        if int(cur_event.year) == cur_year:
                            if dt.date(cur_year, sm, sd) > today:
                                ret_text = "{}{}将于{}年（即今年）举办。".format(extra, jc_txt, cur_event.year)
                            elif dt.date(cur_year, em, ed) > today:
                                ret_text = "{}{}正于{}年（即今年）举办。".format(extra, jc_txt, cur_event.year)
                            else:
                                ret_text = "{}{}已于{}年（即今年）举办。".format(extra, jc_txt, cur_event.year)
                        else:
                            ret_text = "{}{}将于{}年举办。".format(extra, jc_txt, cur_event.year)
                elif typ == '地点':
                    if cat == '世界杯':
                        ret_text = "{}{}的主办国是{}。".format(extra, jc_txt, cur_event.place)
                    else:
                        ret_text = "{}{}的主办地是{}。".format(extra, jc_txt, cur_event.place)
                elif typ == '日期':
                    dist = (dt.date(int(cur_event.year), em, ed) - dt.date(int(cur_event.year), sm, sd)).days + 1
                    ret_text = "{}{}开幕于{}，闭幕于{}，共计{}天。".format(extra, jc_txt, cur_event.start_date,
                                                               cur_event.end_date, dist)
                elif typ == '第1名' and cur_event.first:
                    ret_text = "{}{}的冠军是{}。".format(extra, jc_txt, cur_event.first)
                elif typ == '第2名' and cur_event.second:
                    ret_text = "{}{}的亚军是{}。".format(extra, jc_txt, cur_event.second)
                elif typ == '第3名' and cur_event.third:
                    ret_text = "{}{}的季军是{}。".format(extra, jc_txt, cur_event.third)
                elif typ == '第4名' and cur_event.fourth:
                    ret_text = "{}{}的第四名是{}。".format(extra, jc_txt, cur_event.fourth)
                elif typ == '名次' and cur_event.status == 1:
                    if cur_event.first == cur_event.second:
                        ret_text = "{}{}的前四名依次为：{}、{}、{}。".format(extra, jc_txt, cur_event.first,
                                                                  cur_event.third, cur_event.fourth)
                    else:
                        ret_text = "{}{}的前四名依次为：{}、{}、{}、{}。".format(extra, jc_txt,cur_event.first,
                                                                     cur_event.second, cur_event.third,
                                                                     cur_event.fourth)
                elif typ in ['第1名', '第2名', '第3名', '第4名', '名次'] and cur_event.status == 2:
                    if dt.date(int(cur_event.year), sm, sd) > today:
                        ret_text = "晕，我也不是先知，{}的比赛还未开始，我如何预测最终的排名呢？".format(extra, jc_txt)
                    elif dt.date(int(cur_event.year), em, ed) > today:
                        ret_text = "{}的比赛还未结束，所以还没有最终的排名。".format(jc_txt)
        return ret_text

    @staticmethod
    def _get_month_day_from_date_text(date_text):  # 10月12日
        mds = date_text[:-1].split('月')
        return int(mds[0]), int(mds[1])


class Story(namedtuple("Story", ["cat", "content"])):
    pass


class Poem(namedtuple("Poem", ["title", "writer", "content", "explanation"])):
    pass


class Lyric(namedtuple("Lyric", ["title", "writer", "composer", "singer", "content"])):
    pass


class SportEvent(namedtuple("SportEvent", ["index", "year", "place", "start_date", "end_date", "first", "second",
                                           "third", "fourth", "status", "cancelled_cause"])):
    pass


if __name__ == '__main__':
    from settings import PROJECT_ROOT

    knbs = KnowledgeBase()
    knbs.load_knbase(os.path.join(PROJECT_ROOT, 'Data', 'KnowledgeBase'))

    # for id, poem in knbs.poems.items():
    #     if id in knbs.poem_dynasties['唐代']:
    #         print("{}\n{}\n{}\n{}\n{}\n\n".format(id, poem.title, poem.writer, poem.content, poem.explanation))
    #
    # print("===")
    # for id, poem in knbs.poems_no_exp.items():
    #     print("{}\n{}\n{}\n{}\n\n".format(id, poem.title, poem.writer, poem.content))
    #
    # print("===")
    # for ln_idx, line in enumerate(knbs.poem_lines_list):
    #     print("{} = {}".format(ln_idx, line))
    #
    # print("===")
    # print(knbs.poem_lines_list[100])
    # print(knbs.poem_lines_list[101])
    # print(knbs.poem_lines_dict.get(knbs.poem_lines_list[100], 'An ID line'))
    # print(knbs.poem_lines_dict.get(knbs.poem_lines_list[101], 'An ID line'))
    #
    # print("===")
    # w_list = knbs.poem_writers['李白']
    # for id in w_list:
    #     poem = knbs.poems[id]
    #     print("{}\n{}\n{}\n{}\n{}\n".format(id, poem.title, poem.writer, poem.content, poem.explanation))
    #
    # for k, v in knbs.celeb_dict.items():
    #     print("===")
    #     print("{}: {}".format(k, v))
    #
    # for k, v in knbs.entry_meaning_dict.items():
    #     print("===")
    #     print("{}: {}".format(k, v))
    #
    # for k, v in knbs.entry_whatis_dict.items():
    #     print("===")
    #     print("{}: {}".format(k, v))
    #
    # print("===")
    # poem_id, cat = knbs.get_poem_id_by_title("长恨歌")
    # if poem_id > 0:
    #     if cat == 'PHID':
    #         print(knbs.poems[poem_id].content)
    #     elif cat == 'PNID':
    #         print(knbs.poems_no_exp[poem_id].content)
    # else:
    #     print("Poem not found.")
    #
    # print("===")
    # cy_cnt = 0
    # for k, v in knbs.chengyu_dict.items():
    #     cy_cnt += 1
    #     if cy_cnt < 10:
    #         print("===")
    #         print(k)
    #         for vi in v:
    #             print("  {}".format(vi))
    #
    # print("===")
    # for k, v in knbs.weather_cities_dict.items():
    #     print("{}:: {}".format(k, v))
    #
    # print("===")
    # for k, v in knbs.timezone_cities_dict.items():
    #     print("{}:: {}".format(k, v))

    print("=== all alias entries")
    for k, v in knbs.celeb_alias_dict.items():
        print("{}: {}".format(k, v))

    # print("=== all females:")
    # for ff in knbs.celeb_female_set:
    #     print(ff)
    #
    # print("=== all males:")
    # for mm in knbs.celeb_male_set:
    #     print(mm)

    print("百家姓总姓氏个数：{}".format(len(knbs.chinese_xing_set)))

    # union
    un = knbs.celeb_female_set | knbs.celeb_male_set
    print("female count = {}; male count = {}; union count: {}".format(
        len(knbs.celeb_female_set), len(knbs.celeb_male_set), len(un)))

    # intersection
    inter = knbs.celeb_female_set & knbs.celeb_male_set
    print("inter count: {}".format(len(inter)))

    print("=== all celeb attr items:")
    for k, v in knbs.attr_celeb_dict.items():
        print(k)
        for k2, v2 in v.items():
            print("{}: {}".format(k2, v2))

    print("=== prov city info dicts:")
    for k, v in knbs.prov_city_dict.items():
        print("{}: {}".format(k, v))

    print("***")
    for k, v in knbs.city_prov_dict.items():
        print("{}: {}".format(k, v))

    print("***")
    for k, v in knbs.state_shoufu_dict.items():
        print("{}: {}".format(k, v))

    print("***")
    for k, v in knbs.shoufu_state_dict.items():
        print("{}: {}".format(k, v))

    print("***")
    for k, v in knbs.prov_city_jc_dict.items():
        print("{}: {}".format(k, v))

    print("***")
    for k, v in knbs.jc_prov_dict.items():
        print("{}: {}".format(k, v))

    print("***")
    for k, v in knbs.jc_city_dict.items():
        print("{}: {}".format(k, v))

    print("==========================")
    for k, v in knbs.lyrics.items():
        tt = v.title
        lid = knbs.get_lyric_id_by_title(tt)
        if lid:
            ly = knbs.lyrics[lid]
            print(ly.title)
            # print(ly.writer)
            # print(ly.composer)
            # print(ly.singer)
            # print(ly.content.replace('_nl_', '\n'))
            # print("==========================")

    for loc in ['美国德州', '德州', '吉林市', '美国首都', '首都', '吉林', '吉林省吉林市', '中国首都', '首都北京',
                '魔都上海', '中国重庆市', '美国佛州迈阿密', '江苏南京', '浙江省杭州市', '华盛顿州', '华盛顿特区']:
        print("{} -> {}".format(loc, knbs._get_converted_obj_key(loc)))

    print("==========================")
    for loc in ['中国首都北京', '中国首都', '首都北京', '首都',
                '美国首都华盛顿', '美国首都', '首都华盛顿',
                '日本首都东京', '日本的首都东京', '首都东京', '日本首都',
                '加国首都', '加拿大首都', '加拿大首都渥太华', '首都渥太华',
                '俄罗斯首都莫斯科', '俄罗斯首都', '俄国首都']:
        print("{} -> {}".format(loc, knbs.parse_loc_text_capital(loc)))

    print("==========================")
    for k, v in knbs.cn_en_trans_dict.items():
        print("{} => {}".format(k, v))

    print("***")
    for k, v in knbs.animal_alias_dict.items():
        print("{}: {}".format(k, v))

    for k, v in knbs.attr_animal_dict.items():
        print("#={}".format(k))
        for k2, v2 in v.items():
            print("{}: {}".format(k2, v2))

    print("animal count = {}; plant count = {}; non-non count: {}".format(
        len(knbs.animal_set), len(knbs.plant_set), len(knbs.non_animal_non_plant_set)))

    for k, v in knbs.sport_event_data_dict.items():
        print("##{}".format(k))
        for line in v:
            print(*line, sep='\t')
