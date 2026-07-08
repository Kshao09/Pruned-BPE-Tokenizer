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
import random
import re
import time

import datetime as dt
import mpmath as mp
from sympy import *
from sympy.parsing.sympy_parser import parse_expr

from chatbot.datautil import BPE_TAG_SET, get_formatted_word, join_formatted_list, to_dense_text
from chatbot.numunitpats import *
from addons.rules.chatsession import Topic
from addons.rules.knowledgebase import *
from addons.rules.nongli import LunarDate, JieQi
from addons.rules.jieri import JieRi, JieRiTopic, JieRiGreeting
from addons.rules.calendarutils import WEEK_DAYS, CalendarUtils as CalUtils, CalendarPatterns as CalPats
from addons.rules.patternutils import parse_unit_amount_text, extract_unit_amount_or_unit_text, \
    extract_unit_amount_unit_and_replace, formalize_temper_unit
from addons.rules.weatherutils import CityWeather, query_weather_by_city
from addons.rules.arithmeticparser import Tree as ArithTree


class FunctionData(object):
    def __init__(self, params, knowledge_base, addons_dict=None):
        """
        Only one instance will exist for the whole application.
        Args:
            params: Hyper-parameters configured in modelparams.py file
            knowledge_base: The knowledge base data needed for prediction.
            addons_dict: The dict to hold all add-on objects.
        """
        self.params = params
        self.knowledge_base = knowledge_base
        self.ner_predictor = None
        self.poem_writer = None
        self.ci_poem_writer = None
        if addons_dict:
            if 'NER_PREDICTOR' in addons_dict:
                self.ner_predictor = addons_dict['NER_PREDICTOR']
            if 'POEM_WRITER' in addons_dict:
                self.poem_writer = addons_dict['POEM_WRITER']
            if 'CI_POEM_WRITER' in addons_dict:
                self.ci_poem_writer = addons_dict['CI_POEM_WRITER']


class SessionFunction(object):
    easy_list = [
        "", "", "", "", "", "",
        "这个简单：",
        "这个不难：",
        "这个问题我会：",
        "这个简单，我知道：",
    ]
    hard_list = [
        "", "", "", "", "", "",
        "这是计算过程和结果：",
        "这个稍稍有点难算：",
        "我花了点时间才算出来。这是结果：",
        "这个有些复杂，花了我点时间。计算如下：",
    ]
    greeting_list = [
        "幸会幸会！",
        "见到你我好高兴！",
        "见到你我好开心！",
        "幸会幸会，请多关照！",
        "认识你我感到很荣幸！",
        "很高兴有机会与你聊天！",
        "很开心有机会与你聊天！",
        "能跟你聊天我感到很愉快！",
        "能跟你聊天我感到好开心！",
        "很高兴有机会结识你这样的朋友！"
    ]

    ask_weather_city_list = [
        "只是你想了解哪个城市的天气呢？",
        "对了，你想查询哪个城市的天气情况呢？",
        "那么，你想知道哪个城市的天气情况呢？"
    ]

    get_poem_alter_list = [
        "对了，这么背诗多没意思呀，不好玩。要不让我给你写首诗吧？ _func_propose_topic_para0_flw [写诗]",
        "矮油，这么背诗多没劲呀，一点也不好玩。要不我给你写首诗吧？ _func_propose_topic_para0_flw [写诗]"
    ]

    # poem_seed_chars = "一三六七八十百千万两大小少年岁月日人春夏秋冬江海谁水故春秋天" \
    #                   "一江春南山天不日白高万秋东何我西清昔野玉故自长上朝北云夜青风" \
    #                   "远君古独闻寒旧一三江天春长十白山万玉不南青秋曾云西风东春江山"

    mugua_shi = "投我以木瓜，报之以琼琚。匪报也，永以为好也！_nl_" \
                "投我以木桃，报之以琼瑶。匪报也，永以为好也！_nl_" \
                "投我以木李，报之以琼玖。匪报也，永以为好也！"
    mugua_shi_exp = "你用木瓜送给我，我用美玉回报你。_nl_美玉不单是回报，也是为求永相好。_np_" \
                    "你用木桃送给我，我用琼瑶作回报。_nl_琼瑶不单是回报，也是为求永相好。_np_" \
                    "你用木李送给我，我用琼玖作回报。_nl_琼玖不单是回报，也是为求永相好。"

    init_chengyu_list = [
        "趁心如意", "应对如流", "赞不绝口", "无边风月", "身无分文",
        "万众一心", "成竹在胸", "雄姿英发", "水涨船高", "如鱼得水",
        "博大精深", "不辞而别", "碧海青天", "表里如一", "杀一儆百",
        "放虎归山", "因小失大", "互通有无", "闭月羞花", "金石为开"
    ]

    def __init__(self, func_data, chat_session):
        """
        One instance per ChatSession. It is a wrapper that holds a link to the common instance 
        of FunctionData, and the chat_session for this session. 
        Args:
            func_data: The common instance of the FunctionData.
            chat_session: The chat session object that can be read and written.
        """
        self.chat_session = chat_session

        self.params = func_data.params
        self.knowledge_base = func_data.knowledge_base
        self.ner_predictor = func_data.ner_predictor
        self.poem_writer = func_data.poem_writer
        self.ci_poem_writer = func_data.ci_poem_writer

        # Functions that can possibly be embedded inside a sentence
        self.inner_fun_dict = {
            # 'set_use_simp': self.set_use_simp,
            # 'get_boshao_contact': self.get_boshao_contact,

            # 'get_date_time': self.get_date_time,
            # 'get_time': self.get_time,
            # 'get_today': self.get_today,
            # 'get_current_year': self.get_current_year,

            'get_story_any': self.get_story_any,
            'get_joke_any': self.get_joke_any,
            'get_duanzi_any': self.get_duanzi_any,

            'get_poem_any': self.get_poem_any,
            # 'compose_poem': self.compose_poem,
            'compose_poem_any': self.compose_poem_any,
            'compose_ci_poem_any': self.compose_ci_poem_any,

            'get_user_name': self.get_user_name,
            'keep_topic': self.keep_topic,
            # 'start_chengyu_jielong': self.start_chengyu_jielong
        }

    """
    # Basic rule: Use simplified or traditional Chinese in the conversation
    # And basic info: get_boshao_contact
    """
    # def set_use_simp(self, use_simp):
    #     if use_simp == 'true':
    #         self.chat_session.use_simplified = True
    #     elif use_simp == 'false':
    #         self.chat_session.use_simplified = False
    #     return '', 0

    # def skip_tradit_convert(self):
    #     self.chat_session.tradit_convert = False
    #     return '', 0

    @staticmethod
    def get_random_out(sentence):
        dense_sent = to_dense_text(sentence)
        item_list = re.findall(r'<=(.*?)=>', dense_sent)
        if len(item_list) == 1 and item_list[0]:
            all_out = item_list[0].replace(' ', '').strip()
            if all_out.find('|') > 0:
                out_opts = all_out.split('|')
                return random.choice(out_opts), 0
            else:
                return all_out, 0
        return '', 0

    def greet_if_not_yet(self):
        if self.chat_session.greeted:
            return '', 0
        self.chat_session.greeted = True
        return random.choice(SessionFunction.greeting_list), 0

    def check_and_set_question_asked(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            quest_key = item_list[0].replace(' ', '').strip()
            ret_text, asked_time = self.chat_session.check_and_set_question_asked(quest_key)
            if asked_time >= 1:
                if self.chat_session.max_allowed_appear < 4:  # == 1
                    self.chat_session.max_allowed_appear = 4

                try_time = asked_time + 1 if asked_time <= 8 else 9
                if try_time > 6:
                    self.chat_session.context_ref_base = '相同话题重复告警 _cr_ 第{}次'.format(try_time-6)
                else:
                    if try_time >= 4:
                        if self.chat_session.max_allowed_appear == 4:
                            for ii in [6, 5]:
                                test_input = quest_key + ' _cr_ 第{}次'.format(ii)
                                if self.knowledge_base.get_preload_pair_value(test_input, self.chat_session):
                                    self.chat_session.max_allowed_appear = ii
                                    break
                        if try_time > self.chat_session.max_allowed_appear:
                            try_time = self.chat_session.max_allowed_appear
                        elif try_time == self.chat_session.max_allowed_appear:
                            self.chat_session.common_questions_asked[quest_key] = 6
                    self.chat_session.context_ref_base = quest_key + ' _cr_ 第{}次'.format(try_time)
                return ret_text, 6  # TIMES_RETRY
            else:
                return ret_text, 0
        else:
            return '', 0

    # def get_xgg_birth_year(self):
    #     year = dt.datetime.now().year
    #     return str(year-self.chat_session.bot_age), 0  # Always gg_age years old

    def reset_session(self):
        self.chat_session.reset()
        return 'Session basic information has been reset.', 0

    def scan_rol_get_block(self, in_question, sentence):
        sentence = to_dense_text(sentence)
        ind_list = [(m.start(0), m.end(0)) for m in re.finditer(ROL_GET_CAT, sentence)]
        if len(ind_list) == 0:
            return sentence

        prev_end_idx = -1
        out_sent = ''
        for idx, (start, end) in enumerate(ind_list):
            blk_txt = sentence[start:end].replace("<rol_get>", "").replace("</rol_get>", "").strip()
            blk_txt = blk_txt.replace(':', '：')
            get_data = blk_txt.split('：')
            get_role, get_cate = get_data[0], get_data[1]

            out_text = ''
            if get_role in ['用户', '客户']:
                if get_cate == '称呼':
                    out_text = self._get_user_name()
                elif get_cate == '全名':
                    out_text = self._get_user_name_full()
                elif get_cate == '姓氏':
                    out_text = self._get_user_name_xing()
            elif get_role == '小木瓜':
                if get_cate in ['邮箱', '邮件']:
                    out_text = 'xiaomugua001@gmail.com'
                elif get_cate == '生年':
                    year = dt.datetime.now().year
                    out_text = str(year - self.chat_session.bot_age)

            if out_text:
                if prev_end_idx > 0:
                    out_sent += sentence[prev_end_idx:start]
                else:
                    out_sent += sentence[:start]
                out_sent += out_text
                prev_end_idx = end
            else:
                raise ValueError("抱歉，我还在学习中，暂时无法正确回复你这个问题呢！")

        if prev_end_idx > 0:
            out_sent += sentence[prev_end_idx:]
        return out_sent

    """
    # Rule 1.1: Date/Time, Week Day, LunarDate, JieQi, and JieRi
    """
    # Input sentence is strictly formatted by cut_text_line method in datautil.py
    def scan_cal_qry_block(self, in_question, sentence):
        sentence = to_dense_text(sentence)
        ind_list = [(m.start(0), m.end(0)) for m in re.finditer(CAL_QRY_CAT, sentence)]
        if len(ind_list) == 0:
            return sentence

        prev_end_idx = -1
        out_sent = ''
        for idx, (start, end) in enumerate(ind_list):
            blk_txt = sentence[start:end].replace("<cal_qry>", "").replace("</cal_qry>", "").strip()
            blk_txt = blk_txt.replace(':', '：')
            if '：' in blk_txt:
                qry_data = blk_txt.split('：')
                qry_type = qry_data[0]
                qry_details = qry_data[1].split('；')
            else:
                qry_type = blk_txt
                qry_details = None

            out_text = ''
            if qry_type == '时间':
                out_text = self._get_time()
            elif qry_type == '节气描述':
                out_text = self._get_today_jieqi_desc()
            elif qry_type == '季节描述':
                out_text = self._get_today_season_desc()
            elif qry_details and len(qry_details) == 1:
                if qry_type in ['日期', '公历日期', '阳历日期', '农历日期', '阴历日期', '星期', '礼拜']:
                    day_text = qry_details[0]
                    if qry_type in ['日期', '公历日期', '阳历日期', '农历日期', '阴历日期']:
                        if re.fullmatch(CalPats.get_prev_next_weekday_cat(), day_text):
                            out_text = self._get_date_prev_next_from_weekday(qry_type, day_text)
                    if out_text == '':
                        out_text = self._get_date_weekday(in_question, qry_type, day_text)
                elif qry_type in ['月份', '公历月份', '阳历月份', '农历月份', '阴历月份']:
                    day_text = qry_details[0]
                    out_text = self._get_year_month(qry_type, day_text)
                elif qry_type in ['年份', '公历年份', '阳历年份', '农历年份', '阴历年份']:
                    year_text = qry_details[0]
                    out_text = self._get_year(qry_type, year_text)
                elif qry_type in ['日期描述', '公历日期描述', '阳历日期描述', '农历日期描述', '阴历日期描述']:
                    day_text = qry_details[0]
                    if re.fullmatch(CalPats.get_week_and_day_cat(), day_text):
                        out_text = self._get_date_with_span_from_weekday(qry_type, day_text)
                    elif re.fullmatch(JieRi.get_jieri_cat(), day_text) or \
                            (day_text.startswith('首次') and re.fullmatch(JieRi.get_jieri_cat(), day_text[2:])):
                        out_text = self._get_jieri_date(in_question, qry_type, day_text)
                elif qry_type in ['地区时间', '地区日期']:
                    area_name = qry_details[0]
                    if area_name in ['小木瓜处', '小香瓜处']:
                        out_text = self._get_xgg_local_area_info(qry_type[2:])
                    else:
                        out_text = self._get_area_time(qry_type, area_name)
                elif qry_type == '时区':
                    area_name = qry_details[0]
                    if area_name in ['小木瓜处', '小香瓜处']:
                        out_text = self._get_xgg_local_area_info(qry_type)
                    else:
                        out_text = self._get_timezone_by_area(area_name)
                elif qry_type in ['小木瓜年龄', '小香瓜年龄']:
                    year_text = qry_details[0]
                    out_text = self._get_xgg_age_ni_ent_cal_year(year_text)
            elif qry_details and len(qry_details) == 2:
                if qry_type in ['日期描述', '公历日期描述', '阳历日期描述', '农历日期描述', '阴历日期描述']:
                    if re.fullmatch(CalPats.get_relative_cal_month_cat(), qry_details[0]) and \
                            re.fullmatch(CalPats.get_day_in_month_cat(), qry_details[1]):
                        out_text = self._get_weekday_or_lunar_date_from_month_day(
                            qry_type, qry_details[0], qry_details[1])
                    elif re.fullmatch(CalPats.get_cal_year_cat(), qry_details[0]):
                        if re.fullmatch(CalPats.get_cal_month_day_cat(), qry_details[1]):
                            out_text = self._get_weekday_or_lunar_date_from_date(qry_type, qry_details[0],
                                                                                 qry_details[1])
                        elif re.fullmatch(JieRi.get_jieri_cat(), qry_details[1]):
                            out_text = self._get_jieri_date_of_year(in_question, qry_type, qry_details[1],
                                                                    qry_details[0])
            if out_text:
                if prev_end_idx > 0:
                    out_sent += sentence[prev_end_idx:start]
                else:
                    out_sent += sentence[:start]
                out_sent += out_text
                prev_end_idx = end
            else:
                raise ValueError("抱歉，我还在学习中，暂时还无法正确回复这个时间或日期问题呢！")

        if prev_end_idx > 0:
            out_sent += sentence[prev_end_idx:]
        return out_sent

    def _get_time(self):
        # self.chat_session.set_categ(self.chat_session.get_topic_categ("CURTIME_QUERY"))
        self.chat_session.set_keep_topic(Topic("CURTIME_QUERY", self.params['default_tz_area']), rounds=1)
        self.chat_session.set_pronoun('ta3', self.params['default_tz_area'])

        now = CalUtils.get_now_in_timezone(self.params)
        wfx, hour = SessionFunction._format_datetime_hour(now.hour)
        if now.minute < 10:
            return "{}{}点0{}分".format(wfx, hour, now.minute)
        else:
            return "{}{}点{}分".format(wfx, hour, now.minute)

    def _get_date_weekday(self, in_question, query_type, day_text):
        if day_text in ['今天', '今日', '现在']:
            day_tm = dt.datetime.now()
        else:
            days = CalPats.parse_relative_cal_day_text(day_text)
            day_tm = dt.datetime.now() + dt.timedelta(days=days)

        if query_type in ['农历日期', '阴历日期']:
            # self.chat_session.set_categ('农历日期')
            ld = LunarDate()
            # "{}年（{}年）{}{}".format(lunar_year, hanzi_year, lunar_month, lunar_day)
            date_text = ld.show_lunar_date(day_tm.year, day_tm.month, day_tm.day)
        else:
            date_text = "{}年{}月{}日".format(day_tm.year, day_tm.month, day_tm.day)
            if query_type in ['星期', '礼拜']:
                # self.chat_session.set_categ('是{}几'.format(query_type))
                date_text += "，{}{}".format(query_type,WEEK_DAYS[day_tm.weekday()])
            else:
                if re.search(r'(什\s么|啥)\s(重\s要\s)?日\s子', in_question):
                    # self.chat_session.set_categ('是什么日子')
                    holiday, _ = JieRiGreeting().get_holiday_jieqi_by_date(day_tm.date())
                    if holiday:
                        date_text += '，{}'.format(holiday)
                # else:
                    # self.chat_session.set_categ('公历日期')
        return date_text

    def _get_year_month(self, query_type, day_text):
        if day_text in ['今天', '今日', '现在']:
            day_tm = dt.datetime.now()
        else:
            days = CalPats.parse_relative_cal_day_text(day_text)
            day_tm = dt.datetime.now() + dt.timedelta(days=days)

        if query_type in ['农历月份', '阴历月份']:
            # self.chat_session.set_categ('是农历几月份')
            ld = LunarDate()
            month_text = ld.show_lunar_date(day_tm.year, day_tm.month, day_tm.day, show_type=2)
        elif query_type in ['月份', '公历月份', '阳历月份']:
            # self.chat_session.set_categ('是几月份')
            month_text = "{}年{}月".format(day_tm.year, day_tm.month)
        else:  # not supposed to come here
            raise ValueError("抱歉，我还在学习中，暂时无法正确回复你的这个关于月份的问题呢！")

        return month_text

    def _get_year(self, query_type, year_text):
        year = None
        today = dt.date.today()
        if year_text in ['今年', '今天', '今日', '现在']:
            year = today.year
        elif CalPats.fullmatch__dist_cal_year_cat(year_text):
            year, _ = CalPats.parse_cal_year_text(today.year, year_text)

        if year:
            if query_type in ['农历年份', '阴历年份']:
                # self.chat_session.set_categ('农历年份')
                ld = LunarDate()
                return ld.show_lunar_date(year, today.month, today.day, show_type=1)  # 不同年的今日
            elif query_type in query_type in ['年份', '公历年份', '阳历年份']:
                # self.chat_session.set_categ('公历年份')
                return "{}年".format(year)

        raise ValueError("抱歉，我还在学习中，暂时无法正确回复你的这个关于年份的问题呢！")
    
    def _get_date_prev_next_from_weekday(self, query_type, day_text):
        weekday_cat = CalPats.get_weekday_only_cat()
        wd_match = re.search(weekday_cat, day_text)
        if not wd_match:
            raise ValueError("晕，我真不知道这个的具体日期呢。")
        wd_text = ''.join(day_text[wd_match.start():wd_match.end()].split())
        if wd_text == '周天':
            raise ValueError("晕，我看不明白这话的意思，所以也无法给你查询具体日期了，抱歉啊！")
        tmp = wd_text.replace('星期', '周').replace('礼拜', '周').replace('周天', '周日')
        day_idx = WEEK_DAYS.index(tmp[-1]) if tmp[-1] in WEEK_DAYS else None
        if day_idx is None:
            raise ValueError("晕，我看不明白这话的意思，所以也无法给你查询具体日期了，抱歉啊！")

        now = dt.datetime.now()
        cur_date = dt.date(now.year, now.month, now.day)
        cur_weekday = cur_date.weekday()

        if query_type in ['农历日期', '阴历日期']:
            self.chat_session.set_categ('农历日期')
        else:
            self.chat_session.set_categ('公历日期')

        if day_text.startswith('过去'):
            rep_idx = -1
        elif day_text.startswith('未来'):
            rep_idx = 1
        else:
            raise ValueError("晕，我看不明白这话的意思，所以也无法给你查询具体日期了，抱歉啊！")

        span = self._get_delta_prev_or_next_weekday(rep_idx, day_idx, cur_weekday)
        if query_type in ['农历日期', '阴历日期']:
            self.chat_session.set_keep_topic(
                Topic("DATE_SPAN_QUERY_WEEKDAY", "NONGLI=REPEAT={}={}".format(rep_idx, day_idx)), rounds=1)
            nongli_text = self._get_nongli_text_presentation_with_day_delta(span, cur_date)
            return nongli_text
        else:
            self.chat_session.set_keep_topic(
                Topic("DATE_SPAN_QUERY_WEEKDAY", "GONGLI=REPEAT={}={}".format(-1, day_idx)), rounds=1)
            date_text, span_text = self._get_text_presentation_with_day_delta(span, cur_date, now)
            return "{}，{}".format(date_text, span_text)

    def _get_date_with_span_from_weekday(self, query_type, cal_week_and_day: str):
        wd_desc0, span_idx, repeat_idx, day_idx = CalPats.parse_week_and_day_text(cal_week_and_day)

        if (span_idx is None and repeat_idx is None) or day_idx is None:
            return "晕，我真不知道{}的具体日期呢。".format(wd_desc0)

        if query_type in ['农历日期描述', '阴历日期描述']:
            self.chat_session.set_categ('农历日期')
            gong_nong = "NONGLI"
        else:
            self.chat_session.set_categ('公历日期')
            gong_nong = "GONGLI"

        if span_idx is not None:
            self.chat_session.set_keep_topic(
                Topic("DATE_SPAN_QUERY_WEEKDAY", "{}=SPAN={}={}".format(gong_nong, span_idx, day_idx)), rounds=1)
        else:
            self.chat_session.set_keep_topic(
                Topic("DATE_SPAN_QUERY_WEEKDAY", "{}=REPEAT={}={}".format(gong_nong, repeat_idx, day_idx)), rounds=1)

        now = dt.datetime.now()
        cur_date = dt.date(now.year, now.month, now.day)
        cur_weekday = cur_date.weekday()

        if self.chat_session.week_span_idx_explained:
            extra = ''
        else:
            self.chat_session.week_span_idx_explained = True
            extra = "（周日为一周的最后一天）"

        if span_idx is not None:
            span = span_idx * 7 + (day_idx - cur_weekday)
            if query_type in ['农历日期描述', '阴历日期描述']:
                nongli_text = self._get_nongli_text_presentation_with_day_delta(span, cur_date)
                return "{}{}是{}。".format(wd_desc0, extra, nongli_text)
            else:
                date_text, span_text = self._get_text_presentation_with_day_delta(span, cur_date, now)
                return "{}{}是{}，{}。".format(wd_desc0, extra, date_text, span_text)
        else:  # repeat_idx is not None
            span = repeat_idx * 7 + (day_idx - cur_weekday)
            repeat_span = self._get_delta_prev_or_next_weekday(repeat_idx, day_idx, cur_weekday)

            nongli_text, date_text, span_text = '', '', ''
            if query_type in ['农历日期描述', '阴历日期描述']:
                nongli_text = self._get_nongli_text_presentation_with_day_delta(span, cur_date)
            else:
                date_text, span_text = self._get_text_presentation_with_day_delta(span, cur_date, now)

            if repeat_span == span:
                if query_type in ['农历日期描述', '阴历日期描述']:
                    return "{}{}是{}。".format(wd_desc0, extra, nongli_text)
                else:
                    return "{}{}是{}，{}。".format(wd_desc0, extra, date_text, span_text)
            else:
                wd_opts = ['上上', '上', '这', '下', '下下']
                d0_text = "{}一个周{}".format(wd_opts[repeat_idx+2], WEEK_DAYS[day_idx])
                d1_text = "{}周的周{}".format(wd_opts[repeat_idx+2], WEEK_DAYS[day_idx])

                if query_type in ['农历日期描述', '阴历日期描述']:
                    nongli0_text = self._get_nongli_text_presentation_with_day_delta(repeat_span, cur_date)
                    return "{}是{}；而{}{}是{}。".format(d0_text, nongli0_text, d1_text, extra, nongli_text)
                else:
                    date0_text, span0_text = self._get_text_presentation_with_day_delta(repeat_span, cur_date, now)
                    return "{}是{}，{}；而{}{}是{}，{}。".format(
                        d0_text, date0_text, span0_text, d1_text, extra, date_text, span_text)

    def _get_weekday_or_lunar_date_from_month_day(self, query_type, cal_month, day_in_month):
        now = dt.datetime.now()
        cur_year, cur_month = now.year, now.month

        month_span_idx, month_desc = CalPats.parse_cal_month_text(cal_month)
        if month_span_idx is None:
            return "晕，我真不换算不出这个的具体日期呢。"

        ask_month = cur_month + month_span_idx
        if ask_month <= 0:
            ask_month += 12
            ask_year = cur_year - 1
        elif ask_month > 12:
            ask_month -= 12
            ask_year = cur_year + 1
        else:
            ask_year = cur_year

        day_in_month_idx = CalPats.parse_day_in_month_text(day_in_month)
        if day_in_month_idx is None:
            return "晕，我真不换算不出这个的具体日期呢。"
        if day_in_month_idx > 0:
            ask_day = day_in_month_idx
            if (ask_day >= 31 and ask_month not in [1, 3, 5, 7, 8, 10, 12]) or (ask_day >= 30 and ask_month == 2):
                return "你真会开玩笑，{}（{}月）也没有{}号呀。".format(month_desc, ask_month, ask_day)
            if ask_month == 2 and ask_day == 29 and not CalUtils.is_leap_year(ask_year):
                return "{}年不是闰年，所以2月也没有29号呀。".format(ask_year)
        else:  # < 0
            if ask_month == 2:
                max_days = 29 if CalUtils.is_leap_year(ask_year) else 28
                if abs(day_in_month_idx) > max_days:
                    return "{}年的2月也没有{}天呀。".format(ask_year, abs(day_in_month_idx))
            elif ask_month in [1, 3, 5, 7, 8, 10, 12]:
                max_days = 31
            else:
                max_days = 30
                if abs(day_in_month_idx) > max_days:
                    return "{}（{}月）也没有{}天呀。".format(month_desc, ask_month, abs(day_in_month_idx))
            ask_day = max_days + day_in_month_idx + 1

        if day_in_month.startswith('正数') or day_in_month.startswith('倒数'):
            ask_date_text = "{}的{}（{}年{}月{}日）".format(month_desc, day_in_month, ask_year, ask_month, ask_day)
        else:
            ask_date_text = "{}{}（{}年{}月{}日）".format(month_desc, day_in_month, ask_year, ask_month, ask_day)

        if query_type in ['农历日期描述', '阴历日期描述']:
            self.chat_session.set_categ('农历日期')
            ld = LunarDate()
            return "{}是{}。".format(ask_date_text, ld.show_lunar_date(ask_year, ask_month, ask_day))
        else:
            self.chat_session.set_categ('公历日期')

            cur_date = dt.date(now.year, now.month, now.day)
            ask_date = dt.date(ask_year, ask_month, ask_day)

            out_text = "{}是{}".format(ask_date_text, "星期" + WEEK_DAYS[ask_date.weekday()])
            span = (ask_date - cur_date).days
            if span in [-3, -2, -1, 0, 1, 2, 3]:
                opts = ['大前天', '前天', '昨天', '今天', '明天', '后天', '大后天']
                out_text += "，也就是{}。".format(opts[span + 3])
            elif span > 0:
                out_text += "，距今还有{}天。".format(span)
            else:
                out_text += "，已经过去{}天。".format(-1 * span)
            return out_text

    def _get_weekday_or_lunar_date_from_date(self, query_type, cal_year, cal_month_day):
        now = dt.datetime.now()

        c_year, y_name = CalPats.parse_cal_year_text(now.year, cal_year)
        if c_year < 1901 or c_year > 2099:  # This should not happen, as the pattern already checks this
            return "我的日历里没有早于公元1901年或迟于2099年的内容，你要问的日期信息我查不到，抱歉啊。"

        c_month, c_day = CalPats.parse_cal_month_day_text(cal_month_day)
        if c_month is None or c_day is None:
            return "晕，我真不换算不出这个的具体日期呢。"
        if (c_day >= 31 and c_month not in [1, 3, 5, 7, 8, 10, 12]) or (c_day >= 30 and c_month == 2):
            return "你真会开玩笑，{}月也没有{}号呀。".format(c_month, c_day)
        if c_month == 2 and c_day == 29 and not CalUtils.is_leap_year(c_year):
            return "{}年不是闰年，所以那年的2月也没有29号呀。".format(c_year)

        ask_date_text = "{}的{}月{}日".format(y_name, c_month, c_day)

        if query_type in ['农历日期描述', '阴历日期描述']:
            self.chat_session.set_categ('农历日期')
            ld = LunarDate()
            return "{}是{}。".format(ask_date_text, ld.show_lunar_date(c_year, c_month, c_day))
        else:
            self.chat_session.set_categ('公历日期')

            cur_date = dt.date(now.year, now.month, now.day)
            ask_date = dt.date(c_year, c_month, c_day)

            out_text = "{}是{}".format(ask_date_text, "星期" + WEEK_DAYS[ask_date.weekday()])
            span = (ask_date - cur_date).days
            if span in [-3, -2, -1, 0, 1, 2, 3]:
                opts = ['大前天', '前天', '昨天', '今天', '明天', '后天', '大后天']
                out_text += "，也就是{}。".format(opts[span + 3])
            elif span > 0:
                out_text += "，距今还有{}天。".format(span)
            else:
                out_text += "，已经过去{}天。".format(-1 * span)
            return out_text

    @staticmethod
    def _get_text_presentation_with_day_delta(span: int, cur_date: dt.date, now: dt.datetime):
        ret_date = cur_date + dt.timedelta(days=span)
        if span >= 0:
            if span in [0, 1, 2, 3]:
                opts = ['今天', '明天', '后天', '大后天']
                span_text = "也就是{}".format(opts[span])
            else:
                span_text = "距今还有{}天".format(span)
            if ret_date.year != now.year:
                date_text = "明年的{}月{}日".format(ret_date.month, ret_date.day)
            else:
                date_text = "{}月{}日".format(ret_date.month, ret_date.day)
        else:
            if span in [-3, -2, -1]:
                opts = ['大前天', '前天', '昨天']
                span_text = "也就是{}".format(opts[span + 3])
            else:
                span_text = "已经过去{}天".format(-1 * span)
            if ret_date.year != now.year:
                date_text = "去年的{}月{}日".format(ret_date.month, ret_date.day)
            else:
                date_text = "{}月{}日".format(ret_date.month, ret_date.day)

        return date_text, span_text

    @staticmethod
    def _get_nongli_text_presentation_with_day_delta(span: int, cur_date: dt.date):
        ret_date = cur_date + dt.timedelta(days=span)
        y, m, d = ret_date.year, ret_date.month, ret_date.day
        gongli_date_text = "公历{}年{}月{}日".format(y, m, d)
        nongli_date_text = LunarDate().show_lunar_date(y, m, d)
        return "{}（{}）".format(nongli_date_text, gongli_date_text)

    @staticmethod
    def _get_delta_prev_or_next_weekday(repeat_idx, day_idx, base_weekday):
        if repeat_idx < 0:
            prev_delta = (7 + base_weekday - day_idx) % 7
            if prev_delta == 0:
                prev_delta = 7
            if repeat_idx == -1:
                return -1 * prev_delta
            else:
                return -1 * prev_delta - abs(repeat_idx + 1) * 7
        else:
            next_delta = (7 + day_idx - base_weekday) % 7
            if next_delta == 0:
                next_delta = 7
            if repeat_idx == 1:
                return next_delta
            else:
                return next_delta + (repeat_idx - 1) * 7

    def _get_jieri_date(self, in_question, query_type, holiday_name):
        if holiday_name.startswith('首次'):
            self.chat_session.set_keep_topic(Topic("JIERI_QUERY", "{}=NO_NEXT".format(holiday_name)), rounds=1)
            self.chat_session.set_categ(self.chat_session.get_topic_categ("JIERI_QUERY"))
            return JieRi.get_first_instance_date(holiday_name[2:])
        else:
            need_next_list = ['多少天', '有几天', '有几日', '要几天', '要几日', '过几天', '过几日',
                              '到哪天', '很多天', '下一个', '下一次', '等不及', '要等']
            need_next_cat = re.compile(r'{}'.format('|'.join([r'\s?'.join(list(nn)) for nn in need_next_list])))
            if re.search(need_next_cat, in_question):
                return self._get_jieri_date_maybe_next(query_type, holiday_name)
            else:
                return self._get_jieri_date_and_weekday(query_type, holiday_name)

    def _get_jieri_date_and_weekday(self, query_type, holiday_name):
        self.chat_session.set_keep_topic(Topic("JIERI_QUERY", "{}=NO_NEXT".format(holiday_name)), rounds=1)
        self.chat_session.set_categ(self.chat_session.get_topic_categ("JIERI_QUERY"))

        out_text = ''
        out_list = JieRi().get_all_holiday_date(holiday_name)
        out_len = len(out_list)

        if out_len == 2 and holiday_name not in self.chat_session.jieri_name_explained_list:
            out_text = "{}可以理解为{}或{}。".format(holiday_name, out_list[0].jr_name, out_list[1].jr_name)
            self.chat_session.jieri_name_explained_list.append(holiday_name)

        last_weekday_id = -1
        for ii in range(out_len):
            jr_info = out_list[ii]
            jr_name, jr_date, jr_type = jr_info.jr_name, jr_info.jr_date, jr_info.jr_type
            year, month, day = jr_date.year, jr_date.month, jr_date.day
            # format weekday
            weekday_id = jr_date.weekday()
            if weekday_id == last_weekday_id:
                weekday = "也是星期" + WEEK_DAYS[weekday_id]
            else:
                weekday = "星期" + WEEK_DAYS[weekday_id]
            # prepare the main output (month, day, and weekday)
            if out_len == 2 and ii == 1:
                out_text += "而{}则是{}月{}日，{}".format(jr_name, month, day, weekday)
            elif out_len == 1 and jr_name in ['春节', '除夕']:
                out_text += "今年（公历年）的{}是{}月{}日，{}".format(jr_name, month, day, weekday)
            else:
                out_text += "今年的{}是{}月{}日，{}".format(jr_name, month, day, weekday)

            # include some extra notes, if needed
            if jr_type == "公历中国":
                out_text += "（以中国节日日期为准）"
            elif jr_type in ["公历美国", "公历周"] and not re.search(r'^(美国|加拿大)', jr_name):
                out_text += "（以美国节日日期为准）"

            if query_type in ['农历日期描述', '阴历日期描述']:
                out_text += "，{}".format(LunarDate().show_lunar_date(year, month, day))

            # end the sentence
            out_text += "。" if ii == out_len - 1 else "；"

            if ii == 0:
                last_weekday_id = weekday_id

        return out_text

    def _get_jieri_date_maybe_next(self, query_type, holiday_name):
        self.chat_session.set_keep_topic(Topic("JIERI_QUERY", "{}=MAYBE_NEXT".format(holiday_name)), rounds=1)
        self.chat_session.set_categ(self.chat_session.get_topic_categ("JIERI_QUERY"))

        out_text = ''
        now = dt.datetime.now()
        cur_date = dt.date(now.year, now.month, now.day)
        out_list = JieRi(now.year).get_all_holiday_date(holiday_name)
        out_len = len(out_list)
        # always has the same length as out_list
        next_list = JieRi(now.year + 1).get_all_holiday_date(holiday_name)

        if out_len == 2 and holiday_name not in self.chat_session.jieri_name_explained_list:
            out_text = "{}可以理解为{}或{}。".format(holiday_name, out_list[0].jr_name, out_list[1].jr_name)
            self.chat_session.jieri_name_explained_list.append(holiday_name)

        for ii in range(out_len):
            jr_info = out_list[ii]
            jr_name, jr_date, jr_type = jr_info.jr_name, jr_info.jr_date, jr_info.jr_type
            year, month, day = jr_date.year, jr_date.month, jr_date.day
            y0_text = '今年（公历年）' if jr_name in ['春节', '除夕'] else '今年'

            # prepare the main output (month, day)
            if out_len == 2 and ii == 1:
                out_text += "至于{}，{}的是{}月{}日".format(jr_name, y0_text, month, day)
            else:
                out_text += "{}的{}是{}月{}日".format(y0_text, jr_name, month, day)

            # include some extra notes, if needed
            if jr_type == "公历中国":
                out_text += "（以中国节日日期为准）"
            elif jr_type in ["公历美国", "公历周"] and not re.search(r'^(美国|加拿大)', jr_name):
                out_text += "（以美国节日日期为准）"

            if query_type in ['农历日期描述', '阴历日期描述']:
                out_text += "，{}".format(LunarDate().show_lunar_date(year, month, day))

            span = (jr_date - cur_date).days
            if span >= 0:
                if span in [0, 1, 2, 3]:
                    opts = ['今天', '明天', '后天', '大后天']
                    out_text += "，也就是{}。".format(opts[span])
                else:
                    out_text += "，距今还有{}天。".format(span)
            else:
                if span in [-3, -2, -1]:
                    opts = ['大前天', '前天', '昨天']
                    out_text += "，也就是{}；".format(opts[span+3])
                else:
                    out_text += "，已经过去{}天；".format(-1*span)
                # make use of the data for the next year only if the holiday in current year has already passed
                next_date = next_list[ii].jr_date
                next_m, next_d, next_span = next_date.month, next_date.day, (next_date - cur_date).days
                if query_type in ['农历日期描述', '阴历日期描述']:
                    ny_extra = "，{}".format(LunarDate().show_lunar_date(year+1, next_m, next_d))
                else:
                    ny_extra = ""
                if next_m == month and next_d == day:
                    out_text += "明年的也是{}月{}日{}，距今还有{}天。".format(next_m, next_d, ny_extra, next_span)
                else:
                    out_text += "明年的则是{}月{}日{}，距今还有{}天。".format(next_m, next_d, ny_extra, next_span)

        return out_text

    def _get_jieri_date_of_year(self, in_question, query_type, holiday_name, cal_year):
        self.chat_session.set_keep_topic(Topic("JIERI_QUERY", "{}={}".format(holiday_name, cal_year)), rounds=1)
        self.chat_session.set_categ(self.chat_session.get_topic_categ("JIERI_QUERY"))

        now = dt.datetime.now()
        cur_date = dt.date(now.year, now.month, now.day)

        c_year, y_name = CalPats.parse_cal_year_text(now.year, cal_year)
        if c_year < 1901 or c_year > 2099:  # This should not happen, as the pattern already checks this
            return "我的日历里没有早于公元1901年或迟于2099年的内容，你要问的节日日期我查不到，抱歉啊。"

        out_text = ''
        out_list = JieRi(c_year).get_all_holiday_date(holiday_name)
        out_len = len(out_list)

        if out_len == 2 and holiday_name not in self.chat_session.jieri_name_explained_list:
            out_text = "{}可以理解为{}或{}。".format(holiday_name, out_list[0].jr_name, out_list[1].jr_name)
            self.chat_session.jieri_name_explained_list.append(holiday_name)

        last_weekday_id = -1
        for ii in range(out_len):
            jr_info = out_list[ii]
            jr_name, jr_date, jr_type = jr_info.jr_name, jr_info.jr_date, jr_info.jr_type
            year, month, day = jr_date.year, jr_date.month, jr_date.day
            # deal with special cases for certain holidays
            year_ex = JieRi.get_year_exception(year, jr_name)
            if year_ex:
                return year_ex

            # prepare the main output (month, day, and weekday)
            if out_len == 2 and ii == 1:
                out_text += "而{}则是{}月{}日".format(jr_name, month, day)
            elif out_len == 1 and jr_name in ['春节', '除夕'] and not y_name[0].isdigit():
                out_text += "{}（公历年）的{}是{}月{}日".format(y_name, jr_name, month, day)
                out_text = out_text.replace('）（公历年）', '，公历年）')
            else:
                out_text += "{}的{}是{}月{}日".format(y_name, jr_name, month, day)

            # include some extra notes, if needed
            if jr_type == "公历中国":
                out_text += "（以中国节日日期为准）"
            elif jr_type in ["公历美国", "公历周"] and not re.search(r'^(美国|加拿大)', jr_name):
                out_text += "（以美国节日日期为准）"

            # format nongli and/or weekday
            present_ny = True if query_type in ['农历日期描述', '阴历日期描述'] else False
            weekday_id = jr_date.weekday()
            if not present_ny or re.search(r'(星\s期|礼\s拜|周)\s几', in_question):
                if weekday_id == last_weekday_id:
                    weekday = "也是星期" + WEEK_DAYS[weekday_id]
                else:
                    weekday = "星期" + WEEK_DAYS[weekday_id]
                out_text += "，{}".format(weekday)
            if present_ny:
                out_text += "，{}".format(LunarDate().show_lunar_date(year, month, day))

            span = (jr_date - cur_date).days
            if span in [-3, -2, -1, 0, 1, 2, 3]:
                opts = ['大前天', '前天', '昨天', '今天', '明天', '后天', '大后天']
                out_text += "，也就是{}".format(opts[span + 3])
            elif span > 0:
                out_text += "，距今还有{}天".format(span)
            else:
                out_text += "，已经过去{}天".format(-1 * span)

            out_text += "。" if ii == out_len - 1 else "；"

            if ii == 0:
                last_weekday_id = weekday_id

        return out_text

    @staticmethod
    def _get_today_jieqi_desc():
        now = dt.datetime.now()
        jq = JieQi()
        jieqi, day_count, _ = jq.get_jieqi_by_date(now.year, now.month, now.day)
        if day_count == 0:
            return "今天{}呀".format(jieqi)
        elif day_count < 0 and jieqi == '小寒':
            return "冬至已经过了好些天，就快小寒了"
        else:  # day_count > 0
            next_jieqi = jq.get_next_jieqi_by_this(jieqi)
            if day_count == 1:
                return "昨天{}啊".format(jieqi)
            elif day_count == 2:
                return "前天{}啦".format(jieqi)
            if day_count < 5:
                return "{}刚刚三五天".format(jieqi)
            elif day_count < 10:
                return "{}已经过去，再过几天就{}了".format(jieqi, next_jieqi)
            else:
                return "{}已经过了好些天，就快{}了".format(jieqi, next_jieqi)

    @staticmethod
    def _get_today_season_desc():
        # 现在是...
        now = dt.datetime.now()
        jq = JieQi()
        jieqi, day_count, season = jq.get_jieqi_by_date(now.year, now.month, now.day)
        next_jieqi = jq.get_next_jieqi_by_this(jieqi)
        if jieqi in ['立春', '立夏', '立秋', '立冬']:
            if day_count == 0:
                return "{}季，今天{}".format(season, jieqi)
            elif day_count == 1:
                return "{}季，昨天{}".format(season, jieqi)
            elif day_count == 2:
                return "{}季，前天{}".format(season, jieqi)
            elif day_count < 5:
                return "{}季，{}刚刚三五天".format(season, jieqi)
            elif day_count < 10:
                return "{}季，{}刚刚没多久".format(season, jieqi)
        elif next_jieqi in ['立春', '立夏', '立秋', '立冬']:
            return "{}季，不过快要{}了".format(season, next_jieqi)
        return "{}季呀".format(season)

    @staticmethod
    def _format_datetime_hour(given_hour):
        if given_hour == 12:
            wfx = '中午'
            hour = 12
        elif given_hour > 18:
            wfx = '晚上'
            hour = given_hour - 12
        elif given_hour > 12:
            wfx = '下午'
            hour = given_hour - 12
        elif given_hour >= 6:
            wfx = '上午'
            hour = given_hour
        else:
            wfx = '凌晨'
            hour = given_hour
        return wfx, hour

    def _get_area_time(self, qry_type, zh_area):
        # self.chat_session.set_categ(self.chat_session.get_topic_categ("CURTIME_QUERY"))
        self.chat_session.set_keep_topic(Topic("CURTIME_QUERY", zh_area), rounds=1)
        self.chat_session.set_pronoun('ta3', zh_area)

        if qry_type == '地区日期':
            self.chat_session.set_categ("问询现在日期")
            for_date = True
        else:
            self.chat_session.set_categ("问询现在时间")
            for_date = False
        ret_text = self._get_time_text_by_area(zh_area, for_date)

        return ret_text

    @staticmethod
    def _get_trimmed_chinese_location(loc_text):
        return re.sub(
            r'中国|省|市|自治区|行政区', '',
            re.sub(r'壮族自治区|回族自治区|维吾尔自治区|维族自治区|维吾尔族自治区|特别行政区', '', loc_text)
        )

    def _get_time_area_city_key_desc(self, area_name):
        new_area = area_name if area_name == '中国' else SessionFunction._get_trimmed_chinese_location(area_name)
        if new_area:
            city_key_desc = self.knowledge_base.timezone_cities_dict.get(new_area)
            if not city_key_desc:
                new_area2 = self.knowledge_base.parse_loc_text_capital(new_area)
                if new_area2 in self.knowledge_base.timezone_cities_dict:
                    city_key_desc = self.knowledge_base.timezone_cities_dict[new_area2]
                    new_area = new_area2
            return city_key_desc, new_area
        return None, new_area

    def _get_time_text_by_area(self, area_name, for_date=False, for_xgg=False):
        city_key_desc, new_area = self._get_time_area_city_key_desc(area_name)
        if city_key_desc:
            if '+' in city_key_desc.eng_city_key:
                city_list = city_key_desc.eng_city_key.split('+')
                city_cnt = len(city_list)
                same_ret, diff_ret = '', ''
                last_date, last_wday, use_same = None, None, True,
                now_in_sys_tz = dt.datetime.now()
                for ii, city in enumerate(city_list):
                    city = city.strip()
                    city_key = self.knowledge_base.timezone_cities_dict[city]
                    utc_ost = CalUtils.get_utc_offset(city_key.eng_city_key)
                    dd_txt, ww_txt, tt_txt = self._format_date_text_by_area(city_key.eng_city_key, now_in_sys_tz)
                    same_ret += "{}. {}（GMT{}）：{}".format((ii+1), city, utc_ost, tt_txt)
                    same_ret += '。' if ii == city_cnt - 1 else '；_nl_'
                    if for_date:
                        diff_ret += "{}. {}（GMT{}）：{}（{}），{}".format((ii+1), city, utc_ost, dd_txt, ww_txt, tt_txt)
                    else:
                        diff_ret += "{}. {}（GMT{}）：{}，{}".format((ii+1), city, utc_ost, dd_txt, tt_txt)
                    diff_ret += '。' if ii == city_cnt - 1 else '；_nl_'
                    if last_date and dd_txt != last_date:
                        use_same = False
                    last_date, last_wday = dd_txt, ww_txt
                if use_same:
                    ret_text = "{}覆盖多个时区，所有这些时区现在均为{}，{}。" \
                               "以下是其不同时区代表城市的具体时间：_np_".format(
                        city_key_desc.zh_city_desc, last_date, last_wday)
                    ret_text += same_ret
                else:
                    ret_text = "{}覆盖多个时区，而且这些时区现在不在同一天。" \
                               "以下是其不同时区代表城市的日期及时间：_np_".format(city_key_desc.zh_city_desc)
                    ret_text += diff_ret
            else:
                if city_key_desc.eng_city_key == 'CN':
                    ask_tz_id = 'Asia/Shanghai'
                    extra = ''
                else:
                    ask_tz_id = city_key_desc.eng_city_key
                    utc_offset = CalUtils.get_utc_offset(ask_tz_id)
                    extra = '' if utc_offset.endswith('00') else "（时区：GMT{}）".format(utc_offset)
                if for_date:
                    dat_txt, wdy_txt, tim_txt = self._format_date_text_by_area(ask_tz_id)
                    ret_text = "{}{}现在是{}，{}。具体时间是{}。".format(
                        city_key_desc.zh_city_desc, extra, dat_txt, wdy_txt, tim_txt)
                else:
                    dt_text = self._format_time_text_by_area(ask_tz_id)
                    if for_xgg:
                        ret_text = "我这里现在是{}。".format(dt_text)
                    else:
                        ret_text = "{}{}现在的时间是{}。".format(city_key_desc.zh_city_desc, extra, dt_text)
        else:
            if new_area[-1] in ['区', '县', '乡', '镇']:
                ret_text = "时间问询仅限于国家，省份或城市级别，无法具体到区县或乡镇。抱歉啊！"
            else:
                print("Area not found for: {}".format(area_name))
                ret_text = "抱歉，我这里暂时没有它的时间信息，无法满足您的查询，不好意思哈。"
        return ret_text

    def _format_date_text_by_area(self, ask_tz_id, now_in_sys_tz=None):
        ask_dt = CalUtils.get_now_in_timezone(self.params, tz_id=ask_tz_id, now_in_sys_tz=now_in_sys_tz)
        wfx, hour = SessionFunction._format_datetime_hour(ask_dt.hour)
        date_text = "{}年{}月{}日".format(ask_dt.year, ask_dt.month, ask_dt.day)
        wday_text = "星期" + WEEK_DAYS[ask_dt.weekday()]
        if ask_dt.minute < 10:
            time_text = "{}{}点0{}分".format(wfx, hour, ask_dt.minute)
        else:
            time_text = "{}{}点{}分".format(wfx, hour, ask_dt.minute)
        return date_text, wday_text, time_text

    def _format_time_text_by_area(self, ask_tz_id):
        ask_dt = CalUtils.get_now_in_timezone(self.params, tz_id=ask_tz_id)
        wfx, hour = SessionFunction._format_datetime_hour(ask_dt.hour)
        weekday = "星期" + WEEK_DAYS[ask_dt.weekday()]
        if ask_dt.minute < 10:
            dt_text = "{}年{}月{}日（{}）{}{}点0{}分".format(
                ask_dt.year, ask_dt.month, ask_dt.day, weekday, wfx, hour, ask_dt.minute)
        else:
            dt_text = "{}年{}月{}日（{}）{}{}点{}分".format(
                ask_dt.year, ask_dt.month, ask_dt.day, weekday, wfx, hour, ask_dt.minute)
        return dt_text

    def _get_xgg_local_area_info(self, info_cat):
        gg_loc = self.chat_session.bot_loc
        if info_cat in ['时间', '日期', '时区']:
            self.chat_session.set_categ(self.chat_session.get_topic_categ("CURTIME_QUERY"))
            self.chat_session.set_keep_topic(Topic("CURTIME_QUERY", gg_loc), rounds=1)
            self.chat_session.set_pronoun('ta3', gg_loc)
            if info_cat == '时间':
                return self._get_time_text_by_area(gg_loc, for_xgg=True)
            elif info_cat == '日期':
                return self._get_time_text_by_area(gg_loc, for_date=True)
            else:
                ct_ky_desc = self.knowledge_base.timezone_cities_dict.get(gg_loc)
                if ct_ky_desc:
                    ask_tz_id = 'Asia/Shanghai' if ct_ky_desc.eng_city_key == 'CN' else ct_ky_desc.eng_city_key
                    std_text, cur_text = CalUtils.get_timezone_texts(ask_tz_id)
                    if cur_text is None:
                        return "{}的时区是{}。".format(gg_loc, std_text)
                    else:
                        return "{}的标准时区是{}，当前夏令时偏移为{}。".format(gg_loc, std_text, cur_text)
        elif info_cat == '天气':
            self.chat_session.set_keep_topic(Topic('CITY_WEATHER', ''))
            self.chat_session.set_categ(self.chat_session.get_topic_categ('CITY_WEATHER'))
            self.chat_session.set_pronoun('ta3', gg_loc)
            return self._get_weather_text_by_city_name(gg_loc)

        return "不好意思，未能查到你要求的信息，抱歉啊！"

    def _get_timezone_by_area(self, area_name):
        if self.chat_session.last_topic and self.chat_session.last_topic.title == 'CURTIME_QUERY':
            self.chat_session.set_categ(self.chat_session.get_topic_categ("CURTIME_QUERY"))
            self.chat_session.set_keep_topic(Topic("CURTIME_QUERY", area_name), rounds=1)
        else:
            self.chat_session.set_categ("问询时区信息")
        self.chat_session.set_pronoun('ta3', area_name)

        city_key_desc, new_area = self._get_time_area_city_key_desc(area_name)
        if city_key_desc:
            if '+' in city_key_desc.eng_city_key:
                city_list = city_key_desc.eng_city_key.split('+')
                city_cnt = len(city_list)
                ret_text = "{}覆盖多个时区，以下是其不同时区及代表城市：_np_".format(city_key_desc.zh_city_desc)
                for ii, city in enumerate(city_list):
                    city = city.strip()
                    city_key = self.knowledge_base.timezone_cities_dict[city]
                    std_text, cur_text = CalUtils.get_timezone_texts(city_key.eng_city_key)
                    kk = ii + 1
                    if cur_text is None:
                        ret_text += "{}. {}：{}".format(kk, city, std_text)
                    else:
                        ret_text += "{}. {}：{}，当前夏令时偏移为{}".format(kk, city, std_text, cur_text)
                    ret_text += '。' if ii == city_cnt - 1 else '；_nl_'
                return ret_text
            else:
                ask_tz_id = 'Asia/Shanghai' if city_key_desc.eng_city_key == 'CN' else city_key_desc.eng_city_key
                std_text, cur_text = CalUtils.get_timezone_texts(ask_tz_id)
                if cur_text is None:
                    return "{}的时区是{}。".format(new_area, std_text)
                else:
                    return "{}的标准时区是{}，当前夏令时偏移为{}。".format(new_area, std_text, cur_text)

    def _get_xgg_age_ni_ent_cal_year(self, ni_ents):
        self.chat_session.set_categ('{}年龄'.format(self.chat_session.bot_name))

        gg_age = self.chat_session.bot_age
        age_id = gg_age - 1
        age_ss = ['一', '两', '三', '四', '五', '六', '七', '八', '九', '十',
                  '十一', '十二', '十三', '十四', '十五', '十六', '十七', '十八', '十九', '二十']
        if ni_ents == '大前年':
            ret_text = "大前年我才{}岁呢。".format(age_ss[age_id-3])
        elif ni_ents == '前年':
            ret_text = "前年我当然是{}岁啦。".format(age_ss[age_id-2])
        elif ni_ents == '去年':
            ret_text = "去年呀？我才{}岁呢。".format(age_ss[age_id-1])
        elif ni_ents == '今年':
            ret_text = "今年我{}岁呀。".format(age_ss[age_id])
        elif ni_ents == '明年':
            ret_text = "明年呀？我就要{}岁了。".format(age_ss[age_id+1])
        elif ni_ents == '后年':
            ret_text = "后年我就是{}岁了。".format(age_ss[age_id+2])
        elif ni_ents == '大后年':
            ret_text = "这还用问？大后年我当然是{}岁啦。".format(age_ss[age_id+3])
        else:
            cur_year = dt.datetime.now().year
            c_year, y_name = CalPats.parse_cal_year_text(cur_year, ni_ents)
            if ni_ents[-1] in '前后':
                y_name = ni_ents
            yr_diff = c_year - cur_year
            if yr_diff < 0:
                yr_diff = abs(yr_diff)
                if yr_diff <= gg_age-1:
                    age = gg_age-yr_diff
                    if age == 2:
                        ret_text = "{}呀，我才两岁呢。".format(y_name)
                    else:
                        ret_text = "{}呀，我才{}岁呢。".format(y_name, age)
                elif yr_diff == gg_age:
                    ret_text = "{}就是我出生的那一样呀。".format(y_name)
                elif yr_diff <= 50:
                    ret_text = "{}呀，我还没出生呢。".format(y_name)
                else:
                    ret_text = "{}呀，我父母还没出生呢，更别说我了。".format(y_name)
            elif yr_diff > 0:  # assume gg_age is <= 16
                fut_age = gg_age + yr_diff
                if yr_diff <= 17-gg_age:
                    ret_text = "{}我当然是{}岁啦。".format(y_name, fut_age)
                elif yr_diff == 18-gg_age:
                    ret_text = "{}我18岁，那时我就成年了。".format(y_name)
                elif yr_diff <= 80-gg_age:
                    ret_text = "{}呀，我都已经{}岁了呢。".format(y_name, fut_age)
                elif yr_diff <= 99-gg_age:
                    ret_text = "{}呀，如果我还健在，那将会是{}岁了呢。".format(y_name, fut_age)
                elif yr_diff == 100-gg_age:
                    ret_text = "{}呀，如果我还活着，那将是百岁老人了。".format(y_name, fut_age)
                else:
                    ret_text = "想那么久远干嘛，能活到那年月再说吧。"
            else:  #
                ret_text = "那不就是今年吗？我{}岁呀。".format(age_ss[age_id])

        return ret_text

    """
    # Rule 1.2: KDG Query
    """
    # Input sentence is strictly formatted by cut_text_line method in datautil.py
    def scan_kdg_qry_block(self, in_question, sentence):
        sentence = to_dense_text(sentence)
        ind_list = [(m.start(0), m.end(0)) for m in re.finditer(KDG_QRY_CAT, sentence)]
        if len(ind_list) == 0:
            return sentence

        prev_end_idx = -1
        out_sent = ''
        for idx, (start, end) in enumerate(ind_list):
            blk_txt = sentence[start:end].replace("<kdg_qry>", "").replace("</kdg_qry>", "").strip()
            blk_txt = blk_txt.replace(':', '：')
            if '：' in blk_txt:
                qry_data = blk_txt.split('：')
                qry_type = qry_data[0]
                qry_details = qry_data[1].split('；')
            else:
                qry_type = blk_txt
                qry_details = None

            out_text = ''
            if qry_details and len(qry_details) == 1:
                if qry_type == '名人':
                    out_text = self._get_lgst_person_by_key(qry_details[0])
                elif qry_type in ['事物', '作品']:
                    out_text = self._get_lgst_thing_by_key(qry_details[0])
                elif qry_type == '某人是谁':
                    out_text = self._get_celeb_info(qry_details[0])
                elif qry_type in ['某物是啥', '某物是什么']:
                    out_text = self._get_whatis_from_entry(qry_details[0])
                elif qry_type == '某人某物':
                    out_text = self._get_whowhatis_from_entry(qry_details[0])

            if out_text:
                if prev_end_idx > 0:
                    out_sent += sentence[prev_end_idx:start]
                else:
                    out_sent += sentence[:start]
                out_sent += out_text
                prev_end_idx = end
            else:
                if qry_type in ['名人', '某人是谁']:
                    raise ValueError("晕，名人的知识我掌握得真不多。你学识丰富，要不你给我介绍一下吧。")
                elif qry_type == '作品':
                    raise ValueError("晕，文艺作品方面的知识我懂得真不多。你学识丰富，要不你给我介绍一下吧。")
                else:  # qry_type == '事物' OR any other cases
                    raise ValueError(random.choice([
                        "晕，我还从来都没听说过呢。唉，真是无知啊。",
                        "抱歉，我不了解呢。你懂得多，还是你来给讲讲吧。",
                        "晕，我不清楚呢。你学识丰富，要不你给我介绍一下吧。",
                    ]))

        if prev_end_idx > 0:
            out_sent += sentence[prev_end_idx:]
        return out_sent
    """
    # Rule 1.3: Web Query: Weather
    """
    # Input sentence is strictly formatted by cut_text_line method in datautil.py
    def scan_web_qry_block(self, in_question, sentence):
        sentence = to_dense_text(sentence)
        ind_list = [(m.start(0), m.end(0)) for m in re.finditer(WEB_QRY_CAT, sentence)]
        if len(ind_list) == 0:
            return sentence

        prev_end_idx = -1
        out_sent = ''
        for idx, (start, end) in enumerate(ind_list):
            blk_txt = sentence[start:end].replace("<web_qry>", "").replace("</web_qry>", "").strip()
            blk_txt = blk_txt.replace(':', '：')
            if '：' in blk_txt:
                qry_data = blk_txt.split('：')
                qry_type = qry_data[0]
                qry_details = qry_data[1].split('；')
            else:
                qry_type = blk_txt
                qry_details = None

            out_text = ''
            if qry_details and len(qry_details) == 1:
                if qry_type == '天气':
                    city_text = qry_details[0]
                    if city_text in ['小木瓜处', '小香瓜处']:
                        out_text = self._get_xgg_local_area_info(qry_type)
                    else:
                        out_text = self._get_weather_from_desc(city_text)

            if out_text:
                if prev_end_idx > 0:
                    out_sent += sentence[prev_end_idx:start]
                else:
                    out_sent += sentence[:start]
                out_sent += out_text
                prev_end_idx = end
            else:
                raise ValueError("抱歉，我还在学习中，暂时还无法正确回复这个时间或日期问题呢！")

        if prev_end_idx > 0:
            out_sent += sentence[prev_end_idx:]
        return out_sent

    def ask_weather_city(self):
        self.chat_session.set_context_prefix(self.chat_session.city_weather_context_prefix)
        return random.choice(SessionFunction.ask_weather_city_list), 0

    def _get_weather_from_desc(self, zh_city):
        self.chat_session.clear_context_prefix()

        self.chat_session.set_keep_topic(Topic('CITY_WEATHER', ''))
        self.chat_session.set_categ(self.chat_session.get_topic_categ('CITY_WEATHER'))

        self.chat_session.set_pronoun('ta3', zh_city)
        if not self.chat_session.context_topic_weather_added:
            tpc_txt = "对了，你喜欢晴天还是阴天啊？ " \
                      "_func_set_context_ref_cache_para0_flw [我刚才查询天气，你现在问我喜欢晴天还是阴天]"
            self.chat_session.add_context_topic(tpc_txt)
            self.chat_session.context_topic_weather_added = True
            print("******** Topic context added ...")
        ret_text = self._get_weather_text_by_city_name(zh_city)

        return ret_text

    def _get_weather_text_by_city_name(self, zh_city):
        ret_text = ''

        new_city = self._get_trimmed_chinese_location(zh_city)
        if new_city:
            city_key_desc = self.knowledge_base.weather_cities_dict.get(new_city)
            if not city_key_desc:
                new_city = self.knowledge_base.parse_loc_text_capital(new_city)
                city_key_desc = self.knowledge_base.weather_cities_dict.get(new_city)
            if city_key_desc:
                success, weather = self._get_weather_by_city_key(city_key_desc.eng_city_key)
                if success:
                    ret_text = "以下是{}为你查询到的关于{}的天气情况：_np_{}".format(
                        self.chat_session.bot_name, city_key_desc.zh_city_desc, weather.get_text_output()
                    )
                else:
                    ret_text = "晕，{}没能为你查询到{}的天气，不好意思哈。{}".format(
                        self.chat_session.bot_name, city_key_desc.zh_city_desc, CityWeather.FAIL_QUERY_TEXT
                    )
            else:
                if zh_city[-1] in ['区', '县', '乡', '镇']:
                    self.chat_session.set_context_prefix(self.chat_session.city_weather_context_prefix)
                    ret_text = "查询天气现仅限于城市级别，还不能具体到区县或乡镇。请问{}位于或属于哪个城市呢？".format(
                        zh_city
                    )
                else:
                    print("### 无法查询#{}#的天气。".format(zh_city))
                    ret_text = "抱歉，我这里暂时没有它的天气信息，无法满足您的查询，不好意思哈。{}".format(
                        CityWeather.UNK_CITY_TEXT
                    )
        return ret_text

    def _get_weather_by_city_key(self, eng_city_desc):
        city_weather = self.knowledge_base.weather_info_cache.get(eng_city_desc)
        if not city_weather or (time.time() - city_weather.query_time) > 3600:  # more than 1 hour
            print("发起一次新的天气查询，查询城市关键词：{}".format(eng_city_desc))
            success, city_weather = query_weather_by_city(eng_city_desc)
            if success:
                self.knowledge_base.weather_info_cache[eng_city_desc] = city_weather
        else:
            print("成功使用缓存的天气信息，获取城市{}的天气".format(eng_city_desc))
            success = True

        return success, city_weather

    """
    # Rule 2.1: Arithmetic ops, Power ops, or Mixed arithmetic expression
    """
    # Input sentence is strictly formatted by cut_text_line method in datautil.py
    def scan_math_equations(self, sentence):
        sentence = to_dense_text(sentence)
        ind_list = [(m.start(0), m.end(0)) for m in re.finditer(MAT_EQU_CAT, sentence)]
        if len(ind_list) == 0:
            return sentence.replace("<mat_equ>", "").replace("</mat_equ>", "")

        prev_num_list, prev_end_idx = self.chat_session.math_equ_res_list, -1
        out_sent = ''
        for idx, (start, end) in enumerate(ind_list):
            the_equ = sentence[start:end].replace("<mat_equ>", "").replace("</mat_equ>", "").strip()
            res_only, res_abs, res_neg = False, False, False
            shulie_def, fangcheng_def, fangcheng_sol = False, False, False
            if re.fullmatch(r'结果\d+', the_equ):
                main_part = the_equ
                res_only = True
            elif re.fullmatch(r'结果\d+的?绝对值', the_equ):
                main_part = re.sub(r'的?绝对值', '', the_equ)
                res_abs = True
            elif re.fullmatch(r'结果\d+的?负数', the_equ):
                main_part = re.sub(r'的?负数', '', the_equ)
                res_neg = True
            elif re.match(r'^结果\d[?？]=', the_equ):  # no output text, no info to be added into prev_num_list
                nums = re.split(r'[?？]=', the_equ)
                if prev_num_list and len(nums) == 2:
                    res_idx = int(nums[0][2:]) - 1
                    if len(prev_num_list) > res_idx:
                        real_res = prev_num_list[res_idx]
                        left = mp.mpf(real_res.replace(',', ''))
                        r_text = nums[1].replace(',', '')
                        if re.fullmatch(HANZI_RELATED_PN_CAT, r_text):
                            r_num = text2num(r_text)
                            if r_num is None:
                                raise ValueError("矮油，“{}”具体等于多少，我不明白。所以这个运算我真的无能为力呢！".format(r_text))
                        else:
                            r_num = r_text
                        right = mp.mpf(r_num)
                        self.chat_session.math_equ_res_tuple = (left, right)
                continue
            elif re.fullmatch(MAT_SHULIE_DEF_CAT, the_equ):
                parts = the_equ.split('=')
                raw_num_list = parts[0].strip().split('，')
                # shulie_idx = int(parts[1][2:]) - 1
                num_list = []
                for raw_num in raw_num_list:
                    act_num = text2num(raw_num)
                    if act_num is None:
                        raise ValueError("矮油，“{}”具体等于多少，我不明白。所以这个运算我真的无能为力呢！".format(raw_num))
                    num_list.append(act_num)
                self.chat_session.math_equ_shulie_list.append(num_list)
                shulie_def = True
                main_part = parts[0]
            elif re.fullmatch(MAT_SHULIE_CAL_CAT, the_equ):
                main_part = the_equ
            elif re.fullmatch(MAT_FANGCHENG_DEF_CAT, the_equ):
                pre_mat = re.match(r'方程\d+[：:]', the_equ)
                if pre_mat:
                    fc_text = the_equ[pre_mat.end():].strip()
                    hz_num_ind_list = [(m.start(0), m.end(0)) for m in re.finditer(HANZI_RELATED_PN_CAT, fc_text)]
                    if len(hz_num_ind_list) >= 1:
                        pre_hz_end = -1
                        new_fc_text = ''
                        for hz_idx, (hz_s, hz_e) in enumerate(hz_num_ind_list):
                            new_num = text2num(fc_text[hz_s:hz_e])
                            if new_num is None:
                                raise ValueError("矮油，“{}”具体等于多少，我不明白。所以这个运算我真的无能为力呢！".
                                                 format(fc_text[hz_s:hz_e]))
                            if pre_hz_end > 0:
                                new_fc_text += fc_text[pre_hz_end:hz_s]
                            else:
                                new_fc_text += fc_text[:hz_s]
                            new_fc_text += str(new_num)
                            pre_hz_end = hz_e
                    else:
                        new_fc_text = fc_text
                    self.chat_session.math_equ_fangcheng_list.append(new_fc_text)
                    fangcheng_def = True
                    main_part = re.sub(r'\s*([*/^])\s*', r'\1', re.sub(r'(?<![(=*/^+-])([=+-])', r' \1 ', new_fc_text))
                    main_part = re.sub(r'\s+', '&nbsp;', re.sub(r'(?<=^[+-])\s', '', main_part.strip()))
                else:
                    main_part = the_equ  # not supposed to happen
            elif re.fullmatch(MAT_FANGCHENG_SOL_CAT, the_equ):
                fangcheng_sol = True
                main_part = the_equ
            else:  # including 单位转换
                main_part = the_equ[:the_equ.find('=')].strip()

            if res_only or res_abs or res_neg:
                new_main = "未知"
                if prev_num_list:
                    res_idx = int(main_part[2:]) - 1
                    if len(prev_num_list) > res_idx:
                        real_res = prev_num_list[res_idx].replace(',', '')
                        if res_abs:
                            new_main = mp.fabs(real_res)
                        elif res_neg:
                            new_main = -mp.mpf(real_res)
                        else:
                            new_main = mp.mpf(real_res)
            elif shulie_def or fangcheng_def or fangcheng_sol:
                new_main = main_part
            else:  # including 单位转换
                new_main, prev_end_idx2 = '', -1
                if prev_num_list:
                    ind2_list = [(m2.start(0), m2.end(0)) for m2 in re.finditer(MAT_PRE_RES_CAT, main_part)]
                    if len(ind2_list) > 0:
                        for idx2, (s2, e2) in enumerate(ind2_list):
                            res_idx = int(main_part[s2:e2][2:]) - 1
                            if len(prev_num_list) > res_idx:
                                real_res = prev_num_list[res_idx]
                                if prev_end_idx2 > 0:
                                    new_main += main_part[prev_end_idx2:s2]
                                else:
                                    new_main += main_part[:s2]
                                new_main += ' ' + str(real_res)
                                prev_end_idx2 = e2
                        if prev_end_idx2 > 0:
                            new_main += main_part[prev_end_idx2:]

                new_main = main_part if new_main == '' else re.sub(r'\s+', ' ', new_main).strip()

            if res_only or res_abs or res_neg:
                if new_main != "未知":
                    _, prev_num = get_output_number(new_main)
                    out_text = str(prev_num)
                else:
                    out_text = "未知" # not supposed to happen
                # prev_num_list.append(prev_num)
            elif shulie_def or fangcheng_def:
                out_text = new_main
            elif re.fullmatch(MAT_MISC_OPS_CAT, new_main):
                num_text = re.sub(r'[的之求取](绝对值|负数|阶乘)', '', new_main).replace(',', '').replace(' ', '')
                if re.fullmatch(HANZI_RELATED_PN_CAT, num_text):
                    act_num = text2num(num_text)
                    if act_num is None:
                        raise ValueError("矮油，“{}”具体等于多少，我不明白。所以这个运算我真的无能为力呢！".format(num_text))
                else:
                    act_num = num_text
                if '绝对值' in new_main:
                    act_num = mp.fabs(act_num)
                elif '负数' in new_main:
                    act_num = -mp.mpf(act_num)
                else:  # 阶乘
                    act_num = mp.mpf(act_num)
                    if act_num < 0 and mp.isint(act_num):
                        raise ValueError("“{}”是个负整数，可负整数的阶乘没有意义啊！".format(num_text))
                    elif act_num > 10_000:
                        raise OverflowError(
                            "{0}的阶乘是个过于庞大的数字，无法直接计算并显示其完整结果。你可以使用斯特灵公式来估算{0}的阶乘："
                            "_np_ n!&nbsp;≈&nbsp;√(2πn)&nbsp;*&nbsp;(n&nbsp;/&nbsp;e)&nbsp;^&nbsp;n _np_在这里，"
                            "n&nbsp;=&nbsp;{1}，π表示圆周率（约为3.14159265），e表示自然对数的底数（约为2.71828183）。"
                            "_nl_使用该公式，你可以得到一个关于{0}阶乘的近似值。请注意，斯特灵公式只能提供阶乘的近似值，而不"
                            "能精确计算阶乘。".format(num_text, act_num))
                    else:
                        try:
                            act_num = mp.fac(act_num)
                        except OverflowError:
                            raise OverflowError(
                                "晕，“{}”太大（或太小），要计算其阶乘实在超出我能力了，抱歉啊！".format(num_text))
                _, prev_num = get_output_number(act_num)
                out_text = "{}&nbsp;=&nbsp;{}".format(new_main, prev_num)
                prev_num_list.append(prev_num)
            elif re.fullmatch(ARITHMETIC_MIXED_CAT, new_main):
                tree = ArithTree.build(new_main)
                _, prev_num = get_output_number(tree.evaluate())
                tree_out_text = re.sub(r'\s*([*/^])\s*', r'\1', tree.get_output_text())
                tree_out_text = re.sub(r'\)([*/])\(', r') \1 (', tree_out_text)  # when )*( or )/(, add spaces
                tree_out_text = re.sub(r'\s+', '&nbsp;', tree_out_text)
                new_main = new_main.replace('（', '(').replace('）', ')')
                if new_main.replace('π', 'pi').replace('√', '').isascii():  # the replaced value won't be kept
                    out_text = "{}&nbsp;=&nbsp;{}".format(tree_out_text, prev_num)
                else:
                    new_main_text = re.sub(r'\s*([*/^])\s*', r'\1', re.sub(r'(?<![Ee])([+-])', r' \1 ', new_main))
                    new_main_txt = re.sub(r'\s+', '&nbsp;', new_main_text)
                    out_text = "{}&nbsp;=&nbsp;{}&nbsp;=&nbsp;{}".format(new_main_txt, tree_out_text, prev_num)
                prev_num_list.append(prev_num)
            elif re.fullmatch(MAT_UNIT_CONVERT_CAT, new_main):
                trunk_text = new_main[5:].strip()  # 1米82->分米, 2,999,000.08平方公里->亩
                un_parts = trunk_text.split('->')
                unit_with_amount, unit2 = un_parts[0], un_parts[1]
                num1, unit1 = parse_unit_amount_text(unit_with_amount)
                if num1 is None or unit1 is None:
                    raise ValueError("“{}”是什么意思啊，我看不懂呢。所以我无法进行您要求的单位转换了，抱歉啊！"
                                     .format(unit_with_amount))

                u1, u2 = unit1.lower(), unit2.lower()
                cat1 = unit_category_dict.get(u1)
                cat2 = unit_category_dict.get(u2)
                if cat1 is None:
                    raise ValueError("{}这个单位我还没听说过呢，你给我讲解一下呗。".format(unit1))
                elif cat2 is None:
                    raise ValueError("{}这个单位我还没听说过呢，你给我讲解一下呗。".format(unit2))

                if cat1 != cat2:
                    raise ValueError("矮油，我太笨了，不知道怎么将{}单位换算成{}单位呢。".format(cat1, cat2))

                if cat1 == '温度':  # cat1 == cat2 already
                    new_unit2 = formalize_temper_unit(u2)
                    if new_unit2 is None:  # Not supposed to be satisfied
                        raise ValueError("{}这个单位我还没听说过呢，你给我讲解一下呗。".format(unit2))
                    num2 = SessionFunction._get_temper_unit2_amount(u1, num1, new_unit2)
                    if num2 is None:
                        raise ValueError("{}{}已经低于绝对零度了，所以没有意义啊。".format(num1, unit1))
                else:
                    num2 = (num1 / unit_convert_dict.get(u2)) * unit_convert_dict.get(u1)

                _, prev_num = get_output_number(num2)
                out_text = str(prev_num)
                prev_num_list.append(prev_num)
            elif re.fullmatch(MAT_SHULIE_CAL_CAT, new_main):
                sl_idx_mat = re.search(r'数列\d+', new_main)
                s_idx, e_idx = sl_idx_mat.start(), sl_idx_mat.end()
                shulie_idx = int(new_main[s_idx:e_idx][2:]) - 1
                this_num, out_text = None, ''
                if len(self.chat_session.math_equ_shulie_list) > shulie_idx:
                    num_list = self.chat_session.math_equ_shulie_list[shulie_idx]
                    if new_main.endswith('最大值'):
                        this_num = max(num_list)
                    elif new_main.endswith('最小值'):
                        this_num = min(num_list)
                    elif new_main.endswith('平均值'):
                        _, this_num = get_output_number(sum(num_list)/len(num_list))
                    elif new_main.endswith('升序排列'):
                        sorted_list = sorted(num_list)
                        out_text = '， '.join([str(n) for n in sorted_list])
                    elif new_main.endswith('降序排列'):
                        sorted_list = sorted(num_list, reverse=True)
                        out_text = '， '.join([str(n) for n in sorted_list])
                if out_text == '':
                    if this_num is None:
                        raise ValueError("也许我理解有误，反正“{}”不是个数学表达式，我无法计算呢！".format(new_main))
                    else:
                        out_text = str(this_num)
                        prev_num_list.append(this_num)
            elif re.fullmatch(MAT_FANGCHENG_SOL_CAT, new_main):
                fcs_mat = re.match(MAT_FANGCHENG_SOL_CAT, new_main)
                # 方程1，2  解x，y  结果1，2
                fc_id_txt, jie_id_txt, jg_id_txt = fcs_mat.group(1)[2:], fcs_mat.group(2)[1:], fcs_mat.group(3)[2:]
                fc_exp_list, fc_sym_list = [], []
                if jie_id_txt[0] == 'x':
                    x = Symbol('x')
                    fc_sym_list.append(x)
                    if jie_id_txt == 'x，y':
                        y = Symbol('y')
                        fc_sym_list.append(y)
                    elif jie_id_txt == 'x，y，z':
                        y = Symbol('y')
                        z = Symbol('z')
                        fc_sym_list.append(y)
                        fc_sym_list.append(z)
                elif jie_id_txt[0] == 'a':
                    a = Symbol('a')
                    fc_sym_list.append(a)
                    if jie_id_txt == 'a，b':
                        b = Symbol('b')
                        fc_sym_list.append(b)
                    elif jie_id_txt == 'a，b，c':
                        b = Symbol('b')
                        c = Symbol('c')
                        fc_sym_list.append(b)
                        fc_sym_list.append(c)
                fc_ids = fc_id_txt.split('，')
                for fc_id in fc_ids:
                    fc_id_idx = int(fc_id) - 1
                    if fc_id_idx < len(self.chat_session.math_equ_fangcheng_list):
                        fc_equ = self.chat_session.math_equ_fangcheng_list[fc_id_idx]
                        lr_parts = fc_equ.split('=')
                        left, right = parse_expr(lr_parts[0]), parse_expr(lr_parts[1])
                        fc_exp_list.append(Eq(left, right))
                fc_res_dict = solve(fc_exp_list, fc_sym_list)
                out_text = ''
                for sym, val in fc_res_dict.items():
                    _, fc_pre_num = get_output_number(val)
                    if out_text != '':
                        out_text += '，'
                    out_text += str(sym) + ' = {}'.format(fc_pre_num)
                    prev_num_list.append(fc_pre_num)
            else:
                raise ValueError("也许我理解有误，反正“{}”不是个数学表达式，我无法计算呢！".format(new_main))
            if prev_end_idx > 0:
                out_sent += sentence[prev_end_idx:start]
            else:
                out_sent += sentence[:start]
            out_sent += out_text
            prev_end_idx = end

        if prev_end_idx > 0:
            out_sent += sentence[prev_end_idx:]
        return out_sent

    def comp_pre_res_with_0(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 3 and self.chat_session.math_equ_res_list:
            pre_res = mp.mpf(self.chat_session.math_equ_res_list[-1].replace(',', ''))
            if pre_res < 0:
                output = item_list[0]
            elif pre_res == 0:
                output = item_list[1]
            else:
                output = item_list[2]
            return to_dense_text(output), 0
        return '', 0

    def equal_pre_ress(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 3 and len(self.chat_session.math_equ_res_tuple) == 2:
            x, y = self.chat_session.math_equ_res_tuple
            if x == y:
                output = item_list[0]
            elif self._relative_almosteq(x, y):
                output = item_list[1]
            else:
                output = item_list[2]
            return to_dense_text(output), 0
        return '', 0

    @staticmethod
    def _relative_almosteq(a, b, rel_eps=0.001):
        if a == 0 or b == 0:
            return False
        return abs((a - b) / max(abs(a), abs(b))) <= rel_eps

    def evaluate_mixed_arith_exp(self, mixed_arith_exp):
        self.chat_session.set_keep_topic(Topic('ARITHMETIC', ''))
        self.chat_session.set_categ(self.chat_session.get_topic_categ('ARITHMETIC'))

        try:
            tree = ArithTree.build(mixed_arith_exp)
            out_num = tree.evaluate()
            desc2 = tree.get_output_text()

            res_type, res = get_output_number(out_num)
            if res_type == 0:
                desc = random.choice(SessionFunction.easy_list)
            elif res_type == 2:
                desc = random.choice(["这是个复数，我用了计算器才算出来：", "这题感觉蛮难的，计算结果是个复数："])
            else:
                desc = random.choice(SessionFunction.hard_list)

            if desc != "":
                desc += "_nl_"
            else:
                desc = "这题我会算：_nl_"  # so that we don't have to deal with desc2[0] in ['i', 'e'] case

            if desc2.find('E') > 0 and not self.chat_session.scientific_not_e_explained:
                self.chat_session.scientific_not_e_explained = True
                extra = "_np_（系统信息：该算术表达式被解释为使用了科学记数法。对于字母E或e，系统首先尝试匹配科学计数法，" \
                        "只要所计数字部分的字符串没被空格分隔，且该字符串始于一个整数或小数，然后是字母E或e，最后是一个带" \
                        "正负号的自然数，当然正号可以省略。否则，字母E或e将被理解为自然常数。如果该规则导致算式的解释不合" \
                        "您预期，请尝试在上述字符串的加或减号前添加空格。在结果的展示中，科学计数法的E均为大写，而自然常" \
                        "数e则为小写。）"
            else:
                extra = ""
            return "{}{} = {}{}".format(desc, desc2, res, extra), 0
        except IndexError:
            return "这个算术表达式里的括号好像不匹配。我真算不了呢，抱歉啊！", 0
        except ZeroDivisionError:
            return "晕，不带这么玩的，因为零既不能作为除数，也不能有负数次幂啊。", 0
        except ValueError as ve:
            return str(ve), 0
        except OverflowError:
            return "晕，这运算数字太大（或太小），我确实算不了呢，抱歉啊！", 0

    """
    # Rule 2.2: Unit conversion, huge obj data (unit related)
    """
    @staticmethod
    def _get_temper_unit2_amount(unit1, num1, unit2):
        if unit1 == '开尔文':
            k_deg = num1
        elif unit1 == '摄氏度':
            k_deg = num1 + 273.15
        else:  # 华氏度
            k_deg = (num1 + 459.67) * 5 / 9

        if k_deg >= 0:
            if unit1 == unit2:  # a special case after validation
                return num1
            if unit2 == '开尔文':
                return k_deg
            elif unit2 == '摄氏度':
                return k_deg - 273.15
            else:  # 华氏度
                return k_deg * 9 / 5 - 459.67
        else:
            return None

    # def get_converted_units(self, sentence, in_question):
    #     item_list = re.findall(r'\[(.*?)\]', sentence)
    #     if in_question and len(item_list) == 2:
    #         found, out_quest, para_list = extract_unit_amount_unit_and_replace(in_question)
    #         if found:
    #             # unit_with_amount, unit2 = para_list[0], para_list[1]
    #             # print("unit_with_amount = {}, #2 = {}".format(unit_with_amount, to_dense_text(item_list[0])))
    #             # assert unit_with_amount == to_dense_text(item_list[0]).replace(' ', '')
    #             # assert unit2.lower() == to_dense_text(item_list[1]).lower()
    #             return self._get_converted_units(sentence, out_quest, para_list[0], para_list[1])
    #     return "抱歉，{}还在学习中，无法完成您要求的单位转换呢。".format(self.chat_session.bot_name), 0

    # @staticmethod
    # def _get_converted_units(sentence, in_question, unit_with_amount, unit2):
    #     num1, unit1 = parse_unit_amount_text(unit_with_amount)
    #     if num1 is None or unit1 is None:
    #         return "“{}”是什么意思啊，我看不懂呢。所以我无法进行您要求的单位转换了，抱歉啊！".format(unit_with_amount), 0
    #
    #     u1, u2 = unit1.lower(), unit2.lower()
    #     cat1 = unit_category_dict.get(u1)
    #     cat2 = unit_category_dict.get(u2)
    #     if cat1 is None:  # Not supposed to happen as the pattern search won't allow to come here
    #         return "{}这个单位我还没听说过呢，你给我讲解一下呗。".format(unit1), 0
    #     elif cat2 is None:  # Not supposed to happen as the pattern search won't allow to come here
    #         return "{}这个单位我还没听说过呢，你给我讲解一下呗。".format(unit2), 0
    #
    #     if cat1 != cat2:
    #         return "矮油，我太笨了，不知道怎么将{}单位换算成{}单位呢。".format(cat1, cat2), 0
    #
    #     if u1 == u2:
    #         # Certain temperature cases may not reach here. But we want to verify the temperature is a valid one
    #         return "{}当然就等于{}{}呀。".format(unit_with_amount, num1, unit2), 0
    #
    #     if cat1 == '温度':  # cat1 == cat2 already
    #         new_unit2 = formalize_temper_unit(u2)
    #         if new_unit2 is None:  # Not supposed to be satisfied
    #             return "{}这个单位我还没听说过呢，你给我讲解一下呗。".format(unit2), 0
    #         num2 = SessionFunction._get_temper_unit2_amount(u1, num1, new_unit2)
    #         if num2 is None:
    #             return "{}{}已经低于绝对零度了，所以没有意义啊。".format(num1, unit1), 0
    #     else:
    #         num2 = (num1 / unit_convert_dict.get(u2)) * unit_convert_dict.get(u1)
    #     data_type, res = get_output_number(num2)
    #     eq = "就等于" if num1 == num2 else "等于"
    #
    #     reversed_quest = True if 0 <= in_question.find('_unit1_') < in_question.find('_unit_with_amount_') else False
    #
    #     desc1 = random.choice(["这个换算不难", "这我会换算", "这个换算很简单", "这我知道"])
    #     desc2 = random.choice(["啊", "呀", ""])
    #     if reversed_quest:
    #         return "{}，{}{}{}{}{}。".format(desc1, res, unit2, eq, unit_with_amount, desc2), 0
    #     else:
    #         return "{}，{}{}{}{}{}。".format(desc1, unit_with_amount, eq, res, unit2, desc2), 0

    def get_huge_most_data(self, sentence, in_question):
        ret_text = ''
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if in_question and len(item_list) == 1:
            in_question = self._remove_prefix_for_ner_predictor(in_question)
            hugo_name, from_most = self.ner_predictor.predict_hugo_name(in_question)
            if hugo_name:
                cat_key = '名称'
                con_mat = re.search(r'(面 积|人 口|海 拔|[长宽高深]\s度)(?!\s[最第])', in_question)
                if con_mat:
                    cat_key = in_question[con_mat.start():con_mat.end()].replace(' ', '')
                elif re.search(r'面 积', in_question) and re.search(r'多\s[少大小]', in_question):
                    cat_key = '面积'
                elif re.search(r'人 口', in_question) and re.search(r'多\s少', in_question):
                    cat_key = '人口'
                elif re.search(r'海 拔', in_question) and re.search(r'多\s少', in_question):
                    cat_key = '海拔'

                if cat_key == '名称':
                    cri_mat = re.search(r"多\s[长短高深宽窄大小]", in_question)
                    if cri_mat:
                        cri_key = in_question[cri_mat.start():cri_mat.end()].replace(' ', '')
                        if cri_key in ['多长', '多短']:
                            cat_key = '长度'
                        elif cri_key == '多高':
                            cat_key = '海拔' if re.search(r'[高山]\s峰', in_question) else '高度'
                        elif cri_key == '多深':
                            cat_key = '深度'
                        elif cri_key in ['多宽', '多窄']:
                            cat_key = '宽度'
                        elif cri_key in ['多大', '多小']:
                            if re.search(r'国\s家|城\s市|[高平草]\s原|省|湖|岛', in_question) and \
                                    not re.search(r'人 口', in_question):
                                cat_key = '面积'

                if cat_key == '名称':
                    mea_mat = re.search(r'(多\s少|几)\s(平\s方|人|([毫厘分千]\s)?米|([公英]\s)?[里尺]|丈|(英\s)?寸)',
                                        in_question)
                    if mea_mat:
                        mea_key = in_question[mea_mat.start():mea_mat.end()].replace(' ', '')
                        if re.fullmatch(r'(多少|几)平方', mea_key):
                            cat_key = '面积'
                        elif re.fullmatch(r'(多少|几)人', mea_key):
                            cat_key = '人口'
                        elif re.search(r'[高山]\s峰', in_question):
                            cat_key = '海拔'
                ret_text = self._get_huge_obj_data_from_keys(cat_key, hugo_name, from_most=from_most)

        return ret_text, 0

    def _get_huge_obj_data(self, attr_key, in_question):
        ret_text = ''
        if attr_key and in_question:
            in_question = self._remove_prefix_for_ner_predictor(in_question)
            hugo_name, from_most = self.ner_predictor.predict_hugo_name(in_question)
            if attr_key and hugo_name:
                maybe_person = False
                if attr_key in ['年龄', '体积']:
                    if re.search(r'[年岁]', in_question) or re.search(r'多\s大\s了', in_question):
                        cat_key, maybe_person = '年龄', True
                    elif re.search(r'多\s大', in_question):
                        cat_key, maybe_person = '体积', True
                ret_text = self._get_huge_obj_data_from_keys(attr_key, hugo_name, from_most=from_most,
                                                             maybe_person=maybe_person)

        return ret_text

    def _get_huge_obj_data_from_keys(self, cat_key, hugo_name, from_most, maybe_person=False, from_person=False):
        ret_text, cv_info = self.knowledge_base.get_huge_obj_data_by_key(cat_key, hugo_name, from_most)
        hugo_name = hugo_name.replace('+', '')
        if cv_info:
            cv_list = cv_info.split('==')
            ta3, tpc_txt = cv_list[0], cv_list[1]
            self.chat_session.set_pronoun('ta3', ta3)
            if '=' in tpc_txt:
                self.chat_session.set_keep_topic(
                    Topic("HUGO_QUERY", "{}={}".format(tpc_txt, cat_key)), rounds=1)
                tpc_list = tpc_txt.split('=')
                if cat_key == '名称':
                    categ = "{}叫什么".format(tpc_list[2])
                else:
                    categ = "{}{}是多少".format(tpc_list[2], cat_key)
                self.chat_session.set_categ(categ)
            else:
                self.chat_session.set_categ(tpc_txt)
        else:
            self.chat_session.set_pronoun('ta3', hugo_name)
            if cat_key != '名称':
                self.chat_session.set_categ("{}是多少".format(cat_key))
        if not ret_text and not from_person and \
                (cat_key in ['高度', '身高', '重量', '体重', '年龄'] or (cat_key == '体积' and maybe_person)):
            # handle cases of prediction error
            if self.knowledge_base.is_a_person_name(hugo_name):
                if cat_key in ['高度', '身高']:
                    per_key = '身高'
                elif cat_key in ['重量', '体重']:
                    per_key = '体重'
                else:
                    per_key = '年龄'
                ret_text = self._get_celeb_attr_text(hugo_name, per_key, from_thing=True)
                if not ret_text:
                    ret_text = "咱又不是狗仔队的，干嘛要关注别人的{}呢？这我真不擅长啊。".format(per_key)

        if not ret_text:
            if cat_key == '名称':
                ret_text = random.choice([
                    "{}读书少，不知道{}叫什么呢。".format(self.chat_session.bot_name, hugo_name),
                    "{}见识少，不清楚{}叫什么呢。".format(self.chat_session.bot_name, hugo_name),
                    "扎心了老铁，我不知道{}叫什么呢。".format(hugo_name), ])
            else:
                extra = '' if '的' in hugo_name else '的'
                ret_text = random.choice([
                    "{}读书少，不知道{}{}{}是多少呢。".format(self.chat_session.bot_name, hugo_name, extra, cat_key),
                    "{}见识少，不清楚{}{}{}是多少呢。".format(self.chat_session.bot_name, hugo_name, extra, cat_key),
                    "扎心了老铁，我不知道{}{}{}是多少呢。".format(hugo_name, extra, cat_key), ])
        return ret_text

    def _get_area_attr(self, attr_key, in_question):
        ret_text = ''
        if attr_key and in_question:
            in_question = self._remove_prefix_for_ner_predictor(in_question)
            hugo_name, from_most = self.ner_predictor.predict_hugo_name(in_question)
            if attr_key and hugo_name:
                # prediction errors: '长度', '海拔', '重量', '表面积', '体积'
                if attr_key in ['面积', '人口', '长度', '海拔', '重量', '表面积', '体积']:
                    ret_text = self._get_huge_obj_data_from_keys(attr_key, hugo_name, from_most=from_most)
                else:
                    ret_text = self._get_area_attr_from_keys(attr_key, hugo_name)

        return ret_text

    def _get_area_attr_from_keys(self, cat_key, area_name):
        ret_text, area_key = self.knowledge_base.get_area_attr_by_key(cat_key, area_name)
        if area_key:
            self.chat_session.set_pronoun('ta3', area_key)
        else:
            self.chat_session.set_pronoun('ta3', area_name)
        if cat_key in ['所在国', '所在省', '所在州']:
            self.chat_session.set_categ("{}是哪个".format(cat_key))
        elif cat_key == '行政区':
            self.chat_session.set_categ("下级行政区有多少")
        # if the prediction is correct, no other cat_key possibilities
        if not ret_text and cat_key in ['户籍', '所在国', '所在省', '所在州']:
            # handle cases of prediction error
            if self.knowledge_base.is_a_person_name(area_name):
                ret_text = self._get_celeb_attr_text(area_name, '户籍', from_thing=True)
                if not ret_text:
                    ret_text = "咱又不是狗仔队的，干嘛要关注别人是哪里的呢？这我真不擅长啊。"

        if not ret_text:
            if cat_key in ['所在国', '所在省', '所在州']:
                ret_text = random.choice([
                    "{}读书少，不清楚{}的{}信息呢。".format(self.chat_session.bot_name, area_name, cat_key),
                    "{}见识少，不了解{}的{}信息呢。".format(self.chat_session.bot_name, area_name, cat_key), ])
            elif cat_key == '行政区':
                ret_text =  random.choice([
                    "{}读书少，不知道{}相关的行政区划呢。".format(self.chat_session.bot_name, area_name),
                    "{}见识少，不熟悉{}相关的行政区划呢。".format(self.chat_session.bot_name, area_name), ])
        return ret_text

    # def extract_prev_unit_with_amount_and_convert(self, sentence, in_question):
    #     last_answer = self.chat_session.last_answer
    #     if last_answer:
    #         last_answer = re.sub(r'_nl_|_np_|_nr_', ' ', last_answer)
    #         ua_text = extract_unit_amount_or_unit_text(last_answer, ex_type='UA')
    #         un_text = extract_unit_amount_or_unit_text(in_question, ex_type='UN')
    #         if ua_text and un_text:
    #             return self._get_converted_units(sentence, in_question, ua_text, un_text)
    #
    #     ret_text = None
    #     item_list = re.findall(r'\[(.*?)\]', sentence)
    #     if len(item_list) == 1 and item_list[0]:
    #         ret_text = item_list[0].replace(' ', '').strip()
    #
    #     if not ret_text:
    #         ret_text = "晕，我被问糊涂了，无法进行您要求的单位转换呢，抱歉啊！"
    #     return ret_text, 0

    """
    # Rule 3.1: Stories, Jokes and Duanzi
    """
    @staticmethod
    def _if_story_till_end(quest):
        neg_mat = r'([别甭]|不\s[要必用])'
        true_mat = r'(({0}\s)?结\s[尾束]|{0})'.format(r'(一\s[下次]|全\s[都部]|整\s个|[全都])')
        true_cat = re.compile(r'\b({}|{}(\s再)?\s分)\b'.format(true_mat, neg_mat))
        false_cat = re.compile(r'\b{}\s{}\b'.format(neg_mat, true_mat))
        opt_cat = re.compile(r'\b(接\s着|继\s续|后\s来|然\s后|(再|重\s新)\s[来讲说]|[来讲说](\s一)?\s个)\b')

        if re.search(false_cat, quest) or (re.search(opt_cat, quest) and not re.search(true_cat, quest)):
            return False
        return True

    def get_story_any(self, func_para1=None):
        self.chat_session.set_keep_topic(Topic('STORY', ''))
        self.chat_session.set_context_ref(self.chat_session.get_topic_categ('STORY'))

        # Always from start of a story
        self.chat_session.reset_story_talker()

        till_end = func_para1 and func_para1.lower() == 'true'
        story, is_start, is_end = self.chat_session.get_next_story_parts(self.knowledge_base, till_end)
        return story, 0

    def get_story_cat(self, sentence, func_para1=None):
        self.chat_session.set_keep_topic(Topic('STORY', ''))
        self.chat_session.set_context_ref(self.chat_session.get_topic_categ('STORY'))

        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            story_cat = item_list[0].replace(' ', '').strip()

            # Always from start of a story
            self.chat_session.reset_story_talker()

            till_end = func_para1 and func_para1.lower() == 'true'
            story, is_start, is_end = self.chat_session.get_next_story_parts(self.knowledge_base,
                                                                             till_end, story_cat)
            if story != '':
                return story, 0
            else:
                return "只是这个类别的故事我也不会呢，真是抱歉啊。", 0

        return "有个坏蛋把{0}的故事书悄悄偷走了。{0}找不到故事了，呜呜。".format(self.chat_session.bot_name), 0

    def continue_story(self, sentence, in_question):
        last_ended_desc, _ = self.chat_session.get_last_story_ended()

        self.chat_session.set_keep_topic(Topic('STORY', ''))
        self.chat_session.set_context_ref(self.chat_session.get_topic_categ('STORY'))

        till_end = SessionFunction._if_story_till_end(in_question)
        print("story till_end = {}".format(till_end))
        story, is_start, is_end = self.chat_session.get_next_story_parts(self.knowledge_base, till_end)
        ret_text = last_ended_desc + ' _nr_' + story

        return ret_text, 0

    def get_joke_any(self):
        self.chat_session.set_keep_topic(Topic('JOKE', ''))
        self.chat_session.set_context_ref(self.chat_session.get_topic_categ('JOKE'))

        jokes = self.knowledge_base.jokes
        content = random.choice(jokes)
        return content, 0

    def get_duanzi_any(self):
        self.chat_session.set_keep_topic(Topic('DUANZI', ''))
        self.chat_session.set_context_ref(self.chat_session.get_topic_categ('DUANZI'))

        duanzi = self.knowledge_base.duanzi
        content = random.choice(duanzi)
        return content, 0

    """
    # Rule 3.2: Poems and Ci Poems
    """
    def get_poem_any(self):
        poem_id = random.choice(self.knowledge_base.poem_dynasties['唐代'])
        content = self.knowledge_base.poems[poem_id].content

        self.chat_session.set_keep_topic(Topic('POEM', "PHID={}".format(poem_id)))
        self.chat_session.set_context_ref(self.chat_session.get_topic_categ('POEM'))
        return content, 0

    def get_poem_by_writer(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            poem_writers = self.knowledge_base.poem_writers
            writer = item_list[0].replace(' ', '').strip()
            if writer and writer in poem_writers:
                poem_id = random.choice(poem_writers[writer])
                content = self.knowledge_base.poems[poem_id].content

                self.chat_session.set_keep_topic(Topic('POEM', "PHID={}".format(poem_id)))
                self.chat_session.set_context_ref(self.chat_session.get_topic_categ('POEM'))
                return content, 0
        return self.get_poem_any()
    
    def get_poem_by_title(self, sentence):
        ret_txt = ''
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            poem_title = item_list[0].replace(' ', '').strip()
            poem_id, poem_cat = self.knowledge_base.get_poem_id_by_title(poem_title)
            if poem_id <= 0:  # for example, 床前明月光
                poem_id, poem_cat = self.knowledge_base.get_poem_id_by_first_line(poem_title)
            if poem_id > 0:
                if poem_cat == 'PHID':
                    ret_txt = self.knowledge_base.poems[poem_id].content
                elif poem_cat == 'PNID':
                    ret_txt = self.knowledge_base.poems_no_exp[poem_id].content

                self.chat_session.set_keep_topic(Topic('POEM', "{}={}".format(poem_cat, poem_id)))
                self.chat_session.set_context_ref(self.chat_session.get_topic_categ('POEM'))

        if not ret_txt:
            ret_txt = "晕，这首诗我想不起来了{捂脸}，真是抱歉啊！"
        return ret_txt, 0

    def get_prev_next_poem_line(self, sentence, in_question):
        ret_text, is_prev = '', False
        if in_question:
            poem_line = self.ner_predictor.predict_poem_line(in_question)
            if poem_line:
                pre_nxt_cat = re.compile(r'歇\s后\s语|[上前下后]\s([头面]\s)?([一半]\s)?句?|接\s下\s[来去]')
                ind_list = [(m.start(0), m.end(0)) for m in re.finditer(pre_nxt_cat, in_question)]
                if len(ind_list) >= 1:
                    ss, ee = ind_list[-1]
                    mat_text = in_question[ss:ee]
                    if mat_text[0] in '上前':
                        is_prev = True
                        ret_text = self._get_prev_poem_line_by_base(poem_line, clear_list=True)
                    else:
                        is_prev = False
                        ret_text = self._get_next_poem_line_by_base(poem_line, clear_list=True)
                        if not ret_text:
                            print("*** trying poem_line from preload_pairs: #{}#".format(poem_line))
                            ret_text = self.knowledge_base.get_preload_pair_value(poem_line, self.chat_session)
                            if not ret_text:
                                # Add "（俗语）" to preload_pairs so that this pair is only used to answer this type
                                # of question, not for when users present the question sentence directly
                                ret_text = self.knowledge_base.get_preload_pair_value(poem_line + "（俗语）",
                                                                                      self.chat_session)
                            if ret_text and '_func_' not in ret_text and ret_text.strip()[-1] not in '。？！}':
                                ret_text += '。'
                            if ret_text and mat_text.replace(' ', '').strip() == '歇后语':
                                # if '_func_set_pending_ans_para0_flw_para1_explain' not in ret_text:
                                #     exp_txt = "矮油，歇后语我是背了不少，但解释我真的不在行呢。"
                                #     self.chat_session.set_pending_ans('explain', exp_txt)
                                #     self.chat_session.set_pending_ans('why', exp_txt)
                                if not re.search(r'[。？！]', poem_line):
                                    ret_text = "{}，{}".format(poem_line, ret_text)

        if not ret_text:
            extra = random.choice(self.get_poem_alter_list)
            ret_text = "上一句我记不得了{捂脸}。" + extra if is_prev else "下一句我记不得了{捂脸}。" + extra

        return ret_text, 0

    def get_poem_from_line(self, sentence, in_question):
        poem = None
        topic_value = ''
        if in_question:
            poem_line = self.ner_predictor.predict_poem_line(in_question)
            poem_id, dict_type = self.knowledge_base.get_poem_id_from_line(poem_line)
            if poem_id > 0:
                if dict_type == 'PHID':
                    poem = self.knowledge_base.poems[poem_id]
                    topic_value = "PHID={}".format(poem_id)
                elif dict_type == 'PNID':
                    poem = self.knowledge_base.poems_no_exp[poem_id]
                    topic_value = "PNID={}".format(poem_id)

        if poem and topic_value:
            p_writer, dynasty = self.knowledge_base.parse_poem_writer(poem.writer)
            pmw = "{}{}".format(dynasty, p_writer) if dynasty else poem.writer
            ret_text = random.choice([
                "是{}的《{}》吧，对吗？".format(pmw, poem.title),
                "这诗我背过的，是{}的《{}》。".format(pmw, poem.title),
                "我知道呀，是{}的《{}》。".format(pmw, poem.title),
            ])

            self.chat_session.set_keep_topic(Topic('POEM', topic_value))
            self.chat_session.set_context_ref(self.chat_session.get_topic_categ('POEM'))
        else:
            ret_text = "这诗歌我不知道呀。" + random.choice(self.get_poem_alter_list)

        return ret_text, 0

    def _set_context_poem_pair_by_base_line(self, base_line, dense_sent):
        old_id, new_id = self._set_context_for_poem_line(base_line)
        if new_id != old_id:  # at least one ID represents a poem
            self.chat_session.poem_line_list = []
        if new_id > 0:
            # Add the poem line related to the question in the pair
            if base_line not in self.chat_session.poem_line_list:
                self.chat_session.poem_line_list.append(base_line)

            # Add the poem line related to the answer in the pair
            list_len = len(self.knowledge_base.poem_lines_list)
            idx = self.knowledge_base.poem_lines_dict.get(base_line, -1)
            if 0 < idx < list_len - 1:
                next_line = self.knowledge_base.poem_lines_list[idx + 1]
                if next_line.startswith('PHID') or next_line.startswith('PNID'):
                    pass
                elif next_line in dense_sent:
                    self.chat_session.poem_line_list.append(next_line)
            elif idx <= 0:
                half_line = self._get_half_line_from_poem_line(base_line, 2)
                if half_line:
                    idx = self.knowledge_base.poem_lines_dict.get(half_line, -1)
                    if 0 < idx < list_len - 2:
                        n1_line = self.knowledge_base.poem_lines_list[idx + 1]
                        n2_line = self.knowledge_base.poem_lines_list[idx + 2]
                        if n1_line.startswith('PHID') or n1_line.startswith('PNID'):
                            pass
                        elif n2_line.startswith('PHID') or n2_line.startswith('PNID'):
                            pass
                        elif n1_line in dense_sent and n2_line in dense_sent:
                            self.chat_session.poem_line_list.append("{}，{}".format(n1_line, n2_line))

    def set_context_poem_pair(self, sentence, in_question):
        base_line = in_question.replace(' ', '').strip()
        if in_question[-1] in ['，', '。', '？', '！']:
            base_line = base_line[:-1]
        self._set_context_poem_pair_by_base_line(base_line, sentence.replace(' ', '').strip())
        return '', 0

    def set_context_poem_pair_by_line(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            base_line = item_list[0].replace(' ', '').strip()
            if base_line:
                self._set_context_poem_pair_by_base_line(base_line, sentence.replace(' ', '').strip())
        return '', 0

    def _get_poem_line_by_context(self, case_text):
        print("poem_line_list = {}".format(self.chat_session.poem_line_list))
        ret_text = ''
        if case_text in ['上句', '下句', '首句', '第一句', '末句', '最后句']:
            if self.chat_session.poem_line_list:
                if case_text == '上句':
                    base_line = self.chat_session.poem_line_list[0]
                    ret_text = self._get_prev_poem_line_by_base(base_line, from_context=True)
                    if not ret_text:
                        ret_text = "上一句我记不得了{捂脸}。" + random.choice(self.get_poem_alter_list)
                elif case_text == '下句':
                    base_line = self.chat_session.poem_line_list[-1]
                    ret_text = self._get_next_poem_line_by_base(base_line, from_context=True)
                    if not ret_text:
                        ret_text = "下一句我记不得了{捂脸}。" + random.choice(self.get_poem_alter_list)
                elif case_text in ['首句', '第一句', '末句', '最后句']:
                    if case_text in ['首句', '第一句']:
                        ret_txt1 = self._get_first_poem_line()
                    else:
                        ret_txt1 = self._get_last_poem_line()
                    if ret_txt1:
                        if ret_txt1.endswith('。'):
                            ret_text = ret_txt1
                        else:
                            ret_list = [
                                "{}。".format(ret_txt1),
                                "我知道，{}。".format(ret_txt1),
                                "我背过的，{}。".format(ret_txt1),
                            ]
                            ret_text = random.choice(ret_list)
                    elif case_text in ['首句', '第一句']:
                        ret_text = "是问第一句吗？我不知道呀，哈哈哈。"
                    else:
                        ret_text = "是问最后一句吗？我不记得了，哈哈哈。"
            else:
                ret_text = "你是要问哪一句呀，{}不明白呢。".format(self.chat_session.bot_name)

        return ret_text, 0

    def _get_poem_by_context(self):
        ret_text = ''
        last_topic = self.chat_session.last_topic
        if last_topic and last_topic.title == 'POEM' and last_topic.value:
            self.chat_session.keep_topic = 2
            self.chat_session.keep_context = True
            self.chat_session.this_context_kept = True

            poem_id = int(last_topic.value[5:])
            if last_topic.value[:4] == 'PHID':
                poem = self.knowledge_base.poems[poem_id]
                ret_text = poem.content
            elif last_topic.value[:4] == 'PNID':
                poem = self.knowledge_base.poems_no_exp[poem_id]
                ret_text = poem.content

        if not ret_text:
            ret_text = "只是，这首诗我记不得了呀，实在抱歉啊。"
            _, need_retry = self.retry_pta3_ctxt_ref()
            if need_retry == 2:
                return ret_text, 2

        return ret_text, 0

    @staticmethod
    def _get_half_line_from_poem_line(full_line, first_or_second):
        half_line = ''
        line_len = len(full_line)
        if 10 <= line_len <= 15:
            if re.search(r'[，？！]', full_line):
                parts = re.sub('[？！]', '，', full_line).split('，')
                if parts and len(parts) == 2 and len(parts[0]) == len(parts[1]):
                    half_line = parts[0] if first_or_second == 1 else parts[1]
            elif line_len == 10:
                half_line = full_line[:5] if first_or_second == 1 else full_line[5:]
            elif line_len == 14:
                half_line = full_line[:7] if first_or_second == 1 else full_line[7:]
        return half_line

    def _get_prev_poem_line_by_base(self, base_line, clear_list=False, from_context=False):
        ret_text = ''
        idx = self.knowledge_base.poem_lines_dict.get(base_line, -1)
        is_long_line = False
        if idx <= 0:
            half_line = self._get_half_line_from_poem_line(base_line, 1)
            if half_line:
                idx = self.knowledge_base.poem_lines_dict.get(half_line, -1)
                is_long_line = True
        if idx > 0:
            old_id, new_id = self._set_context_for_poem_line(base_line)
            if new_id > 0 and (new_id != old_id or clear_list):
                self.chat_session.poem_line_list = []
            if base_line not in self.chat_session.poem_line_list:
                self.chat_session.poem_line_list.insert(0, base_line)

            prev_line = self.knowledge_base.poem_lines_list[idx - 1]
            if prev_line.startswith('PHID') or prev_line.startswith('PNID'):
                ret_text = "这首诗的第一句都已经讲了，没有上句了呀。"
            elif is_long_line:
                p_prev_line = self.knowledge_base.poem_lines_list[idx - 2]
                if p_prev_line.startswith('PHID') or p_prev_line.startswith('PNID'):  # not supposed to happen
                    ret_text = "这首诗的第一句都已经讲了，没有上句了呀。"
                else:
                    com_line = "{}，{}".format(p_prev_line, prev_line)
                    if from_context:
                        ret_text = "“{}”的上句是{}。".format(base_line, com_line)
                    else:
                        ret_list = [
                            "{}。".format(com_line),
                            "我知道，{}。".format(com_line),
                            "我背过的，{}。".format(com_line),
                        ]
                        ret_text = random.choice(ret_list)
                    self.chat_session.poem_line_list.insert(0, com_line)
            else:
                if from_context:
                    ret_text = "“{}”的上句是{}。".format(base_line, prev_line)
                else:
                    ret_list = [
                        "{}。".format(prev_line),
                        "我知道，{}。".format(prev_line),
                        "我背过的，{}。".format(prev_line),
                    ]
                    ret_text = random.choice(ret_list)
                self.chat_session.poem_line_list.insert(0, prev_line)

        return ret_text

    def _get_next_poem_line_by_base(self, base_line, clear_list=False, from_context=False):
        ret_text = ''
        idx = self.knowledge_base.poem_lines_dict.get(base_line, -1)
        is_long_line = False
        if idx <= 0:
            half_line = self._get_half_line_from_poem_line(base_line, 2)
            if half_line:
                idx = self.knowledge_base.poem_lines_dict.get(half_line, -1)
                is_long_line = True
        list_len = len(self.knowledge_base.poem_lines_list)
        if idx > 0:
            old_id, new_id = self._set_context_for_poem_line(base_line)
            if new_id > 0 and (new_id != old_id or clear_list):
                self.chat_session.poem_line_list = []
            if base_line not in self.chat_session.poem_line_list:
                self.chat_session.poem_line_list.append(base_line)

            if idx < list_len - 1:
                next_line = self.knowledge_base.poem_lines_list[idx + 1]
                if next_line.startswith('PHID') or next_line.startswith('PNID'):
                    ret_text = "这首诗的最后一句都已经讲了，没有下句了呀。"
                elif is_long_line:
                    n_next_line = self.knowledge_base.poem_lines_list[idx + 2]
                    if n_next_line.startswith('PHID') or n_next_line.startswith('PNID'):  # not supposed to happen
                        ret_text = "这首诗的最后一句都已经讲了，没有下句了呀。"
                    else:
                        com_line = "{}，{}".format(next_line, n_next_line)
                        if from_context:
                            ret_text = "“{}”的下句是{}。".format(base_line, com_line)
                        else:
                            ret_list = [
                                "{}。".format(com_line),
                                "我知道，{}。".format(com_line),
                                "我背过的，{}。".format(com_line),
                            ]
                            ret_text = random.choice(ret_list)
                        self.chat_session.poem_line_list.append(com_line)
                else:
                    if from_context:
                        ret_text = "“{}”的下句是{}。".format(base_line, next_line)
                    else:
                        ret_list = [
                            "{}。".format(next_line),
                            "我记得，{}。".format(next_line),
                            "我背过的，{}。".format(next_line),
                        ]
                        ret_text = random.choice(ret_list)
                    self.chat_session.poem_line_list.append(next_line)
            elif idx == list_len - 1:
                ret_text = "这首诗的最后一句都已经讲了，没有下句了呀。"

        return ret_text

    def _get_first_poem_line(self):
        ret_line = ''
        if self.chat_session.poem_line_list:
            self._continue_topic_for_same_poem()  # maintain poem context
            base_line = self.chat_session.poem_line_list[0]
            idx = self.knowledge_base.poem_lines_dict.get(base_line, -1)
            is_long_line = False
            if idx <= 0:
                half_line = self._get_half_line_from_poem_line(base_line, 1)
                if half_line:
                    idx = self.knowledge_base.poem_lines_dict.get(half_line, -1)
                    is_long_line = True
            while idx > 0:
                idx -= 1
                prev_line = str(self.knowledge_base.poem_lines_list[idx])
                if prev_line.startswith('PHID') or prev_line.startswith('PNID'):
                    n1_line = str(self.knowledge_base.poem_lines_list[idx + 1])
                    if is_long_line:
                        n2_line = str(self.knowledge_base.poem_lines_list[idx + 2])
                        ret_line = "{}，{}".format(n1_line, n2_line)
                    else:
                        ret_line = n1_line
                    if ret_line not in self.chat_session.poem_line_list:
                        self.chat_session.poem_line_list.insert(0, ret_line)
                    else:
                        ret_line = "刚刚已经提到过了，{}。".format(ret_line)
                    break

        return ret_line

    def _get_last_poem_line(self):
        ret_line = ''
        if self.chat_session.poem_line_list:
            self._continue_topic_for_same_poem()  # maintain poem context
            list_len = len(self.knowledge_base.poem_lines_list)
            base_line = self.chat_session.poem_line_list[-1]
            idx = self.knowledge_base.poem_lines_dict.get(base_line, -1)
            is_long_line = False
            if idx <= 0:
                half_line = self._get_half_line_from_poem_line(base_line, 2)
                if half_line:
                    idx = self.knowledge_base.poem_lines_dict.get(half_line, -1)
                    is_long_line = True
            while 0 < idx < list_len - 1:
                idx += 1
                next_line = str(self.knowledge_base.poem_lines_list[idx])

                if next_line.startswith('PHID') or next_line.startswith('PNID'):
                    last_line = str(self.knowledge_base.poem_lines_list[idx - 1])
                    if is_long_line:
                        p_last_line = str(self.knowledge_base.poem_lines_list[idx - 2])
                        ret_line = "{}，{}".format(p_last_line, last_line)
                    else:
                        ret_line = last_line
                    if ret_line not in self.chat_session.poem_line_list:
                        self.chat_session.poem_line_list.append(ret_line)
                    else:
                        ret_line = "刚刚已经提到过了，{}。".format(ret_line)
                    break
                elif idx == list_len - 1:
                    last_line = str(self.knowledge_base.poem_lines_list[idx])
                    if is_long_line:
                        p_last_line = str(self.knowledge_base.poem_lines_list[idx - 1])
                        ret_line = "{}，{}".format(p_last_line, last_line)
                    else:
                        ret_line = last_line
                    if ret_line not in self.chat_session.poem_line_list:
                        self.chat_session.poem_line_list.append(ret_line)
                    else:
                        ret_line = "刚刚已经提到过了，{}。".format(ret_line)
                    break

        return ret_line

    def _set_context_for_poem_line(self, base_line):
        old_id = self._get_old_poem_id_from_session()  # get old_id before being overwritten

        poem_id, dict_type = self.knowledge_base.get_poem_id_from_line(base_line)
        if poem_id > 0 and dict_type:  # no matter changed or not, the context needs to be set
            topic_value = "{}={}".format(dict_type, poem_id)
            self.chat_session.set_keep_topic(Topic('POEM', topic_value))
            self.chat_session.set_context_ref(self.chat_session.get_topic_categ('POEM'))
            self.chat_session.set_pronoun('ta3', base_line)

        return old_id, poem_id

    def _get_old_poem_id_from_session(self):
        last_topic = self.chat_session.last_topic
        if last_topic and last_topic.title == 'POEM' and last_topic.value:
            return int(last_topic.value[5:])
        return 0

    def _continue_topic_for_same_poem(self):
        last_topic = self.chat_session.last_topic
        if last_topic and last_topic.title == 'POEM' and last_topic.value:
            self.chat_session.set_keep_topic(Topic('POEM', last_topic.value))
            self.chat_session.set_context_ref(self.chat_session.get_topic_categ('POEM'))

    def _get_poem_info(self, input_type):
        last_topic = self.chat_session.last_topic
        if last_topic and last_topic.title in ['POEM', 'WRITE_POEM', 'WRITE_CI_POEM']:
            self.chat_session.keep_topic = 2
            self.chat_session.keep_context = True
            self.chat_session.this_context_kept = True

            eng_type = None
            if input_type in ['标题', '题目', '名字']:
                eng_type = 'title'
            elif input_type == '作者':
                eng_type = 'writer'
            elif input_type in ['大意', '意思', '解释', '说明']:
                eng_type = 'explanation'

            ret_text, need_retry = self._get_poem_info_details(last_topic, eng_type)
            if ret_text:
                return ret_text, need_retry

        need_retry = 0
        if self.chat_session.has_context():
            self.chat_session.keep_context = True
            need_retry = 1

        return "晕，我们刚刚是在谈论诗歌吗？我怎么一点印象都没有了呢{捂脸}", need_retry

    def _get_poem_info_details(self, last_topic, info_type):
        if last_topic.title == 'POEM' and last_topic.value:
            poem_id = int(last_topic.value[5:])
            if last_topic.value[:4] == 'PHID':
                poem = self.knowledge_base.poems[poem_id]
            elif last_topic.value[:4] == 'PNID':
                poem = self.knowledge_base.poems_no_exp[poem_id]
            else:
                return "主人把程序搞错了，让我真尴尬。抱歉啊。", 0

            if info_type == 'title':
                if poem.title == '无题' and poem.writer == '网络':
                    return "这是流传于网络的组诗，没有名字吧。", 0
                return "名字是《{}》。".format(poem.title), 0
            elif info_type == 'writer':
                p_writer, dynasty = self.knowledge_base.parse_poem_writer(poem.writer)
                if dynasty:
                    if p_writer == '无名氏':
                        w_list = ['作者是{}无名氏。'.format(dynasty),
                                  '作者是{}诗人，具体是谁无法考证。'.format(dynasty)]
                    else:
                        ex_text = '' if len(dynasty) > 2 else '诗人'
                        w_list = ['是{}{}{}呀。'.format(dynasty, ex_text, p_writer),
                                  '作者是{}{}{}吧。'.format(dynasty, ex_text, p_writer)]
                elif poem.title == '无题' and poem.writer == '网络':
                    w_list = ['出处不详，应该是不具名的网友写的吧。',
                              '这是流传于网路的组诗，作者是谁真不知道呢。']
                else:
                    w_list = ['这我知道，作者是{}。'.format(poem.writer),
                              '作者是{}。'.format(poem.writer)]
                return random.choice(w_list), 0
            elif info_type == 'explanation':
                if last_topic.value[:4] == 'PNID':
                    return "晕，背背倒还行，但解释就不会了，不要笑话我啦。", 0
                else:
                    expl = poem.explanation
                    return expl, 0
        elif last_topic.title in ['WRITE_POEM', 'WRITE_CI_POEM']:
            if info_type == 'title':
                return "随便写写的，没有给标题啦。", 0
            elif info_type == 'writer':
                return "当代著名诗人：{}。".format(self.chat_session.bot_name), 0
            elif info_type == 'explanation':
                return "只是...... 我写得不好。自己都看不懂了，不要笑话我啦。", 0

        return '', 0

    # def get_poem_mugua_info(self, sentence):
    #     self.chat_session.set_context_ref('木瓜辞')
    #     item_list = re.findall(r'\[(.*?)\]', sentence)
    #     if len(item_list) == 1 and item_list[0]:
    #         info_cat = item_list[0].replace(' ', '').strip()
    #         if info_cat in ['大意', '意思', '解释', '说明']:
    #             return self.mugua_shi_exp, 0
    #     return self.mugua_shi, 0

    def compose_poem_any(self):
        if self.poem_writer is None:
            return "麻烦您转告我主人，他忘了打开我身上的唐诗模块了。", 0

        # Add an extra LS5 to increase its chance
        poem_type = random.choice(['[JJ5]', '[JJ7]', '[LS5]', '[LS5]', '[LS7]'])
        poem = self.poem_writer.write_valid_poem(poem_type)
        if poem is None or poem == '':
            poem = "{}状态今天太差，写不出来了。羞死了。".format(self.chat_session.bot_name)
        else:
            self.chat_session.set_keep_topic(Topic('WRITE_POEM', ''))
            self.chat_session.set_context_ref(self.chat_session.get_topic_categ('WRITE_POEM'))
        return poem, 0

    def compose_poem_type(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            type_text = item_list[0].replace(' ', '').strip()
            text_id_dict = {
                '绝句': 'jj57',
                '律诗': 'ls57',
                '五绝': 'jj5',
                '七绝': 'jj7',
                '五律': 'ls5',
                '七律': 'ls7',
            }
            if type_text not in text_id_dict:
                return self.compose_poem_any()
            else:
                return self._compose_poem(text_id_dict[type_text])

    def _compose_poem(self, poem_type):
        if self.poem_writer is None:
            return "麻烦您转告我主人，他忘了打开我身上的唐诗模块了。", 0
        elif poem_type is None or poem_type == '':
            return self.compose_poem_any()

        poem = self.poem_writer.write_poem_by_type(poem_type)
        if poem is None or poem == '':
            poem = "{}状态今天太差，写不出来了。羞死了。".format(self.chat_session.bot_name)
        else:
            self.chat_session.set_keep_topic(Topic('WRITE_POEM', poem_type))
            self.chat_session.set_context_ref(self.chat_session.get_topic_categ('WRITE_POEM'))
        return poem, 0

    def compose_ci_poem_any(self):
        if self.ci_poem_writer is None:
            return "麻烦您转告我主人，他忘了打开我身上的宋词模块了。", 0

        cipai = random.choice(self.ci_poem_writer.best_list)
        poem = self.ci_poem_writer.write_poem_by_cipai(cipai)

        self.chat_session.set_keep_topic(Topic('WRITE_CI_POEM', ''))
        self.chat_session.set_context_ref(self.chat_session.get_topic_categ('WRITE_CI_POEM'))
        if self.chat_session.context_prefix['state'] == self.chat_session.cipai_context_prefix:
            self.chat_session.context_prefix['count'] += 1
        return poem, 0

    def compose_ci_poem(self, cipai):
        if self.poem_writer is None:
            return "麻烦您转告我主人，他忘了打开我身上的宋词模块了。", 0
        elif cipai is None or cipai == '' or cipai == '_cipai_':
            return self.compose_ci_poem_any()

        poem = self.ci_poem_writer.write_poem_by_cipai(cipai)
        if poem is None or poem == '':
            poem = "{}状态今天太差，写不出来了。羞死了。".format(self.chat_session.bot_name)
        else:
            self.chat_session.set_keep_topic(Topic('WRITE_CI_POEM', cipai))
            self.chat_session.set_context_ref(self.chat_session.get_topic_categ('WRITE_CI_POEM'))
            if self.chat_session.context_prefix['state'] == self.chat_session.cipai_context_prefix:
                self.chat_session.context_prefix['count'] += 1
        return poem, 0

    def get_lyric_by_title(self, sentence):
        ret_txt = ''
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            lyric_title = item_list[0].replace(' ', '').strip()
            lyric_id = self.knowledge_base.get_lyric_id_by_title(lyric_title)
            if lyric_id > 0:
                ret_txt = self.knowledge_base.lyrics[lyric_id].content

                self.chat_session.set_keep_topic(Topic('LYRIC', "{}={}".format(lyric_id, lyric_title)))
                self.chat_session.set_context_ref(self.chat_session.get_topic_categ('LYRIC'))

        if not ret_txt:
            ret_txt = "晕，这首歌的歌词我想不起来了{捂脸}，真是抱歉啊！"
        return ret_txt, 0

    def _get_song_info(self, info_type):
        last_topic = self.chat_session.last_topic
        if last_topic and last_topic.title == 'LYRIC' and last_topic.value:
            self.chat_session.keep_topic = 2
            self.chat_session.keep_context = True
            self.chat_session.this_context_kept = True

            sp_idx = last_topic.value.find('=')
            lyric_id = int(last_topic.value[:sp_idx])
            lyric = self.knowledge_base.lyrics[lyric_id]

            if info_type in ['标题', '题目', '名字']:
                return "歌曲的名字是《{}》。".format(lyric.title), 0
            elif info_type == '作者':
                if lyric.writer and lyric.composer:
                    if lyric.writer == lyric.composer:
                        return "这首歌的词曲作者均为{}。".format(lyric.writer), 0
                    elif lyric.composer.endswith('民歌'):
                        return "这首歌是{}，它的词作者是{}。".format(lyric.composer, lyric.writer), 0
                    else:
                        return "这首歌的词作者是{}，曲作者是{}。".format(lyric.writer, lyric.composer), 0
                elif lyric.writer:
                    return "这首歌的词作者是{}。".format(lyric.writer), 0
                elif lyric.composer:
                    if lyric.composer.endswith('民歌'):
                        return "这首歌是{}。".format(lyric.composer), 0
                    else:
                        return "这首歌的曲作者是{}。".format(lyric.composer), 0
                else:
                    return "{}孤陋寡闻，不知道这首歌的词曲作者是谁呢。".format(self.chat_session.bot_name), 0
            elif info_type == '演唱者':
                if lyric.singer:
                    return "这首歌的原唱是{}。".format(lyric.singer), 0
                else:
                    return "{}孤陋寡闻，不知道这首歌的原唱是谁呢。".format(self.chat_session.bot_name), 0
            elif info_type == '歌词':
                return "歌词我记得，是这样的： _nr_{}".format(lyric.content), 0

        need_retry = 0
        if self.chat_session.get_decision_pronoun_ref_text():
            self.chat_session.keep_pronoun_ref = True
            need_retry = 2
        elif self.chat_session.context_ref_text:
            self.chat_session.keep_context = True
            need_retry = 1

        return "矮油，{}孤陋寡闻，对歌曲更是了解不多呢{{捂脸}}".format(self.chat_session.bot_name), need_retry

    def get_pmsg_context(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 2 and item_list[0] and item_list[1]:
            info_cate = item_list[0].replace(' ', '').strip()
            info_type = item_list[1].replace(' ', '').strip()
            if info_cate == '诗歌':
                if info_type == '全文':
                    return self._get_poem_by_context() + (2,)
                elif info_type in ['标题', '题目', '名字', '作者', '大意', '意思', '解释', '说明']:
                    return self._get_poem_info(info_type) + (2,)
                elif info_type in ['上句', '下句', '首句', '第一句', '末句', '最后句']:
                    return self._get_poem_line_by_context(info_type) + (3,)
            elif info_cate == '歌曲':
                return self._get_song_info(info_type) + (2,)
            elif info_cate in ['木瓜诗', '木瓜辞']:
                self.chat_session.set_context_ref('木瓜辞')
                if info_type in ['大意', '意思', '解释', '说明']:
                    return self.mugua_shi_exp, 0, 2
                return self.mugua_shi, 0, 2
        return "抱歉，我还在学习中，暂时无法正确回复这个诗词或歌曲类的问题呢！", 0, 2

    """
    Rule 4: Start/Keep topic, propose a topic and continue last topic
    """
    def keep_topic(self, sentence, proposed=False):
        # This function is executed when we only have an idea of a broad topic, without
        # its details, therefore, topic.value is set to empty.
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            text_topic = item_list[0].replace(' ', '')

            tp_dict = {
                '故事': 'story',
                '笑话': 'joke',
                '段子': 'duanzi',
                '背诗': 'poem',
                '背词': 'ci_poem',
                '写诗': 'write_poem',
                '作词': 'write_ci_poem',
            }
            if text_topic in tp_dict:
                upper_topic = tp_dict[text_topic].upper()
                if self.chat_session.in_topic_categs(upper_topic):
                    self.chat_session.set_keep_topic(Topic(upper_topic, ''))
                    if proposed:
                        self.chat_session.set_context_ref_cache(
                            self.chat_session.get_topic_context_refs_proposed(upper_topic))
                    else:
                        self.chat_session.set_context_ref_cache(self.chat_session.get_topic_categ(upper_topic))
        return '', 0

    def propose_topic(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            new_topic = item_list[0].replace(' ', '').strip()
            if new_topic == "故事":
                # This is to give the end user the chance to pick joke or duanzi, in case he/she does not
                # prefer story.
                # self.chat_session.set_missing_ans("笑话",
                #                                   "更愿意听笑话呀，那好，我就讲一个笑话吧。 _nr_ _func_get_joke_any")
                # self.chat_session.set_missing_ans("段子",
                #                                   "更愿意听段子呀，那好，我就说一个段子吧。 _nr_ _func_get_duanzi_any")
                # This is the main part of the proposal response.
                self.chat_session.set_pending_ans(
                    "yes", "那我就随便讲一个故事啊。 _nr_ _func_get_story_any_para1_false")
                self.chat_session.set_pending_ans("no", "那好吧。有什么我能为你做的，你千万不要客气哈。")
                return self.keep_topic("故事", proposed=True)
            elif new_topic == "随机":
                # This is to give the end user the chance to directly pick one. In case the end user
                # asks "STORY", an extra confirmation will be needed as training all these to use
                # get_missing_ans may not be the best option.
                # self.chat_session.set_missing_ans("笑话", "那我就讲一个笑话吧。 _nr_ _func_get_joke_any")
                # self.chat_session.set_missing_ans("段子", "那我就说一个段子吧。 _nr_ _func_get_duanzi_any")

                sel = random.choice(['story', 'joke', 'duanzi']).upper()
                if sel == "STORY":
                    self.chat_session.set_pending_ans(
                        "yes", "那我就讲一个故事吧。 _nr_ _func_get_story_any_para1_false")
                    self.chat_session.set_pending_ans("no", "那好吧。有什么我能为你做的，你千万不要客气哈。")
                    return self.keep_topic("故事", proposed=True)
                elif sel == "JOKE":
                    self.chat_session.set_pending_ans("yes", "那我就讲一个笑话吧。 _nr_ _func_get_joke_any")
                    self.chat_session.set_pending_ans("no", "那好吧。有啥能为你效劳的，你一定告诉我哟。")
                    return self.keep_topic("笑话", proposed=True)
                else:  # DUANZI
                    self.chat_session.set_pending_ans("yes", "那我就说一个段子吧。 _nr_ _func_get_duanzi_any")
                    self.chat_session.set_pending_ans("no", "那好吧。有啥能为你服务的，你一定告诉我哟。")
                    return self.keep_topic("段子", proposed=True)
            elif new_topic == "写诗":
                self.chat_session.set_pending_ans("yes", "好的，献丑了。 _nr_ _func_compose_poem_any")
                self.chat_session.set_pending_ans("no", "哈哈，看来是我太急于显摆了。")
                return self.keep_topic("写诗", proposed=True)
            elif new_topic == "作词":
                self.chat_session.set_pending_ans("yes", "好的，没问题。 _nr_ _func_compose_ci_poem_any")
                self.chat_session.set_pending_ans("no", "哈哈，看来是我太急于表现了。")
                return self.keep_topic("作词", proposed=True)
        return '', 0

    def beg_for(self, something):
        if something and something.upper() == "TREAT":
            self.chat_session.treat_begged = True
            self.chat_session.set_context_ref("你想让我请你吃饭")
            yes_list = [
                "可以好好搓一顿吗？", "咱们能好好搓一顿吗？", "咱们可以大吃一顿吗？好久没吃到好吃的了。"
            ]
            self.chat_session.set_pending_ans_with_cref("yes", random.choice(yes_list),
                                                        "我同意请你吃饭。你问是否可以大吃一顿")
            self.chat_session.set_pending_ans("no", "这么小气呀，那就算了吧。")
        return '', 0

    def beg_for_if_not(self, something):
        if something:
            upper_thing = something.upper()
            if upper_thing == "TREAT":
                if self.chat_session.treat_begged:
                    return '', 0
                self.chat_session.treat_begged = True
                self.chat_session.set_context_ref("你想让我请你吃饭")
                yes_list = [
                    "可以好好搓一顿吗？", "谢谢啊！对了，咱们能好好搓一顿吗？",
                    "咱们可以大吃一顿吗？好久没吃到好吃的了。"
                ]
                self.chat_session.set_pending_ans_with_cref("yes", random.choice(yes_list),
                                                            "我同意请你吃饭。你问是否可以大吃一顿")
                self.chat_session.set_pending_ans("no", "这么小气呀，唉，那算了吧。")
                ret_list = [
                    "要不你请我吃顿饱饭吧，好吗？", "对了，你可以请我吃饭吗？"
                ]
                return random.choice(ret_list), 0
        return '', 0

    def run_last_topic(self, sentence, in_question):
        if self.chat_session.last_topic:
            if self.chat_session.last_topic.title == "STORY":
                return self.continue_story(sentence, in_question)
            elif self.chat_session.last_topic.title == "JOKE":
                return self.get_joke_any()
            elif self.chat_session.last_topic.title == "DUANZI":
                return self.get_duanzi_any()
            elif self.chat_session.last_topic.title == "POEM":
                return self.get_poem_any()
            elif self.chat_session.last_topic.title == "WRITE_POEM":
                return self._compose_poem(self.chat_session.last_topic.value)
            elif self.chat_session.last_topic.title == "CI_POEM":
                return self.compose_ci_poem(self.chat_session.last_topic.value)
            elif self.chat_session.last_topic.title == "WRITE_CI_POEM":
                return self.compose_ci_poem(self.chat_session.last_topic.value)

        need_retry = 0
        if self.chat_session.has_context():
            # Did not find last topic or any topic above, and will try the context ref
            # since it is available.
            # print("coming here: No last topic, but context ref is available.")
            self.chat_session.keep_context = True
            need_retry = 1

        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            return item_list[0].replace(' ', ''), need_retry
        else:
            return "抱歉，不过您想要听点什么方面的呢？", need_retry

    """
    # Rule 5.1: Context reference (based on context cat 4: context ref)
    """
    def set_context_ref(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            ref_text = item_list[0].replace(' ', '').strip()
            if ref_text:
                self.chat_session.set_context_ref(ref_text)
                if self.chat_session.context_prefix['state'] == self.chat_session.city_weather_context_prefix:
                    print("COMING HERE TO CLEAR CONTEXT PREFIX ...")
                    self.chat_session.clear_context_prefix()
        return '', 0

    def set_context_ref_cache(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            ref_cache = item_list[0].replace(' ', '').strip()
            if ref_cache:
                self.chat_session.set_context_ref_cache(ref_cache)
        return '', 0

    def set_categ(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            ref_text = item_list[0].replace(' ', '').strip()
            if ref_text:
                self.chat_session.set_categ(ref_text)
        return '', 0

    def set_context_by_quest(self, sentence, in_question):
        if in_question:
            self.chat_session.set_context_ref(in_question)
        return '', 0

    def keep_context_if_any(self):
        # Keeps the context_ref and pronoun list for the next round
        if self.chat_session.has_context():
            self.chat_session.keep_context = True
            self.chat_session.this_context_kept = True
        if self.chat_session.get_decision_pronoun_ref_text():
            self.chat_session.keep_pronoun_ref = True
        self.chat_session.keep_topic += 1
        self.chat_session.keep_niobj_topic = True
        return '', 0

    def get_pending_yes_with_context(self, sentence):
        need_retry = 0  # default to be 0, which is no retry
        # Try to use stored information
        if self.chat_session.proposed_pending_yes_quest:  # proposed pending question is available
            self.chat_session.clear_pending_ans_yes_no_context()  # ret_text below will always be None
            need_retry = 4

        ret_text = self.chat_session.get_pending_ans('yes')
        if not ret_text:
            # Retrieve the default response in the prediction, if available
            item_list = re.findall(r'\[(.*?)\]', sentence)
            if len(item_list) == 1 and item_list[0]:
                ret_text = item_list[0].replace(' ', '')

        # This type of sentences are normally ambiguous, and keeping context is also important
        self.keep_context_if_any()

        if not ret_text:  # No any useful information available
            ret_text = "我觉得还是不要乱说的好。"
        return self._output_sentence_with_inner_function_executed(ret_text), need_retry

    def retry_context_ref(self):
        if self.chat_session.has_context():
            self.chat_session.keep_context = True
            return '', 1
        return '', 0

    def retry_context_ref_with_base(self, sentence):
        if self.chat_session.has_context():
            self.chat_session.keep_context = True

            item_list = re.findall(r'\[(.*?)\]', sentence)
            if len(item_list) == 1 and item_list[0]:
                self.chat_session.context_ref_base = to_dense_text(item_list[0])
            return '', 1
        return '', 0

    def retry_pron_ctxt_ref(self):
        # pronoun_dist = 1
        if not self.chat_session.pronoun_ref_text:
            pronoun_ref, _ = self.chat_session.get_last_pronoun()
            xgg_pro_ref = self.chat_session.get_xgg_categ_pronoun()
            if pronoun_ref:
                self.chat_session.pronoun_ref_text = pronoun_ref.replace('_kehu_', '我')
            elif xgg_pro_ref:
                self.chat_session.pronoun_ref_text = xgg_pro_ref

        if self.chat_session.has_pronoun_context():
            if self.chat_session.context_ref_text:
                self.chat_session.keep_context = True
            return '', 2
        return '', 0

    # def retry_pron_ctxt_ref_with_base(self, sentence):
    #     # pronoun_dist = 1
    #     if not self.chat_session.pronoun_ref_text:
    #         pronoun_ref, _ = self.chat_session.get_last_pronoun()
    #         xgg_pro_ref = self.chat_session.get_xgg_categ_pronoun()
    #         if pronoun_ref:
    #             self.chat_session.pronoun_ref_text = pronoun_ref.replace('_kehu_', '我')
    #         elif xgg_pro_ref:
    #             self.chat_session.pronoun_ref_text = xgg_pro_ref
    #
    #     if self.chat_session.has_pronoun_context():
    #         if self.chat_session.context_ref_text:
    #             self.chat_session.keep_context = True
    #
    #         item_list = re.findall(r'\[(.*?)\]', sentence)
    #         if len(item_list) == 1 and item_list[0]:
    #             self.chat_session.context_ref_base = item_list[0].replace(' ', '')
    #         return '', 2
    #     return '', 0

    def retry_pta3_ctxt_ref(self):
        # last_topic = self.chat_session.last_topic
        # if last_topic and self.chat_session.context_ref_text:
        #     if last_topic.title in ['POEM', 'WRITE_POEM', 'WRITE_CI_POEM'] or \
        #             (last_topic.title == 'LYRIC' and last_topic.value):
        #         self.chat_session.keep_context = True
        #         return '', 2

        pronoun_ref = self.chat_session.get_pronoun('ta3')
        if pronoun_ref:
            self.chat_session.pronoun_ref_text = pronoun_ref
            self.chat_session.try_pronoun_first = True

        if self.chat_session.has_pronoun_context():
            if self.chat_session.context_ref_text:
                self.chat_session.keep_context = True
            return '', 2
        return '', 0

    def retry_context_or_last_quest(self):
        # This retry does not use categ_ref_text
        if self.chat_session.context_ref_text or self.chat_session.last_question:
            self.chat_session.keep_context = True
            return '', 3
        return '', 0

    """
    # Rule 5.2: Pronoun reference (based on context cat 4: pronoun ref)
    """
    def set_pronoun(self, sentence, key):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            value = item_list[0].replace(' ', '').strip()
            if key == 'ta1_ta2':
                self.chat_session.set_pronoun('ta1', value)
                self.chat_session.set_pronoun('ta2', value)
            elif key == 'ta2_ta3':
                self.chat_session.set_pronoun('ta2', value)
                self.chat_session.set_pronoun('ta3', value)
            elif key == 'ta1_tamen1':
                self.chat_session.set_pronoun('ta1', value)
                self.chat_session.set_pronoun('tamen1', value)
            elif key == 'ta3_tamen3':
                self.chat_session.set_pronoun('ta3', value)
                self.chat_session.set_pronoun('tamen3', value)
            else:
                self.chat_session.set_pronoun(key, value)
                value = value.lower()
                if value.startswith('_kehu_') and not self.chat_session.gender:
                    rel_part = value.replace('_kehu_', '').strip()
                    if rel_part in ['男友', '前男友', '前任男友', '老公', '前夫', '前老公', '前任老公']:
                        self.chat_session.gender = '女'
                    elif rel_part in ['女友', '前女友', '前任女友', '老婆', '前妻', '前老婆', '前任老婆']:
                        self.chat_session.gender = '男'
        return '', 0

    def retry_def_pronoun(self, sentence, in_question):
        key_map = {
            '他': 'ta1',
            '她': 'ta2',
            '他们': 'tamen1', '他俩': 'tamen1',
            '她们': 'tamen2', '她俩': 'tamen2',
            '它们': 'tamen3', '它俩': 'tamen3',
        }
        need_retry = 0
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 3 and item_list[0] and item_list[1] and item_list[2]:
            pronoun_ref = None
            text_key = item_list[0].replace(' ', '').strip()
            if text_key in key_map:
                pronoun_ref = self.chat_session.get_pronoun(key_map[text_key])
                if pronoun_ref:
                    self.chat_session.pronoun_ref_text = pronoun_ref.replace('_kehu_', '我')
                    self.chat_session.try_pronoun_first = True
                    need_retry = 2
            if self.chat_session.niobj_topic == self.chat_session.niobj_topic_opts['1']:
                # or self.chat_session.niobj_topic == self.chat_session.niobj_topic_opts['4']:
                self.chat_session.keep_niobj_topic = True

            # Cache in_question as the context_ref, which will finally be useful if no valid
            # retry happens
            if in_question:
                self.chat_session.set_context_ref_cache(in_question)
            if need_retry >= 1 and item_list[1].find('_ref_phd_') >= 0 and pronoun_ref:
                # the default return when we can find the pronoun ref
                tmp_ref = pronoun_ref.replace('_kehu_', '你')
                ret_text = to_dense_text(item_list[1]).replace('_ref_phd_', tmp_ref)
            else:  # the default return when we cannot find the pronoun ref
                ret_text = to_dense_text(item_list[2])
                if need_retry == 0 and self.chat_session.context_ref_text:  # no categ_ref_text here
                    # Did not find pronoun_ref, and will try the context ref since it is available
                    self.chat_session.keep_context = True
                    need_retry = 1
            return ret_text, need_retry
        return '', need_retry

    """
    # Rule 6.1: User name and gender (based on context cat 5: context prefix)
    """
    def ask_name_if_not_yet(self):
        if self.chat_session.username['asked']:
            return '', 0
        self.chat_session.username['asked'] = True
        self.chat_session.set_context_prefix(self.chat_session.username_context_prefix)
        return self.chat_session.get_ask_name_option(option=1), 0

    def set_user_name(self, sentence, in_question):
        self.chat_session.clear_context_prefix()
        self.chat_session.keep_context = False
        # No matter the outcome, the name won't be asked again
        self.chat_session.username['asked'] = True

        ret_text = ''
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 2:
            if in_question and in_question.startswith('_st_expect'):
                c_index = in_question.find('：')
                if c_index > 1 and c_index + 1 < len(in_question):
                    in_question = in_question[c_index+1:].strip()
            if in_question:
                resp_text = item_list[1].strip()
                if '_un_final_' in resp_text:
                    xing, ming, title, final = self.ner_predictor.predict_usernames(in_question, is_final=True)
                else:
                    xing, ming, title, final = self.ner_predictor.predict_usernames(in_question)
                print("xing = {}, ming = {}, title = {}, final = {}".format(xing, ming, title, final))
                if final:
                    self.chat_session.username['final'] = final
                if xing:
                    self.chat_session.username['xing'] = xing
                if ming:
                    self.chat_session.username['ming'] = ming
                if title:
                    self.chat_session.username['title'] = title
                    if title in ['姐', '姨']:
                        self.chat_session.gender = '女'
                    elif title in ['兄', '叔']:
                        self.chat_session.gender = '男'
                    else:
                        self.chat_session.gender = None
                    if title == '名字' and xing and ming:
                        self.chat_session.username['final'] = xing + ming
                    elif title in ['兄', '姐'] and ming:
                        self.chat_session.username['final'] = ming + title
                    elif title != '名字' and xing:
                        self.chat_session.username['final'] = xing + title
                if xing and not title and not final:
                    title = '老师'
                    self.chat_session.username['title'] = title
                    self.chat_session.username['final'] = xing + title
                if item_list[1]:
                    ret_text = item_list[1].replace(' ', '')
                    if xing and title:
                        ret_text = ret_text.replace('_un_xing_tit_', xing+title)
                    if ming and title:
                        ret_text = ret_text.replace('_un_ming_tit_', ming+title)

                    if self.chat_session.username['final']:
                        ret_text = ret_text.replace('_un_final_', self.chat_session.username['final'])
                    elif xing and ming:
                        ret_text = ret_text.replace('_un_final_', xing+ming)
                    else:
                        ret_text = ret_text.replace('_un_final_', '亲')

                    if xing and ming:
                        ret_text = ret_text.replace('_un_full_', xing+ming)
                        ref_name = xing+ming
                    elif self.chat_session.username['final']:
                        ret_text = ret_text.replace('_un_full_', self.chat_session.username['final'])
                        ref_name = final
                    else:
                        ret_text = ret_text.replace('_un_full_', '亲')
                        ref_name = ''

                    # In case the replacements above failed.
                    if '您' in ret_text:
                        ret_text = ret_text.replace('_un_xing_tit_', '您').replace('_un_ming_tit_', '您')
                    else:
                        ret_text = ret_text.replace('_un_xing_tit_', '你').replace('_un_ming_tit_', '你')

                    if ref_name:
                        ref_desc = self.knowledge_base.spec_uname_dict.get(ref_name.lower())
                        if ref_desc:
                            ret_text += ref_desc

        if not self.chat_session.username['final']:
            self.chat_session.username['final'] = '亲'

        return ret_text, 0

    def _set_user_name_rejected(self):
        self.chat_session.clear_context_prefix()
        self.chat_session.username['asked'] = True
        self.chat_session.username['final'] = '亲'
        # return '', 0

    def set_iam_name(self, sentence, in_question):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1:
            if in_question and in_question.startswith('_st_expect'):
                c_index = in_question.find('：')
                if c_index > 1 and c_index + 1 < len(in_question):
                    in_question = in_question[c_index + 1:].strip()
            if in_question:
                xing, ming, title, final = self.ner_predictor.predict_usernames(in_question, is_final=True)
                print("xing = {}, ming = {}, title = {}, final = {}".format(xing, ming, title, final))
                if final:
                    self.chat_session.iam_name = final
        return '', 0

    def get_user_name(self):
        return self._get_user_name(), 0

    def _get_user_name(self):
        if self.chat_session.username['final']:
            return self.chat_session.username['final']
        if self.chat_session.iam_name:
            return self.chat_session.iam_name
        return '亲'

    def _get_user_name_full(self):
        if self.chat_session.username['xing'] and self.chat_session.username['ming']:
            return self.chat_session.username['xing'] + self.chat_session.username['ming']
        return self._get_user_name()

    def _get_user_name_xing(self):
        if self.chat_session.username['title'] != '名字' and self.chat_session.username['xing']:
            return self.chat_session.username['xing'] + self.chat_session.username['title']
        return self._get_user_name()

    def _get_user_name_with_title_updated(self, title):
        self.chat_session.username['title'] = title

        if title in ['姐', '姨']:
            self.chat_session.gender = '女'
        elif title in ['兄', '叔']:
            self.chat_session.gender = '男'
        # elif title != '名字':
        #     self.chat_session.gender = None

        if title == '名字' and self.chat_session.username['xing'] and self.chat_session.username['ming']:
            self.chat_session.username['final'] = self.chat_session.username['xing'] + self.chat_session.username['ming']
        elif title in ['兄', '姐'] and self.chat_session.username['ming']:
            self.chat_session.username['final'] = self.chat_session.username['ming'] + title
        elif title != '名字':
            if self.chat_session.username['xing']:
                self.chat_session.username['final'] = self.chat_session.username['xing'] + title
            elif self.chat_session.username['ming']:
                self.chat_session.username['final'] = self.chat_session.username['ming'] + title
            elif title == '叔':
                self.chat_session.username['final'] = '叔叔'
            elif title == '姨':
                self.chat_session.username['final'] = '阿姨'

        if not self.chat_session.username['final']:
            self.chat_session.username['final'] = '亲'

        return self.chat_session.username['final']

    # def set_gender(self, sentence):
    #     item_list = re.findall(r'\[(.*?)\]', sentence)
    #     if len(item_list) == 1 and item_list[0]:
    #         gender = item_list[0].replace(' ', '')
    #         if gender in ['男', '女']:
    #             self.chat_session.gender = gender
    #     return '', 0

    def output_if_gender(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 3:
            if self.chat_session.gender == '男':
                output = item_list[0] if item_list[0] else ''
            elif self.chat_session.gender == '女':
                output = item_list[1] if item_list[1] else ''
            else:
                output = item_list[2] if item_list[2] else ''
            return to_dense_text(output), 0
        return '', 0

    def _clear_user_name(self):
        self.chat_session.username['xing'] = None
        self.chat_session.username['ming'] = None
        self.chat_session.username['title'] = None
        self.chat_session.username['final'] = None
        self.chat_session.gender = None
        # return '', 0

    def scan_usr_set_block(self, in_question, sentence):
        sentence = to_dense_text(sentence)
        ind_list = [(m.start(0), m.end(0)) for m in re.finditer(USR_SET_CAT, sentence)]
        if len(ind_list) == 0:
            return sentence

        prev_end_idx = -1
        out_sent = ''
        for idx, (start, end) in enumerate(ind_list):
            blk_txt = sentence[start:end].replace("<usr_set>", "").replace("</usr_set>", "").strip()
            blk_txt = blk_txt.replace(':', '：')
            set_data = blk_txt.split('：')
            set_cate, set_attr = set_data[0], set_data[1]

            out_text, status_id = '', 0
            if set_cate == '名字':
                if set_attr == '拒绝':
                    self._set_user_name_rejected()
                    status_id = 1
                elif set_attr == '清除':
                    self._clear_user_name()
                    status_id = 1
            elif set_cate == '称呼':
                if set_attr in ['兄', '姐', '叔', '姨', '名字']:
                    out_text = self._get_user_name_with_title_updated(set_attr)
            elif set_cate == '性别':
                if set_attr in ['男', '女']:
                    self.chat_session.gender = set_attr
                    status_id = 1

            if out_text or status_id == 1:
                if prev_end_idx > 0:
                    out_sent += sentence[prev_end_idx:start]
                else:
                    out_sent += sentence[:start]
                out_sent += out_text
                prev_end_idx = end
            else:
                raise ValueError("抱歉，我还在学习中，暂时无法正确回复你这个问题呢！")

        if prev_end_idx > 0:
            out_sent += sentence[prev_end_idx:]
        return out_sent

    """
    # Rule 6.2: Misc name/entry (CELEB, CAP_NATION, NATION_CAP, MEANING, WHATIS, WHOWHATIS, NIOBJ, CELEB_ATTR, 
    # ANI_ATTR, HIST_EVENTS, SPORT_EVENT_DATA)
    """
    @staticmethod
    def _remove_prefix_for_ner_predictor(question):
        if question.startswith('_st_expect'):
            c_index = question.find('：')
            if c_index > 1 and c_index + 1 < len(question):
                return question[c_index + 1:].strip()
        return question

    def _get_celeb_info(self, qry_key):
        if qry_key:
            key = qry_key.lower()
            self.chat_session.niobj_topic = self.chat_session.niobj_topic_opts['1']
            self.chat_session.keep_niobj_topic = True
            ret_text, _ = self._get_celeb_info_from_kbase(key)
            return ret_text
        return None

    def _get_celeb_info_from_kbase(self, celeb_name):
        celeb_desc = self.knowledge_base.get_celeb_info_from_local(celeb_name)
        if celeb_desc:
            return celeb_desc.strip(), True

        isp, wp = self.knowledge_base.get_first_paragraph_from_wiki(celeb_name)
        if wp and isp:  # Verify this entry is really about a person so that no tricked by: 谁是工人？
            return wp.strip(), True

        out_text = ''
        c_name_len = len(celeb_name)
        if c_name_len > 3 and celeb_name[-3:] in ['女朋友', '男朋友']:
            out_text = "这么关心别人的{}干嘛？可惜了，我还真不擅长打探他人的隐私呢。".format(celeb_name[-3:])
        elif c_name_len > 2:
            if celeb_name[-2:] in ['老婆', '媳妇', '妻子', '女友', '前妻', '妈妈', '母亲', '奶奶', '祖母', '女儿']:
                out_text = "{}呀？我不知道她是谁啊。矮油，咱还是不要打探别人的家事或隐私了。".format(celeb_name)
            elif celeb_name[-2:] in ['老公', '夫君', '丈夫', '男友', '前夫', '爸爸', '父亲', '爷爷', '祖父', '儿子']:
                out_text = "{}呀？我不知道他是谁啊。矮油，咱还是不要打探别人的家事或隐私了。".format(celeb_name)
            elif celeb_name[-2:] in ['初恋', '前任']:
                out_text = "关心别人的{}干嘛？我可不擅长打探他人的隐私呢。".format(celeb_name[-2:])

        if not out_text:
            if re.search(r'[看闻听说讲读写干做演唱吃喝有]', celeb_name) or len(celeb_name) >= 4:
                out_list = [
                    "晕，我不清楚。你学识渊博，要不你给我讲讲呗。",
                    "我也想知道呢，要不你给介绍介绍呗？",
                ]
            else:
                out_list = [
                    "{}就是{}啊，哈哈哈。".format(celeb_name, celeb_name),
                    "哼，又想考我。那你来说说，{}是谁啊？".format(celeb_name),
                    "我也想知道是谁呢，要不你给讲讲呗？",
                ]
            out_text = random.choice(out_list)
        return out_text, False

    def get_nation_cap_info(self, sentence, in_question):
        ret_text = ''
        if in_question:
            in_question = self._remove_prefix_for_ner_predictor(in_question)
            if re.search(r'(哪个?国?家?|哪个?地方?|谁|哪[里儿])([的之])?首都', in_question.replace(' ', '')):
                cap_name = self.ner_predictor.predict_capname(in_question)
                self.chat_session.niobj_topic = self.chat_session.niobj_topic_opts['3']
                self.chat_session.keep_niobj_topic = True
                if cap_name:
                    ret_text, _ = self._get_nation_from_cap_from_kbase(cap_name)
            else:
                nation_name, veri_cap = self.ner_predictor.predict_nationname(in_question)
                self.chat_session.niobj_topic = self.chat_session.niobj_topic_opts['2']
                self.chat_session.keep_niobj_topic = True
                if nation_name:
                    ret_text, _ = self._get_cap_from_nation_from_kbase(nation_name, veri_cap)

        return ret_text, 0

    def _get_cap_from_nation_from_kbase(self, nation_name, veri_cap=None):
        cap_name = self.knowledge_base.nation_cap_dict.get(nation_name)
        if not cap_name:
            if nation_name == '台湾':
                return "台湾是中国的一个省呀。台北是其省会，不能叫首都的。", False
            found, _ = self._get_shi_by_sheng_from_kbase(nation_name, veri_cap, '首府')
            if found:
                no_text = "可是首都一般用来指中央政府所在地，而不指某个省或州的地方政府所在地呀。"
                self.chat_session.propose_pending_yes_quest("{}的首府在哪里？".format(nation_name))
                self.chat_session.set_pending_ans('no', no_text)
                return "你是想问{}的首府在哪里吗？".format(nation_name), True
            return random.choice([
                "我不知道它的首都是哪里呢。",
                "不好意思哈，{}不知道它的首都在哪啊{{抓狂}}".format(self.chat_session.bot_name), ]), False
        else:
            cap_name = cap_name.strip()
            if veri_cap:
                if cap_name == veri_cap or self.knowledge_base.whatis_alias_dict.get(cap_name) == veri_cap or \
                        cap_name == self.knowledge_base.whatis_alias_dict.get(veri_cap):
                    out_list = [
                        "对的，{}的首都就是{}啊。".format(nation_name, veri_cap),
                        "对的，{}的首都是{}，地理小常识，哈哈。".format(nation_name, veri_cap),
                        "对呀，{}的首都是{}，这点地理常识我还是有的。".format(nation_name, veri_cap),
                        "对呀，{}的首都就是{}，地理我还是学过一些的。".format(nation_name, veri_cap),
                    ]
                else:
                    out_list = [
                        "好像不对吧，印象中{}的首都是{}啊。".format(nation_name, cap_name),
                        "应该不对吧，我记得{}的首都是{}啊。".format(nation_name, cap_name),
                        "好像不对吧，{}的首都是{}，地理我还是学过一些的。".format(nation_name, cap_name),
                        "应该不对吧，{}的首都是{}，这个难不倒我的，哈哈哈。".format(nation_name, cap_name),
                    ]
            else:
                out_list = [
                    "{}的首都是{}啊，这个我知道的。".format(nation_name, cap_name),
                    "{}的首都是{}，地理小常识，哈哈。".format(nation_name, cap_name),
                    "{}的首都是{}，这点地理常识我还是有的。".format(nation_name, cap_name),
                    "{}的首都是{}吧，这个难不倒我的，哈哈哈。".format(nation_name, cap_name),
                    "{}的首都是{}啊，这个我记得。".format(nation_name, cap_name)
                ]
            return random.choice(out_list), True

    def _get_nation_from_cap_from_kbase(self, cap_name):
        nation_name = self.knowledge_base.cap_nation_dict.get(cap_name)
        if not nation_name and (cap_name.endswith('市') or cap_name.endswith('城')):
            nation_name = self.knowledge_base.cap_nation_dict.get(cap_name[:-1])
        if not nation_name:
            if cap_name in ['台北', '台北市']:
                return "台北是台湾的省会，而台湾是中国的一个省，所以不能叫做首都的。", False
            return random.choice([
                "我不知道它是哪个国家的首都呢。",
                "不好意思哈，{}不知道它是谁的首都啊{{抓狂}}".format(self.chat_session.bot_name), ]), False
        else:
            nation_name = nation_name.strip()
            out_list = [
                "{}是{}的首都啊，这个我知道的。".format(cap_name, nation_name),
                "{}是{}的首都，地理小常识，哈哈。".format(cap_name, nation_name),
                "{}是{}的首都，这点地理常识我还是有的。".format(cap_name, nation_name),
                "{}是{}的首都吧，这个难不倒我的，哈哈。".format(cap_name, nation_name),
                "{}是{}的首都啊，这个我记得。".format(cap_name, nation_name)
            ]
            return random.choice(out_list), True

    def get_province_city_info(self, sentence, in_question):
        ret_text = ''
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if in_question and len(item_list) == 2:
            cate_key = item_list[1].replace(' ', '').strip()
            in_question = self._remove_prefix_for_ner_predictor(in_question)
            if cate_key in ['省会', '首府']:
                sheng, shi = self.ner_predictor.predict_shenghui_name(in_question)
                if re.search(r'(哪个?省?份?|哪个?(地方?|自治区|州)|谁|哪[里儿])([的之])?(省会|州府|首府)',
                             to_dense_text(in_question)):
                    if shi:
                        self.chat_session.set_pronoun('ta3', shi)
                        self.chat_session.set_categ("是哪里的{}".format(cate_key))
                        if shi in self.knowledge_base.cap_nation_dict or \
                                (re.search(r'[城市]$', shi) and shi[:-1] in self.knowledge_base.cap_nation_dict):
                            no_key = '省会' if re.search(r'省会', to_dense_text(in_question)) else '首府'
                            no_text = "可是{}一般用来指地方政府所在地，而不是某个国家的中央政府所在地呀。".format(no_key)
                            self.chat_session.propose_pending_yes_quest("{}是哪个国家的首都？".format(shi))
                            self.chat_session.set_pending_ans('no', no_text)
                            return "你是想问{}是谁的首都吗？".format(shi), 0
                        ret_text = self._get_sheng_by_shi_from_kbase(shi, cate_key)
                elif sheng:
                    self.chat_session.set_pronoun('ta3', sheng)  # might be replaced later
                    self.chat_session.set_categ("{}是哪里".format(cate_key))
                    if not shi and sheng in self.knowledge_base.nation_cap_dict:
                        no_key = '省会' if re.search(r'省会', to_dense_text(in_question)) else '首府'
                        no_text = "可是{}一般用来指地方政府所在地，而不是某个国家的中央政府所在地呀。".format(no_key)
                        self.chat_session.propose_pending_yes_quest("{}的首都在哪里？".format(sheng))
                        self.chat_session.set_pending_ans('no', no_text)
                        return "你是想问{}的首都在哪里吗？".format(sheng), 0
                    _, ret_text = self._get_shi_by_sheng_from_kbase(sheng, shi, cate_key)
            elif cate_key == '简称':
                di_name, jc_name = self.ner_predictor.predict_ss_jiancheng_name(in_question)
                if re.search(r'哪个?(省|自治区)', in_question.replace(' ', '')):
                    from_ss = 1
                elif re.search(r'哪([个座])?[城市]', in_question.replace(' ', '')):
                    from_ss = 2
                elif re.search(r'(哪个?地方?|谁|哪[里儿])([的之])?简称|是[谁哪]', in_question.replace(' ', '')):
                    from_ss = 3
                else:
                    from_ss = 0
                if from_ss == 0 and di_name:
                    self.chat_session.set_pronoun('ta3', di_name)
                    self.chat_session.set_categ("简称叫什么")
                    ret_text = self._get_jiancheng_by_shengshi_from_kbase(di_name)
                elif from_ss in [1, 2, 3] and jc_name:
                    self.chat_session.set_pronoun('ta3', jc_name)  # might be replaced later
                    self.chat_session.set_categ("是哪里的简称")
                    ret_text = self._get_shengshi_by_jiancheng_from_kbase(jc_name, from_ss)

        return ret_text, 0

    def _get_shi_by_sheng_from_kbase(self, sheng_name, veri_shi, cate_key):
        new_sheng = self._get_trimmed_chinese_location(sheng_name)
        shi_name = self.knowledge_base.prov_city_dict.get(new_sheng)
        is_state = False
        if not shi_name:
            new_sheng = re.sub(r'[美米]国', '', new_sheng)
            if new_sheng.endswith('州') and len(new_sheng) > 2:
                new_sheng = new_sheng[:-1]
            shi_name = self.knowledge_base.state_shoufu_dict.get(new_sheng)
            is_state = True  # if not shi_name, this value won't be used
        if not shi_name:
            return False, random.choice([
                "晕，我不知道它的{}是哪呢。".format(cate_key),
                "不好意思哈，{}不知道它的{}在哪啊{{抓狂}}".format(self.chat_session.bot_name, cate_key), ])
        else:
            shi_name = shi_name.strip()
            cate_desc = '省会' if not is_state else '首府'
            if veri_shi:
                if shi_name == veri_shi:
                    out_list = [
                        "对的，{}的{}就是{}啊。".format(sheng_name, cate_desc, shi_name),
                        "对的，{}的{}是{}，地理小常识，哈哈哈。".format(sheng_name, cate_desc, shi_name),
                        "对呀，{}的{}是{}，这点地理常识我还是有的。".format(sheng_name, cate_desc, shi_name),
                        "对呀，{}的{}就是{}，地理我还是学过一些的。".format(sheng_name, cate_desc, shi_name),
                    ]
                else:
                    out_list = [
                        "好像不对吧，{}的{}是{}啊，这个我知道的。".format(sheng_name, cate_desc, shi_name),
                        "应该不对吧，{}的{}是{}啊，这个我记得的。".format(sheng_name, cate_desc, shi_name),
                        "好像不对吧，{}的{}是{}，地理我还是学过一些的。".format(sheng_name, cate_desc, shi_name),
                        "应该不对吧，{}的{}是{}，这个难不倒我的，哈哈哈。".format(sheng_name, cate_desc, shi_name),
                    ]
            else:
                self.chat_session.set_pronoun('ta3', shi_name)
                out_list = [
                    "{}的{}是{}啊，这个我知道的。".format(sheng_name, cate_desc, shi_name),
                    "{}的{}是{}，地理小常识，哈哈哈。".format(sheng_name, cate_desc, shi_name),
                    "{}的{}是{}，地理我还是学过一些的。".format(sheng_name, cate_desc, shi_name),
                    "{}的{}是{}吧，这个难不倒我的，哈哈哈。".format(sheng_name, cate_desc, shi_name),
                    "{}的{}是{}啊，这个我记得。".format(sheng_name, cate_desc, shi_name),
                ]
            return True, random.choice(out_list)

    def _get_sheng_by_shi_from_kbase(self, shi_name, cate_key):
        new_shi = re.sub(r'中国|市', '', shi_name)
        sheng_name = self.knowledge_base.city_prov_dict.get(new_shi)
        is_state = False
        if not sheng_name:
            new_shi = re.sub(r'[美米]国', '', new_shi)
            sheng_name = self.knowledge_base.shoufu_state_dict.get(new_shi)
            if sheng_name:
                sheng_name += '州'
                is_state = True
        if sheng_name:
            sheng_name = sheng_name.strip()
            cate_desc = '省会' if not is_state else '首府'
            out_list = [
                "{}是{}的{}啊，这个我知道的。".format(shi_name, sheng_name, cate_desc),
                "{}是{}的{}，地理小常识，哈哈。".format(shi_name, sheng_name, cate_desc),
                "{}是{}的{}，地理我还是学过一些的。".format(shi_name, sheng_name, cate_desc),
                "{}是{}的{}吧，这个难不倒我的，哈哈哈。".format(shi_name, sheng_name, cate_desc),
                "{}是{}的{}啊，这个我记得。".format(shi_name, sheng_name, cate_desc)
            ]
        else:
            out_list = [
                "我不知道它是谁的{}呢。".format(cate_key),
                "不好意思哈，{}不知道它是谁的{}啊{{抓狂}}".format(self.chat_session.bot_name, cate_key),
            ]
        return random.choice(out_list)

    def _get_jiancheng_by_shengshi_from_kbase(self, shengshi_name):
        new_name = self._get_trimmed_chinese_location(shengshi_name)
        jc_name = self.knowledge_base.prov_city_jc_dict.get(new_name)
        if jc_name:
            out_list = [
                "{}的简称是{}啊，这个我知道的。".format(shengshi_name, jc_name),
                "{}简称{}，这个难不倒我的，哈哈哈。".format(shengshi_name, jc_name),
                "{}简称{}呀，这个我记得。".format(shengshi_name, jc_name)
            ]
        else:
            out_list = [
                "我不知道它的简称呢。",
                "不好意思哈，{}不知道它简称什么啊{{抓狂}}".format(self.chat_session.bot_name),
            ]
        return random.choice(out_list)

    def _get_shengshi_by_jiancheng_from_kbase(self, jc_name, from_ss):
        # from_ss == 1: sheng; from_ss == 2: shi; from_ss == 3: sheng & shi
        assert from_ss in [1, 2, 3]
        if from_ss == 1:
            sheng_name = self.knowledge_base.jc_prov_dict.get(jc_name)
            if not sheng_name:
                shi_name = self.knowledge_base.jc_city_dict.get(jc_name)
                if shi_name:
                    if '（' in shi_name:
                        return "我不清楚它是哪个省份的简称，但我知道简称{}的城市为{}。".format(jc_name, shi_name)
                    else:
                        return "我不清楚它是哪个省份的简称，但我知道{}简称{}。".format(shi_name, jc_name)
            else:
                self.chat_session.set_pronoun('ta3', sheng_name)
                return random.choice([
                    "{}是{}的简称，这个我知道的。".format(jc_name, sheng_name),
                    "{}是{}的简称呀，这个难不倒我的，哈哈哈。".format(jc_name, sheng_name),
                    "{}是{}的简称，这个我记得。".format(jc_name, sheng_name), ])
        elif from_ss == 2:
            shi_name = self.knowledge_base.jc_city_dict.get(jc_name)
            if not shi_name:
                sheng_name = self.knowledge_base.jc_prov_dict.get(jc_name)
                if sheng_name:
                    return "我不清楚它是哪个城市的简称，但我知道{}简称{}。".format(sheng_name, jc_name)
            else:
                self.chat_session.set_pronoun('ta3', shi_name)
                return random.choice([
                    "{}是{}的简称，这个我知道的。".format(jc_name, shi_name),
                    "{}是{}的简称呀，这个难不倒我的，哈哈哈。".format(jc_name, shi_name),
                    "{}是{}的简称，这个我记得。".format(jc_name, shi_name), ])
        else:
            sheng_name = self.knowledge_base.jc_prov_dict.get(jc_name)
            shi_name = self.knowledge_base.jc_city_dict.get(jc_name)
            if sheng_name and shi_name:
                return "这我知道，它不但是{}的简称，而且{}也简称{}呢。".format(sheng_name, shi_name, jc_name)
            elif sheng_name or shi_name:
                find_name = sheng_name if sheng_name else shi_name
                self.chat_session.set_pronoun('ta3', find_name)
                return random.choice([
                    "{}是{}的简称，这个我知道的。".format(jc_name, find_name),
                    "{}是{}的简称呀，这个难不倒我的，哈哈哈。".format(jc_name, find_name),
                    "{}是{}的简称，这个我记得。".format(jc_name, find_name), ])
        return "晕，我不清楚它是谁的简称呢。"

    # def get_meaning_from_entry(self, sentence, in_question):
    #     ret_text = ''
    #     if in_question:
    #         in_question = self._remove_prefix_for_ner_predictor(in_question)
    #         mean_ents = self.ner_predictor.predict_meaning(in_question)
    #         self.chat_session.niobj_topic = self.chat_session.niobj_topic_opts['4']
    #         self.chat_session.keep_niobj_topic = True
    #         if mean_ents:
    #             ret_text, _ = self._get_meaning_from_entry_from_kbase(mean_ents)
    #
    #     return ret_text, 0

    # def _get_meaning_from_entry_from_kbase(self, mean_ents):
    #     mean_desc = self.knowledge_base.entry_meaning_dict.get(mean_ents.lower())
    #     if not mean_desc:
    #         mean_desc = self.knowledge_base.entry_whatis_dict.get(mean_ents.lower())
    #     if mean_desc:
    #         return mean_desc.strip(), True
    #
    #     isp, wp = self.knowledge_base.get_first_paragraph_from_wiki(mean_ents)
    #     if wp and not isp:  # Make sure this entry is not about a person
    #         return wp.strip(), True
    #
    #     out_list = [
    #         "晕，我不清楚是什么意思。你见多识广，要不你给我讲讲吧。",
    #         "我也想知道呀。再说了，哪有那么多意思，人家可是只对你有意思呢{捂脸}",
    #     ]
    #     return random.choice(out_list), False

    def _get_whatis_from_entry(self, qry_key):
        if qry_key:
            key = qry_key.lower()
            self.chat_session.niobj_topic = self.chat_session.niobj_topic_opts['5']
            self.chat_session.keep_niobj_topic = True
            ret_text, _ = self._get_whatis_from_entry_from_kbase(qry_key)
            return ret_text
        return None

    def _get_whatis_from_entry_from_kbase(self, whis_ents):
        whis_desc = self.knowledge_base.get_entry_whatis_from_local(whis_ents)
        if not whis_desc:
            whis_desc = self.knowledge_base.entry_meaning_dict.get(whis_ents.lower())
        if whis_desc:
            return whis_desc.strip(), True

        isp, wp = self.knowledge_base.get_first_paragraph_from_wiki(whis_ents)
        if wp and not isp:  # Make sure this entry is not about a person, so that no tricked by: 陆毅是什么？
            return wp.strip(), True

        out_list = [
            "晕，我不清楚是什么。你见多识广，要不你给我介绍介绍吧。",
            "我不知道是啥呢。矮油，管那么多干嘛，简简单单地生活多好呀{捂脸}",
        ]
        return random.choice(out_list), False

    def _get_whowhatis_from_entry(self, qry_key):
        if qry_key:
            key = qry_key.lower()
            ret_text = self._get_whowhatis_from_entry_from_kbase(key)
            return ret_text
        return None

    def _get_whowhatis_from_entry_from_kbase(self, wwis_ents):
        wwis_desc, low_pri = self.knowledge_base.get_celeb_info_from_local(wwis_ents, return_low_pri=True)
        if wwis_desc and not low_pri:  # find info in celeb_dict and not a low_pri case
            self.chat_session.niobj_topic = self.chat_session.niobj_topic_opts['1']
            self.chat_session.keep_niobj_topic = True
            return wwis_desc.strip()

        wwis_desc = self.knowledge_base.get_entry_whatis_from_local(wwis_ents)
        if not wwis_desc:
            wwis_desc = self.knowledge_base.entry_meaning_dict.get(wwis_ents.lower())
        if wwis_desc:  # find info in entry_whatis_dict or entry_meaning_dict
            self.chat_session.niobj_topic = self.chat_session.niobj_topic_opts['5']
            self.chat_session.keep_niobj_topic = True
            return wwis_desc.strip()

        isp, wp = self.knowledge_base.get_first_paragraph_from_wiki(wwis_ents)
        if wp:
            if isp:  # is a person entry
                self.chat_session.niobj_topic = self.chat_session.niobj_topic_opts['1']
            else:
                self.chat_session.niobj_topic = self.chat_session.niobj_topic_opts['5']
            self.chat_session.keep_niobj_topic = True
            return wp.strip()

        return random.choice([
            "晕，我不清楚。你学识丰富，要不你给我介绍一下吧。",
            "我没听说过呢。矮油，管那么多干嘛，简单一点多好呀{捂脸}",
        ])

    def _get_lgst_person_by_key(self, qry_key):
        key = qry_key.lower()
        wwis_desc = self.knowledge_base.get_celeb_info_from_local(key)
        if wwis_desc:  # find info in celeb_dict
            return wwis_desc.strip()

        wwis_desc = self.knowledge_base.get_entry_whatis_from_local(key)
        if wwis_desc:  # find info in entry_whatis_dict
            return wwis_desc.strip()

        isp, wp = self.knowledge_base.get_first_paragraph_from_wiki(key)
        if wp:
            return wp.strip()
        return None

    # def get_lg_person_by_key(self, sentence):
    #     ret_text = self._get_lgst_person_by_key(sentence)
    #     if not ret_text:
    #         ret_text = random.choice([
    #             "晕，我还从来都没听说过呢。唉，真是无知啊。",
    #             "抱歉，我不了解呢。你懂得多，还是你来给讲讲吧。",
    #             "晕，名人的知识我掌握得真不多。你学识丰富，要不你给我介绍一下吧。",
    #         ])
    #     return ret_text, 0

    # def get_maybe_person_by_key(self, sentence):
    #     ret_text = self._get_lgst_person_by_key(sentence)
    #     if not ret_text:
    #         ret_text = random.choice([
    #             "晕，{}笨笨的，不明白你说的半句话呢。".format(self.chat_session.bot_name),
    #             "矮油，{}我不学无术，不理解你说的这个词呢。".format(self.chat_session.bot_name),
    #             "矮油，{}我读书少，不明白你想说什么呢。".format(self.chat_session.bot_name),
    #         ])
    #     return ret_text, 0

    def _get_lgst_thing_by_key(self, qry_key):
        key = qry_key.lower()
        wwis_desc = self.knowledge_base.get_entry_whatis_from_local(key)
        if wwis_desc:  # find info in entry_whatis_dict
            return wwis_desc.strip()

        wwis_desc = self.knowledge_base.get_celeb_info_from_local(key)
        if wwis_desc:  # find info in celeb_dict
            return wwis_desc.strip()

        mean_desc = self.knowledge_base.entry_meaning_dict.get(key)
        if mean_desc:
            return mean_desc.strip()

        isp, wp = self.knowledge_base.get_first_paragraph_from_wiki(key)
        if wp:
            return wp.strip()
        return None

    # def get_lg_thing_by_key(self, sentence):
    #     ret_text = self._get_lgst_thing_by_key(sentence)
    #     if not ret_text:
    #         ret_text = random.choice([
    #             "晕，我还从来都没听说过呢。唉，真是无知啊。",
    #             "抱歉，我不了解呢。你懂得多，还是你来给讲讲吧。",
    #             "晕，我不清楚呢。你学识丰富，要不你给我介绍一下吧。",
    #         ])
    #     return ret_text, 0

    # def get_lg_work_by_key(self, sentence):
    #     ret_text = self._get_lgst_thing_by_key(sentence)
    #     if not ret_text:
    #         ret_text = random.choice([
    #             "晕，我还从来都没听说过呢。唉，真是无知啊。",
    #             "抱歉，我不了解呢。你懂得多，还是你来给讲讲吧。",
    #             "晕，文艺作品方面的知识我懂得真不多。你学识丰富，要不你给我介绍一下吧。",
    #         ])
    #     return ret_text, 0

    # def get_maybe_thing_by_key(self, sentence):
    #     ret_text = self._get_lgst_thing_by_key(sentence)
    #     if not ret_text:
    #         ret_text = random.choice([
    #             "晕，{}笨笨的，不明白你说的半句话呢。".format(self.chat_session.bot_name),
    #             "矮油，{}我不学无术，不理解你说的这个词呢。".format(self.chat_session.bot_name),
    #             "矮油，{}我读书少，不明白你想说什么呢。".format(self.chat_session.bot_name),
    #         ])
    #     return ret_text, 0

    # def get_maybe_work_by_key(self, sentence):
    #     ret_text = self._get_lgst_thing_by_key(sentence)
    #     if not ret_text:
    #         ret_text = random.choice([
    #             "矮油，{}我读书少，不理解你说的这个词呢。".format(self.chat_session.bot_name),
    #             "这是一部文学或艺术作品的名字吗？{}都不曾听说过呢。".format(self.chat_session.bot_name),
    #             "看着像是一部文学或艺术作品的名字，但细节我就不知道了。",
    #         ])
    #     return ret_text, 0

    # def get_xgg_age_in_year(self, sentence, in_question):
    #     year_cat = CalPats.get_cal_year_cat()
    #     year_mat = re.search(year_cat, in_question)
    #     if year_mat:
    #         cal_year = ''.join(in_question[year_mat.start():year_mat.end()].split())
    #         return self._get_xgg_age_ni_ent_cal_year(cal_year)

    def get_niobj_info(self, sentence, in_question):
        # Case 1: Has context
        # 1.1 Found the entry desc, use the desc based on the context
        # 1.2 Did not find the entry desc, check the niobj entry desc
        # 1.2.1 Find the niobj entry desc, use it
        # 1.2.2 Use a general desc based on the context
        # Case 2: No context
        # 2.1 Look for the niobj entry
        # 2.1.1 Find the niobj entry desc, use it
        # 2.1.2 Use a general desc for niobj
        ret_text = ''
        need_retry = 0
        if in_question:
            if self.chat_session.last_topic:
                if self.chat_session.last_topic.title == 'JIERI_QUERY':
                    jr_topic = JieRiTopic.extract_jieri_topic_content(in_question)
                    if jr_topic:
                        is_nongli = CalPats.is_for_nongli(in_question)
                        qy_type = '农历日期描述' if is_nongli else '公历日期描述'

                        jr_name, cal_year = jr_topic.jr_name, jr_topic.cal_year
                        prev_jr_info = self.chat_session.last_topic.value.split("=")
                        if jr_name and cal_year:
                            return self._get_jieri_date_of_year(in_question, qy_type, jr_name, cal_year), 0
                        elif jr_name:
                            if prev_jr_info[1] == 'NO_NEXT':
                                return self._get_jieri_date_and_weekday(qy_type, jr_name), 0
                            elif prev_jr_info[1] == 'MAYBE_NEXT':
                                return self._get_jieri_date_maybe_next(qy_type, jr_name), 0
                            else:  # with cal_year info previously
                                return self._get_jieri_date_of_year(in_question, qy_type, jr_name, prev_jr_info[1]), 0
                        else:  # cal_year
                            return self._get_jieri_date_of_year(in_question, qy_type, prev_jr_info[0], cal_year), 0
                elif self.chat_session.last_topic.title == 'DATE_SPAN_QUERY_WEEKDAY':
                    ds_type, span_idx, repeat_idx, day_idx = CalPats.extract_week_and_or_day_pattern(in_question)
                    _week_opts = ['上上', '上', '这', '下', '下下']
                    prev_wd_info = self.chat_session.last_topic.value.split("=")

                    is_nongli = CalPats.is_for_nongli(in_question)
                    query_type = '农历日期描述' if is_nongli else '公历日期描述'

                    if ds_type >= 1:
                        if in_question.find('周') >= 0:
                            wd = '周'
                        elif in_question.find('星 期') >= 0:
                            wd = '星期'
                        else:
                            wd = '礼拜'
                        tmp_in = in_question + ' _cr_ 农 历 日 期' if prev_wd_info[0] == 'NONGLI' else in_question

                        if ds_type == 3:
                            if repeat_idx is not None:
                                week_a_day = "{}一个{}{}".format(_week_opts[repeat_idx+2], wd, WEEK_DAYS[day_idx])
                            else:  # span_idx is not None:
                                week_a_day = "{}{}的{}{}".format(_week_opts[span_idx+2], wd, wd, WEEK_DAYS[day_idx])
                            return self._get_date_with_span_from_weekday(query_type, week_a_day), 0
                        elif ds_type == 1:
                            if prev_wd_info[1] == 'SPAN':
                                span_idx = int(prev_wd_info[2])
                                week_a_day = "{}{}的{}{}".format(_week_opts[span_idx+2], wd, wd, WEEK_DAYS[day_idx])
                            else:
                                repeat_idx = int(prev_wd_info[2])
                                week_a_day = "{}一个{}{}".format(_week_opts[repeat_idx+2], wd, WEEK_DAYS[day_idx])
                            return self._get_date_with_span_from_weekday(query_type, week_a_day), 0
                        elif ds_type == 2 and prev_wd_info[1] == 'SPAN':
                            day_idx = int(prev_wd_info[3])
                            week_a_day = "{}{}的{}{}".format(_week_opts[span_idx+2], wd, wd, WEEK_DAYS[day_idx])
                            return self._get_date_with_span_from_weekday(query_type, week_a_day), 0
                    else:
                        gong_nong = CalPats.for_gongli_or_nongli(in_question)
                        if gong_nong is not None:
                            tmp_in = in_question + ' _cr_ 农 历 日 期' if gong_nong == 'NONGLI' else in_question
                            extra = "跟前面的历法相同啊，那我就再说一遍吧：" if gong_nong == prev_wd_info[0] else ""
                            day_idx = int(prev_wd_info[3])
                            if prev_wd_info[1] == 'SPAN':
                                span_idx = int(prev_wd_info[2])
                                week_a_day = "{}周的周{}".format(_week_opts[span_idx+2], WEEK_DAYS[day_idx])
                            else:
                                repeat_idx = int(prev_wd_info[2])
                                week_a_day = "{}一个周{}".format(extra, _week_opts[repeat_idx+2], WEEK_DAYS[day_idx])
                            return "{}{}".format(extra, self._get_date_with_span_from_weekday(query_type, week_a_day)), 0

            in_question = self._remove_prefix_for_ner_predictor(in_question)
            ni_ents = self.ner_predictor.predict_niobj(in_question)
            if ni_ents:
                if self.chat_session.last_topic and self.chat_session.last_topic.title == 'HUGO_QUERY':
                    cat_key, hugo_name = KnowledgeBase.find_matched_hugo_niobj(ni_ents,
                                                                               self.chat_session.last_topic.value)
                    if cat_key and hugo_name:
                        return self._get_huge_obj_data_from_keys(cat_key, hugo_name, from_most=True), 0
                if self.chat_session.categ_ref_text == '{}年龄'.format(self.chat_session.bot_name) and \
                        CalPats.fullmatch_cal_year_cat(ni_ents):
                    return self._get_xgg_age_ni_ent_cal_year(ni_ents), 0

                if self.chat_session.niobj_topic == self.chat_session.niobj_topic_opts['1']:
                    ret_text, real_info = self._get_celeb_info_from_kbase(ni_ents)
                    self.chat_session.keep_niobj_topic = True
                elif self.chat_session.niobj_topic == self.chat_session.niobj_topic_opts['2']:
                    ret_text, real_info = self._get_cap_from_nation_from_kbase(ni_ents)
                    self.chat_session.keep_niobj_topic = True
                elif self.chat_session.niobj_topic == self.chat_session.niobj_topic_opts['3']:
                    ret_text, real_info = self._get_nation_from_cap_from_kbase(ni_ents)
                    self.chat_session.keep_niobj_topic = True
                # elif self.chat_session.niobj_topic == self.chat_session.niobj_topic_opts['4']:
                #     ret_text, real_info = self._get_meaning_from_entry_from_kbase(ni_ents)
                #     self.chat_session.keep_niobj_topic = True
                elif self.chat_session.niobj_topic == self.chat_session.niobj_topic_opts['5']:
                    ret_text, real_info = self._get_whatis_from_entry_from_kbase(ni_ents)
                    self.chat_session.keep_niobj_topic = True
                else:
                    out_list = [
                        "{}就是{}啊，哈哈。".format(ni_ents, ni_ents),
                        "{}呢？我也想问呐，哈哈{}".format(ni_ents, '{捂脸}'),
                        "{}怎么了呀？我不明白呢。".format(ni_ents),
                        "那你来说说，{}怎么了啊？".format(ni_ents),
                        "{}啊，我不清楚你什么意思呢。".format(ni_ents),
                    ]
                    ret_text = random.choice(out_list)
                    real_info = False

                    if self.chat_session.has_context():
                        self.chat_session.try_categ_first = True
                        self.chat_session.keep_context = True
                        need_retry = 1

                # 1) No context or
                # 2) Did not find any description from the entry based on the context
                if not real_info:
                    opt_text, get_info = self._get_niobj_info_from_entry_from_kbase(ni_ents)
                    if get_info:
                        ret_text = opt_text

        if ret_text:
            return ret_text, need_retry
        else:
            return "晕，我没看懂。可以给解释一下吗？", 0

    def _get_niobj_info_from_entry_from_kbase(self, niobj):
        niobj_desc = self.knowledge_base.niobj_info_dict.get(niobj.lower())
        if niobj_desc:  # find info
            return niobj_desc.strip(), True
        else:
            return niobj_desc, False

    def _get_celeb_attr_info(self, attr_key, celeb_name, in_question):
        ret_text = ''
        if attr_key and celeb_name and in_question:
            in_question = self._remove_prefix_for_ner_predictor(in_question)
            # if attr_key in CELEB_WORK_MAP:
            #     celeb_name = self.ner_predictor.predict_celebname_4work(in_question)
            # else:
            #     celeb_name = self.ner_predictor.predict_celebname_4attr(in_question)
            if attr_key and celeb_name and attr_key in CELEB_ATTR_CONVERT_MAP:
                ret_text = self._get_celeb_attr_text(celeb_name, attr_key, in_question)
            if not ret_text:
                ret_text = "咱又不是狗仔队的，干嘛要关注名人的点点滴滴呢？这我真不擅长啊。"

        return ret_text

    def _get_celeb_attr_text(self, celeb_name, attr_key, in_question=None, from_thing=False):
        celeb_name = KnowledgeBase.remove_celeb_title(celeb_name)
        if attr_key in CELEB_ATTR_MAP:
            ret_text = self.knowledge_base.get_celeb_attr_info_text(celeb_name, attr_key, in_question)
            if not ret_text and not from_thing and attr_key in ['高度', '身高', '重量', '体重', '年龄']:
                # handle cases of prediction error
                if self.knowledge_base.is_a_thing_name(celeb_name):
                    if attr_key in ['高度', '身高']:
                        thing_key = '高度'
                    elif attr_key in ['重量', '体重']:
                        thing_key = '重量'
                    else:
                        thing_key = '年龄'
                    ret_text = self._get_huge_obj_data_from_keys(thing_key, celeb_name, from_most=False,
                                                                 from_person=True)
        elif attr_key in CELEB_RELATION_MAP:
            ret_text, new_key = self.knowledge_base.get_celeb_relation_info_text(
                celeb_name, attr_key, in_question)
            if new_key:
                attr_key = new_key
        else:  # attr_key in CELEB_WORK_MAP
            ret_text = self.knowledge_base.get_celeb_work_info_text(celeb_name, attr_key, in_question)

        new_name = self.knowledge_base.celeb_alias_dict.get(celeb_name) or celeb_name
        if new_name in self.knowledge_base.celeb_female_set:
            self.chat_session.set_pronoun('ta2', new_name)
        elif new_name in self.knowledge_base.celeb_male_set:
            self.chat_session.set_pronoun('ta1', new_name)
        else:  # set both ta1 and ta2
            self.chat_session.set_pronoun('ta1', new_name)
            self.chat_session.set_pronoun('ta2', new_name)

        key_categ = CELEB_ATTR_CATEG_DICT.get(CELEB_ATTR_CONVERT_MAP.get(attr_key))
        if key_categ:
            self.chat_session.set_categ(key_categ)
            if not ret_text:
                if attr_key in CELEB_WORK_MAP:
                    ret_text = random.choice([
                        "{}读书少，没有这些高大上的知识呢。".format(self.chat_session.bot_name),
                        "矮油，你这真是难为我了，我哪记得名人的这些作品呀。",
                        "干嘛要关注名人的作品细节呢？这我真不擅长啊。", ])
                else:
                    if key_categ[0] in ['是', '叫']:
                        key_part = "{}{}".format(new_name, key_categ)
                    else:
                        key_part = "{}的{}".format(new_name, key_categ)
                    ret_text = random.choice([
                        "{}读书少，不知道{}呢。".format(self.chat_session.bot_name, key_part),
                        "矮油，你这真是难为我了，我哪记得{}呀。".format(key_part),
                        "晕，这不是难为我嘛，我哪知道{}呀。".format(key_part),
                        "干嘛要关注名人的隐私呢？这我真不擅长啊。", ])
        return ret_text

    def _get_celeb_other_relation(self, in_question):
        ret_text = ''
        if in_question:
            in_question = self._remove_prefix_for_ner_predictor(in_question)
            celeb_name = self.ner_predictor.predict_celebname_4attr(in_question)
            if celeb_name:
                attr_key = '其他'
                celeb_name = KnowledgeBase.remove_celeb_title(celeb_name)
                ret_text, new_key = self.knowledge_base.get_celeb_relation_info_text(celeb_name, attr_key, in_question)
                if new_key:
                    attr_key = new_key

                new_name = self.knowledge_base.celeb_alias_dict.get(celeb_name) or celeb_name
                if new_name in self.knowledge_base.celeb_female_set:
                    self.chat_session.set_pronoun('ta2', new_name)
                elif new_name in self.knowledge_base.celeb_male_set:
                    self.chat_session.set_pronoun('ta1', new_name)
                else:  # set both ta1 and ta2
                    self.chat_session.set_pronoun('ta1', new_name)
                    self.chat_session.set_pronoun('ta2', new_name)

                key_categ = CELEB_ATTR_CATEG_DICT.get(CELEB_ATTR_CONVERT_MAP.get(attr_key))
                if key_categ:
                    self.chat_session.set_categ(key_categ)
            if not ret_text:
                ret_text = random.choice([
                    "干嘛要这么关注名人的社会关系呢？这我真不擅长啊。",
                    "咱又不是狗仔队的，干嘛要关注名人的隐私呢？这我真不擅长啊。"])

        return ret_text

    def _get_work_attr(self, attr_key, work_name, in_question):
        main_text = ''
        if attr_key and in_question:
            in_question = self._remove_prefix_for_ner_predictor(in_question)
            if attr_key in ['作者']:
                last_topic = self.chat_session.last_topic
                if last_topic and self.chat_session.context_ref_text:
                    if last_topic.title in ['POEM']:
                        pronoun_ref = self.chat_session.get_pronoun('ta3')
                        dense_quest = to_dense_text(in_question)
                        if dense_quest.endswith(to_dense_text('_cr_ {}'.format(pronoun_ref))):
                            ret_text, _ = self._get_poem_info_details(last_topic, 'writer')
                            if ret_text:
                                return ret_text

            # work_name = self.ner_predictor.predict_workname_4attr(in_question)
            if attr_key and work_name and attr_key in self.knowledge_base.work_attr_data:
                new_work, ret_text = self.knowledge_base.get_work_attr_info(work_name, attr_key)
                self.chat_session.set_pronoun('ta3', new_work)  # set pronoun context
                self.chat_session.set_categ("{}是谁".format(attr_key))  # set categ context
                if ret_text:
                    ft_index = ret_text.find('_fc_t')
                    if ft_index > 0:
                        rest_text = ret_text[ft_index:].strip()
                        main_text = ret_text[:ft_index].strip()
                        for parts in rest_text.split('_fc_'):
                            clean_parts = parts.strip()
                            if clean_parts:
                                lr_data = clean_parts[:-1].split(' [')
                                left, right = lr_data[0].strip(), lr_data[1].strip()
                                print("left = #{}#, right = #{}#".format(left, right))
                                self.chat_session.set_pronoun(left, right)  # set pronoun context

        if not main_text:
            main_text = "矮油，{}读书少，对这些文学或艺术作品真的了解不多呢。".format(self.chat_session.bot_name)

        return main_text

    def _get_org_leader(self, in_question):
        ret_text = ''
        if in_question:
            in_question = self._remove_prefix_for_ner_predictor(in_question)
            org, pos = self.ner_predictor.predict_org_leader_entries(in_question)
            if org and pos:
                ret_text, new_org = self.knowledge_base.get_org_leader_info(org, pos)
                self.chat_session.set_categ("{}是谁".format(pos))
                if new_org:
                    self.chat_session.set_pronoun('ta3', new_org)
                else:
                    self.chat_session.set_pronoun('ta3', org)

            if not ret_text:
                ret_text = "矮油，我知道这些人在一定范围内很有影响力，但咱何必这么关注他们呢？"

        return ret_text

    def _get_animal_attr_info(self, attr_key, in_question):
        ret_text = ''
        if attr_key and in_question:
            in_question = self._remove_prefix_for_ner_predictor(in_question)
            animal_name = self.ner_predictor.predict_animal_name(in_question)
            if attr_key and animal_name and attr_key in ANIMAL_ATTR_MAP:
                ret_text = self.knowledge_base.get_animal_attr_info_text(animal_name, attr_key)
                if ret_text:
                    fun_idx = ret_text.find('_func_')
                    if fun_idx > 0:
                        fun_txt = ret_text[fun_idx:]
                        ret_text = ret_text[:fun_idx].strip()
                        if fun_txt == '_func_ta1':
                            self.chat_session.set_pronoun('ta1', animal_name)
                        elif fun_txt == '_func_ta1_tamen1':
                            self.chat_session.set_pronoun('ta1', animal_name)
                            self.chat_session.set_pronoun('tamen1', animal_name)
                        elif fun_txt == '_func_ta3_tamen3':
                            self.chat_session.set_pronoun('ta3', animal_name)
                            self.chat_session.set_pronoun('tamen3', animal_name)

                self.chat_session.set_categ(ANIMAL_ATTR_MAP[attr_key])

            if not ret_text:
                ret_text = "矮油，我没学过生物课，真没有那么多动植物的知识呢。"

        return ret_text

    def scan_att_qry_block(self, in_question, sentence):
        sentence = to_dense_text(sentence)
        ind_list = [(m.start(0), m.end(0)) for m in re.finditer(ATT_QRY_CAT, sentence)]
        if len(ind_list) == 0:
            return sentence

        prev_end_idx = -1
        out_sent = ''
        for idx, (start, end) in enumerate(ind_list):
            blk_txt = sentence[start:end].replace("<att_qry>", "").replace("</att_qry>", "").strip()
            blk_txt = blk_txt.replace(':', '：')
            att_data = blk_txt.split('：')
            att_cate, att_details = att_data[0], att_data[1].split('；')

            out_text = ''
            if att_details and len(att_details) == 2:
                attr_key = att_details[0]
                if att_cate == '名人个体':
                    out_text = self._get_celeb_attr_info(attr_key, att_details[1], in_question)
                elif att_cate == '名人关系':
                    if attr_key in CELEB_MAIN_RELATION_SET:
                        out_text = self._get_celeb_attr_info(attr_key, att_details[1], in_question)
                    elif attr_key in CELEB_RELATION_MAP:
                        out_text = self._get_celeb_other_relation(in_question)
                elif att_cate == '名人成就':
                    out_text = self._get_celeb_attr_info(attr_key, att_details[1], in_question)
                elif att_cate == '作品属性':
                    out_text = self._get_work_attr(attr_key, att_details[1], in_question)
                elif att_cate == '巨物度量':
                    out_text = self._get_huge_obj_data(attr_key, in_question)
                elif att_cate == '国家地区':
                    out_text = self._get_area_attr(attr_key, in_question)
                elif att_cate == '机构领导':
                    out_text = self._get_org_leader(in_question)
                elif att_cate == '动植物':
                    if attr_key == '归类':
                        attr_key = '动植物'
                    out_text = self._get_animal_attr_info(attr_key, in_question)

            if out_text:
                if prev_end_idx > 0:
                    out_sent += sentence[prev_end_idx:start]
                else:
                    out_sent += sentence[:start]
                out_sent += out_text
                prev_end_idx = end
            else:
                if att_cate == '名人个体':
                    raise ValueError("抱歉，我读书少，对名人了解也不多，不知如何回答您的问题呢！")
                elif att_cate == '名人关系':
                    raise ValueError("矮油，我孤陋寡闻，对名人的家庭和社会关系了解不多，不知道如何回答您的问题呢！")
                elif att_cate == '名人成就':
                    raise ValueError("矮油，我孤陋寡闻，对名人的成就和作品也关注很少，不知道如何回答您的问题呢！")
                elif att_cate == '作品属性':
                    raise ValueError("矮油，我读书少，对小说影视等作品也了解不多，不知道如何回答您的问题呢！")
                elif att_cate == '巨物度量':
                    raise ValueError("晕，怪我读书少，地理和天文等方面的知识更是缺乏，不知道如何回答您的问题呢！")
                elif att_cate == '国家地区':
                    raise ValueError("晕，我地理知识缺乏，对国家和很多地区的细节更是知之甚少，不知道如何回答您的问题呢！")
                elif att_cate == '机构领导':
                    raise ValueError("晕，我读书少，对头头脑脑的事情关注更少，真的不知道如何回答这个问题呢！")
                elif att_cate == '动植物':
                    raise ValueError("晕，怪我读书少，动植物方面的知识更是缺乏，不知道如何回答您的问题呢！")
                else:  # any other cases
                    raise ValueError("晕，我读书少，不知道如何回答这个问题呢！")

        if prev_end_idx > 0:
            out_sent += sentence[prev_end_idx:]
        return out_sent

    def get_hist_event_time(self, sentence, in_question):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        ret_text = ''
        if len(item_list) == 1 and item_list[0]:
            kk = item_list[0].replace(' ', '').strip()
            vv = self.knowledge_base.hist_events_dict.get(kk)
            if vv:
                ret_text = vv

        if ret_text:
            opt_idx = ret_text.find('_opt_')
            if opt_idx > 0:
                if re.search(r'(多少|几)[年月日天]|多久', to_dense_text(in_question)):
                    ret_text = ret_text.replace('_opt_', '')
                    zn_cnt_mat = re.search(r'_zn_cnt_\d+_', ret_text)
                    if zn_cnt_mat:
                        ss, ee = zn_cnt_mat.start(), zn_cnt_mat.end()
                        year_len = dt.datetime.now().year - int(ret_text[ss:ee][8:-1])
                        ret_text = "{}{}{}".format(ret_text[:ss], year_len, ret_text[ee:])
                else:
                    ret_text = ret_text[:opt_idx]
            return ret_text, 0
        else:
            return "{}读书少，真的不记得这些历史事件的时间呢。".format(self.chat_session.bot_name), 0

    def get_sport_event_info(self, sentence, in_question):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        ret_text = ''
        if len(item_list) == 2 and item_list[0] and item_list[1]:
            cat = item_list[0].replace(' ', '').strip()
            typ = item_list[1].replace(' ', '').strip()
            if cat and typ:
                pre_spt_yr, pre_abs_jc, pre_rel_jc = 0, 0, -100
                if self.chat_session.last_topic and self.chat_session.last_topic.title == 'SPORT_EVENT':
                    pre_se_info = self.chat_session.last_topic.value.split("=")
                    pre_typ = pre_se_info[1]
                    pre_spt_yr, pre_abs_jc, pre_rel_jc = int(pre_se_info[2]), int(pre_se_info[3]), int(pre_se_info[4])
                    if typ == '上文':
                        typ = pre_typ
                self.chat_session.set_categ("询问{}信息".format(cat))

                dense_quest = to_dense_text(in_question)
                if typ == '名次':
                    _hanzi_num_chs = "零一二三四五六七八九两十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟"
                    mc_mat = re.search(r'[冠亚季殿]军|第([一二三四五六七八九]|[1-9])(?!({}|[0-9]))'.format(_hanzi_num_chs),
                                       dense_quest)
                    if mc_mat:
                        mc_txt = dense_quest[mc_mat.start():mc_mat.end()]
                        if mc_txt in ['冠军', '亚军', '季军', '殿军']:
                            typ = '第{}名'.format(['冠军', '亚军', '季军', '殿军'].index(mc_txt)+1)
                        else:
                            typ = '第{}名'.format(text2num(mc_txt[1:]))
                # 1）年份；2）绝对的届次；3）相对的届次
                spt_yr, abs_jc, rel_jc = 0, 0, -100
                now = dt.datetime.now()

                year_cat = CalPats.get_cal_year_cat()
                year_mat = re.search(year_cat, in_question)
                if year_mat:
                    cal_year = ''.join(in_question[year_mat.start():year_mat.end()].split())
                    spt_yr, _ = CalPats.parse_cal_year_text(now.year, cal_year)
                if spt_yr == 0:
                    if re.search(r'[首头]一?[届次]', dense_quest):
                        abs_jc = 1
                    elif re.search(r'[这此]一?[届次]|刚结束|[正在]举[行办]', dense_quest):
                        rel_jc = 0
                    elif re.search(r'[上大]上一?[届次]', dense_quest):
                        rel_jc = -2
                    elif re.search(r'[上前]一?[届次]', dense_quest):
                        rel_jc = -1
                    elif re.search(r'[下大]下一?[届次]', dense_quest):
                        rel_jc = 2
                    elif re.search(r'下一?[届次]', dense_quest):
                        rel_jc = 1
                if spt_yr == 0 and rel_jc not in [-2, -1, 0, 1, 2]:
                    _hanzi_1_9 = r'[一二三四五六七八九]'
                    abs_mat = re.search(r'第?([1-9][0-9]?|{0}十{0}?|十?{0})[届次]'.format(_hanzi_1_9), dense_quest)
                    if abs_mat:
                        abs_jc_text = dense_quest[abs_mat.start():abs_mat.end()]
                        abs_jc = text2num(abs_jc_text.replace('第', '')[:-1])
                        if not abs_jc_text.startswith('第') and abs_jc < 10:
                            abs_jc = 0  # give up this match as it may be a mistake
                if spt_yr == 0 and abs_jc == 0 and rel_jc not in [-2, -1, 0, 1, 2]:  # 此问句没有指定届次信息
                    if pre_spt_yr > 0 or pre_abs_jc > 0 or pre_rel_jc in [-2, -1, 0, 1, 2] and ' _cr_ ' in dense_quest:
                        spt_yr, abs_jc, rel_jc = pre_spt_yr, pre_abs_jc, pre_rel_jc
                if spt_yr > 0 or abs_jc > 0 or rel_jc in [-2, -1, 0, 1, 2]:
                    ret_text = self.knowledge_base.get_sport_event_info(cat=cat, typ=typ, spt_yr=spt_yr,
                                                                        abs_jc=abs_jc, rel_jc=rel_jc)
                    topic_val = "{}={}={}={}={}".format(cat, typ, spt_yr, abs_jc, rel_jc)
                    self.chat_session.set_keep_topic(Topic('SPORT_EVENT', topic_val), rounds=1)

        if not ret_text:
            ret_text = "{}读书少，对这些体育赛事真的了解不多呢。".format(self.chat_session.bot_name)

        return ret_text, 0

    """
    # Rule 6.3: 中文翻译成英文
    """
    def get_en_trans_from_cn(self, sentence, in_question):
        ret_text = ''
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if in_question and len(item_list) == 1:
            in_question = self._remove_prefix_for_ner_predictor(in_question)
            tracn_et = self.ner_predictor.predict_cn_en_trans(in_question)
            if tracn_et:
                print("tracn_et = #{}#".format(tracn_et))
                en_desc = self.knowledge_base.cn_en_trans_dict.get(tracn_et)
                if en_desc:
                    if en_desc.startswith('=='):
                        ret_text = en_desc[2:]
                    else:
                        self.chat_session.set_context_ref("用英语怎么说")
                        extra = random.choice(["这我知道，", "这我学过，", "这我记得，", "", ""])
                        if en_desc.startswith('K='):
                            ret_text = "_NO_CUT_{}{}译成英语是：{}".format(extra, tracn_et, en_desc[2:])
                        else:
                            ret_text = random.choice([
                                "_NO_CUT_{}{}译成英语是：{}。".format(extra, tracn_et, en_desc),
                                "_NO_CUT_{}{}用英语说就是{}呀。".format(extra, tracn_et, en_desc), ])
                else:
                    self.chat_session.set_context_ref("用英语怎么说")
                    en_desc = KnowledgeBase.translate_to_english(tracn_et)
                    if en_desc:
                        ret_text = "_NO_CUT_“{}”译成英语是：{}（结果来自谷歌翻译非官方版）".format(tracn_et, en_desc)

        if not ret_text:
            self.chat_session.set_context_ref("用英语怎么说")
            ret_text = random.choice([
                "你还是别考我英语了，我英语很差的。",
                "我英语水平很差的，你还是别考我这些了。",
                "抱歉啊，我不会。对了，上班时间老板不让我们讲英语的。"]
            )
        return ret_text, 0

    """
    # Rule 6.4: CHENGYU JIELONG (成语接龙: based on context cat 5: context prefix)
    """
    def _start_chengyu_jielong(self):
        self.chat_session.set_context_prefix("_st_jielong 接龙成语：")
        self.chat_session.clear_chengyu_jl_cache()

        ret_text = ''
        if not self.chat_session.chengyu_jl_rule_explained:
            ret_text = "成语接龙的基本规则是这样的：_np_" \
                       "1、前后两句相接成语的连接字必须是同一汉字：比如人山人海 =》海纳百川 =》川流不息；" \
                       "2、成语必须由四个字组成，并且是一般成语词典上能查到的成语；" \
                       "3、自己或对方使用过的成语不得再次出现。_nr_"
            self.chat_session.chengyu_jl_rule_explained = True
        ret_text += '你先来，说成语吧。'
        return ret_text

    def _get_init_jielong_chengyu(self):
        init_chengyu = random.choice(self.init_chengyu_list)

        self.chat_session.chengyu_jl_list.append(init_chengyu)
        # balance the count deducted in the forgetting-context function
        self.chat_session.context_prefix['count'] += 1
        out_list = [
            "{}，该你了。".format(init_chengyu),
            "{}，你请吧。".format(init_chengyu),
            "{}，尾字是{}。".format(init_chengyu, init_chengyu[-1]),
            "{}，尾字是{}，你请吧。".format(init_chengyu, init_chengyu[-1]),
            "{}，尾字是{}，轮到你了。".format(init_chengyu, init_chengyu[-1]),
        ]
        ret_text = random.choice(out_list)
        return ret_text

    def _get_jielong_chengyu(self, in_question):
        ret_text = ''
        if in_question and in_question.startswith('_st_jielong'):
            c_index = in_question.find('：')
            if c_index > 1 and c_index + 1 < len(in_question):
                in_question = in_question[c_index + 1:].strip()
                if in_question:
                    chengyu = self.ner_predictor.predict_jielong_chengyu(in_question)  # 用户所接成语
                    if chengyu and self.chat_session.chengyu_jl_list:
                        last_chengyu = self.chat_session.chengyu_jl_list[-1]
                        if last_chengyu[-1] != chengyu[0]:
                            self.chat_session.context_prefix['count'] += 1
                            out_list = [
                                "矮油，你不认真。你说的{}跟前面的{}也接不上啊。你重来一次，要以“{}”字开头哟。".format(
                                    chengyu, last_chengyu, last_chengyu[-1]),
                                "晕，你说的{}跟前面的{}也接不上啊。要不你再来一次吧，不要忘了以“{}”字开头哟。".format(
                                    chengyu, last_chengyu, last_chengyu[-1]),
                            ]
                            ret_text = random.choice(out_list)
                    if chengyu and ret_text == '':
                        self.chat_session.chengyu_jl_list.append(chengyu)
                        rep_chengyu = self._get_jielong_chengyu_by_seed_char(chengyu[-1],
                                                                             self.chat_session.chengyu_jl_list)
                        if rep_chengyu:
                            self.chat_session.chengyu_jl_list.append(rep_chengyu)
                            # balance the count deducted in the forgetting-context function
                            self.chat_session.context_prefix['count'] += 1
                            out_list = [
                                "{}".format(rep_chengyu),
                                "{}".format(rep_chengyu),  # Make this appear more often
                                "{}，该你了。".format(rep_chengyu),
                                "这个简单，{}。".format(rep_chengyu),
                                "{}，你来接，尾字是{}。".format(rep_chengyu, rep_chengyu[-1]),
                                "这个我会，{}，哈哈。轮到你了。".format(rep_chengyu),
                            ]
                            ret_text = random.choice(out_list)
                        else:
                            out_list = [
                                "晕，我找不到以{}字开头的成语。你真厉害，我认输啦。".format(chengyu[-1]),
                                "不会了，我找不到以{}字开头的成语，我认输。还是你牛啊。".format(chengyu[-1]),
                                "不行，我找不到以{}字开头的成语，不知道该什么接了，我认输。".format(chengyu[-1])
                            ]
                            ret_text = random.choice(out_list)
                            self._end_chengyu_jielong()  # End from XGG's side

        if not ret_text:
            ret_text = "不会了，我实在不知道该什么接了，我认输。"
            self._end_chengyu_jielong()  # End from XGG's side
        return ret_text

    def _get_jielong_chengyu_for_user(self):
        ret_text = ''
        if self.chat_session.chengyu_jl_list:
            last_chengyu = self.chat_session.chengyu_jl_list[-1]
            rep_chengyu = self._get_jielong_chengyu_by_seed_char(last_chengyu[-1],
                                                                 self.chat_session.chengyu_jl_list)
            if rep_chengyu:
                self.chat_session.chengyu_jl_list.append(rep_chengyu)
                # balance the count deducted in the forgetting-context function
                self.chat_session.context_prefix['count'] += 1
                out_list = [
                    "我想出来啦：{}，你来接吧。".format(rep_chengyu),
                    "{}，这回又该你了。".format(rep_chengyu),
                    "我找到了：{}，你可以接了，尾字是{}。".format(rep_chengyu, rep_chengyu[-1]),
                ]
                ret_text = random.choice(out_list)

        if not ret_text:
            out_list = ["我晕，我不会。算了，这局结束了。",
                        "我也想不出，看来只有放弃了。",
                        "我想了一会，不知道怎么接。看来只能放弃了。"]
            ret_text = random.choice(out_list)
            self._end_chengyu_jielong()  # End from XGG's side
        return ret_text

    def _get_jielong_chengyu_by_seed_char(self, seed_char, used_list):
        option_list = []
        val_list = self.knowledge_base.chengyu_dict.get(seed_char)
        if val_list:
            for val_item in val_list:
                rep_chengyu = seed_char + val_item
                if rep_chengyu not in used_list:
                    option_list.append(rep_chengyu)
                    if len(option_list) >= 3:
                        break

        if option_list:
            return random.choice(option_list)
        return None

    def _end_chengyu_jielong(self):
        self.chat_session.clear_context_prefix()
        self.chat_session.clear_chengyu_jl_cache()

        # Let the system know that we have just ended CHENGYU JIELONG
        self.chat_session.set_keep_topic(Topic('CHENGYU_JIELONG', ''))
        self.chat_session.set_categ(self.chat_session.get_topic_categ('CHENGYU_JIELONG'))
        return ''

    """
    # Rule 6.5: Mimic role (based on context cat 5: context prefix)
    """
    def _start_mimic_role(self, role):
        self.chat_session.set_context_prefix("_st_mimic {}：".format(role))
        return ''

    def _check_mimic_step(self):
        if self.chat_session.context_prefix['count'] <= 1:
            self.chat_session.clear_context_prefix()
            return "{}说：不行了，这是最后一个问题了。模仿秀必须结束了。".format(self.chat_session.bot_name)
        return ''

    def _end_mimic_role(self):
        self.chat_session.clear_context_prefix()
        return ''

    """
    # Rule 6.x: scan_msc_ctl_blk
    """
    def scan_msc_ctl_block(self, in_question, sentence):
        sentence = to_dense_text(sentence)
        ind_list = [(m.start(0), m.end(0)) for m in re.finditer(MSC_CTL_CAT, sentence)]
        if len(ind_list) == 0:
            return sentence

        prev_end_idx = -1
        out_sent = ''
        for idx, (start, end) in enumerate(ind_list):
            blk_txt = sentence[start:end].replace("<msc_ctl>", "").replace("</msc_ctl>", "").strip()
            blk_txt = blk_txt.replace(':', '：')
            ctl_data = blk_txt.split('：')
            ctl_type, ctl_details = ctl_data[0], ctl_data[1].split('；')

            out_text, status_id = '', 0
            if ctl_type in ['简繁体', '簡繁體'] and ctl_details and len(ctl_details) == 1:
                if ctl_details[0] in ['简体', '簡體']:
                    self.chat_session.use_simplified = True
                elif ctl_details[0] in ['繁体', '繁體']:
                    self.chat_session.use_simplified = False
                elif ctl_details[0] in ['原样', '原樣', '不变', '不變']:
                    self.chat_session.tradit_convert = False
                else:
                    pass
                status_id = 1
            elif ctl_type in ['长下文', '长上下文'] and ctl_details and len(ctl_details) == 1:
                self.chat_session.add_context_topic(ctl_details[0])
                status_id = 1
            elif ctl_type == '待完成' and ctl_details and len(ctl_details) == 1:
                if ctl_details[0] == '智能家居':
                    out_text = ' _nr_ （系统信息：暂无智能家居接口。）'
                elif ctl_details[0] == '增值业务':
                    out_text = ' _nr_ （系统信息：暂无外部或线下服务接口。）'
                status_id = 1
            elif ctl_type in ['成语接龙', '接龙成语'] and ctl_details and len(ctl_details) == 1:
                if ctl_details[0] == '开局':
                    out_text = self._start_chengyu_jielong()
                elif ctl_details[0] == '出题':
                    out_text = self._get_init_jielong_chengyu()
                elif ctl_details[0] in ['代答', '替答']:
                    out_text = self._get_jielong_chengyu_for_user()
                elif ctl_details[0] == '结束':
                    out_text = self._end_chengyu_jielong()
                    status_id = 1
                else:
                    out_text = self._get_jielong_chengyu(in_question)
            elif ctl_type in ['模仿', '模仿秀'] and ctl_details and len(ctl_details) in [1, 2]:
                if ctl_details[0] == '开局' and len(ctl_details) == 2:
                    out_text = self._start_mimic_role(ctl_details[1])
                elif ctl_details[0] == '单步' and len(ctl_details) == 1:
                    out_text = self._check_mimic_step()
                elif ctl_details[0] == '结束' and len(ctl_details) == 1:
                    out_text = self._end_mimic_role()
                status_id = 1

            if out_text or status_id == 1:
                if prev_end_idx > 0:
                    out_sent += sentence[prev_end_idx:start]
                else:
                    out_sent += sentence[:start]
                out_sent += out_text
                prev_end_idx = end
            else:
                if ctl_type in ['成语接龙', '接龙成语']:
                    raise ValueError("抱歉，成语接龙我玩得还不精，我认输！")
                elif ctl_type in ['模仿', '模仿秀']:
                    raise ValueError("抱歉，模仿秀我玩得还不精，我认输！")
                else:  # any other cases
                    raise ValueError("这个游戏我还不在行，我认输！")

        if prev_end_idx > 0:
            out_sent += sentence[prev_end_idx:]
        return out_sent

    """
    # Rule 7.1: Pending answer (based on context cat 6.1: pending answer)
    """
    # def set_pending_ans(self, sentence, para1=None):
    #     item_list = re.findall(r'\[(.*?)\]', sentence)
    #     if para1:
    #         if len(item_list) == 1 and item_list[0]:
    #             item0 = item_list[0].strip()
    #             # if para1 == 'tru_ser':
    #             #     self.chat_session.set_pending_ans('true', item0)
    #             #     self.chat_session.set_pending_ans('serious', item0)
    #             #     self.chat_session.set_context_ref('你有话不愿直说')
    #             # if para1 == 'explain':
    #             #     self.chat_session.set_pending_ans('explain', item0)
    #             #     self.chat_session.set_pending_ans('why', item0)
    #             # elif para1 != 'yes_no' and para1 != 'tru_fal':
    #             if para1 != 'yes_no':
    #                 self.chat_session.set_pending_ans(para1, item0)
    #         elif len(item_list) == 2 and item_list[0] and item_list[1]:
    #             if para1 == 'yes_no':
    #                 self.chat_session.set_pending_ans('yes', item_list[0].strip())
    #                 self.chat_session.set_pending_ans('no', item_list[1].strip())
    #             # elif para1 == 'tru_fal':
    #             #     self.chat_session.set_pending_ans('true', item_list[0].strip())
    #             #     self.chat_session.set_pending_ans('false', item_list[1].strip())
    #     return '', 0

    def propose_pending_quest(self, sentence, para1=None):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if para1 == 'yes_no' and len(item_list) == 2 and item_list[0] and item_list[1]:
            self.chat_session.propose_pending_yes_quest(item_list[0].strip())
            self.chat_session.set_pending_ans('no', item_list[1].strip())
        return '', 0

    def get_pending_ans(self, sentence, key):
        need_retry = 0  # default to be 0, which is no retry
        # Try to use stored information
        ret_text = self.chat_session.get_pending_ans(key)
        if ret_text:  # Found the pending answer
            # if key in ['true', 'serious'] and self.chat_session.context_ref_text == '你有话不愿直说':
            #     self.chat_session.context_ref_text = None
            # Assume we do not have a chance to use the context ref, therefore, keep it if any
            self.chat_session.keep_context = True
        else:
            # Retrieve the default response in the prediction, if available
            item_list = re.findall(r'\[(.*?)\]', sentence)
            if len(item_list) == 1 and item_list[0]:
                ret_text = item_list[0].strip()

            if self.chat_session.has_context():
                # Did not find the pending answer, and will try the context ref since it is available
                # print("coming here: No pending answer, but context ref is available.")
                self.chat_session.keep_context = True
                need_retry = 1

        if not ret_text:  # No any useful information available
            ret_text = "我最好还是不要乱说。"
        return self._output_sentence_with_inner_function_executed(ret_text), need_retry

    def _check_ret_text_for_keep_niobj_topic(self, ret_text):
        if ret_text.strip().endswith('_func_keep_niobj_topic_as_celeb'):
            ret_text = ret_text.replace('_func_keep_niobj_topic_as_celeb', '').strip()
            # so that in the following conversion, if the user asks something like: “龚玥呢？”,
            # it knows how to answer
            self.chat_session.niobj_topic = self.chat_session.niobj_topic_opts['1']
            self.chat_session.keep_niobj_topic = True
        elif ret_text.strip().endswith('_func_keep_niobj_topic_as_whatis'):
            ret_text = ret_text.replace('_func_keep_niobj_topic_as_whatis', '').strip()
            self.chat_session.niobj_topic = self.chat_session.niobj_topic_opts['5']
            self.chat_session.keep_niobj_topic = True
        return ret_text

    def get_pending_ans_with_base(self, sentence, key):
        need_retry = 0  # default to be 0, which is no retry
        # Try to use stored information
        if key == 'yes' and self.chat_session.proposed_pending_yes_quest:  # proposed pending question is available
            self.chat_session.clear_pending_ans_yes_no_context()  # ret_text below will always be None
            need_retry = 4

        ret_text = self.chat_session.get_pending_ans(key)
        new_base = None
        if ret_text:  # Found the pending answer
            # Assume we do not have a chance to use the context ref, therefore, keep it if any
            self.chat_session.keep_context = True
            if key == 'no':
                ret_text = self._check_ret_text_for_keep_niobj_topic(ret_text)
        else:
            # Retrieve the default response in the prediction, if available
            item_list = re.findall(r'\[(.*?)\]', sentence)
            if len(item_list) == 2 and item_list[0] and item_list[1]:
                ret_text = item_list[0].replace(' ', '')
                new_base = to_dense_text(item_list[1])

            if need_retry == 0:
                if self.chat_session.has_context():
                    # Did not find the pending answer, and will try the context ref since it is available
                    # print("coming here: No pending answer, but context ref is available.")
                    if new_base:
                        self.chat_session.context_ref_base = new_base
                    self.chat_session.keep_context = True
                    need_retry = 1
                elif key == 'who' and self.chat_session.check_and_set_pronoun_context(['ta1', 'ta2']):
                    if new_base:
                        self.chat_session.context_ref_base = new_base
                    need_retry = 2

            if ret_text and ret_text.strip() == '_infunc_get_context_topic':
                # default response is used, since no context available
                # extra = self.chat_session.get_context_topic()
                # if extra:
                #     return extra, need_retry
                # else:
                #     return "对了，在忙什么呢？", need_retry
                return ret_text, need_retry

        if not ret_text:  # No any useful information available
            ret_text = "我觉得还是不要乱说的好。"
        return self._output_sentence_with_inner_function_executed(ret_text), need_retry

    def get_pending_ans_with_extra_output(self, sentence, key):
        # extra_output will be appended in the following cases:
        # 1. default output + extra_output
        # 2. all stored output (including yes but no proposed_pending_yes_quest) + extra_output
        # 3. when need_retry == 4, after retry, its output + extra_output
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 2 and item_list[0] and item_list[1]:
            def_output = item_list[0].replace(' ', '')  # default output for the first part only
            extra_output = item_list[1].replace(' ', '')
        else:
            return "我觉得还是不要乱说的好。", 0

        need_retry = 0  # default to be 0, which is no retry
        # Try to use stored information
        if key == 'yes' and self.chat_session.proposed_pending_yes_quest:  # proposed pending question is available
            self.chat_session.clear_pending_ans_yes_no_context()  # ret_text below will always be None
            if extra_output:  # This is cached for the second round only
                # will be concatenated to the output if it is from the retry output of proposed_pending_yes_quest
                self.chat_session.extra_output_cached = extra_output
            need_retry = 4

        ret_text = self.chat_session.get_pending_ans(key)
        if ret_text:  # Found the pending answer
            # Assume we do not have a chance to use the context ref, therefore, keep it if any
            self.chat_session.keep_context = True
            if key == 'no':
                ret_text = self._check_ret_text_for_keep_niobj_topic(ret_text)
        else:
            # output the default response in the prediction, if available
            ret_text = def_output
            if need_retry == 0 and self.chat_session.has_context():
                # Did not find the pending answer, and will try the context ref since it is available
                # print("coming here: No pending answer, but context ref is available.")
                self.chat_session.keep_context = True
                need_retry = 1

        if not ret_text:  # No any useful information available
            ret_text = "我觉得还是不要乱说的好。"
        elif extra_output:
            # all first round return already includes extra_output, including the return of failure retry,
            # which is based on the default response
            ret_text += extra_output
        return self._output_sentence_with_inner_function_executed(ret_text), need_retry

    """
    # Rule 7.2: Missing answer (based on context cat 6.2: missing answer)
    """
    # @staticmethod
    # def _convert_missing_ans_keys(old_key):
    #     key = old_key
    #     if key == '比如':
    #         key = '比如|具体|详细'
    #     elif key == '出处':
    #         key = '出处|谁写的|谁说的'
    #     elif key == '谁写的':
    #         key = '谁写的|出处'
    #     elif key == '谁说的':
    #         key = '谁说的|出处'
    #     return key

    # def _set_mentioned_lyric(self, val_text):
    #     song_mat = re.search(r'《.*》', val_text)
    #     if song_mat:
    #         song_name = to_dense_text(val_text[song_mat.start()+1:song_mat.end()-1])
    #         lyric_id = self.knowledge_base.get_lyric_id_by_title(song_name)
    #         if lyric_id > 0:
    #             self.chat_session.set_keep_topic(Topic('LYRIC', "{}={}".format(lyric_id, song_name)))
    #             # Therefore, if this answer needs to set other context_ref, make this method tag the last one,
    #             # which will be replaced by the context_ref manually designed
    #             self.chat_session.set_context_ref('{}提及一句歌词'.format(self.chat_session.bot_name))

    # def set_missing_ans(self, sentence):
    #     item_list = re.findall(r'\[(.*?)\]', sentence)
    #     if len(item_list) == 2 and item_list[0] and item_list[1]:
    #         key = item_list[0].replace(' ', '').strip()
    #         val = item_list[1].strip()
    #         if key == '谁唱的':
    #             self._set_mentioned_lyric(val)
    #         key = self._convert_missing_ans_keys(key)
    #         if key.find('|') > 0:
    #             keys = key.split('|')
    #             for k in keys:
    #                 self.chat_session.set_missing_ans(k.strip(), val)
    #         else:
    #             self.chat_session.set_missing_ans(key, val)
    #     return '', 0

    # def set_missing_ans_with_cref(self, sentence):
    #     item_list = re.findall(r'\[(.*?)\]', sentence)
    #     if len(item_list) == 3 and item_list[0] and item_list[1] and item_list[2]:
    #         key = item_list[0].replace(' ', '').strip()
    #         val = item_list[1].strip()
    #         if key == '谁唱的':
    #             self._set_mentioned_lyric(val)
    #         key = self._convert_missing_ans_keys(key)
    #         cref = item_list[2].replace(' ', '').strip()
    #         if key.find('|') > 0:
    #             keys = key.split('|')
    #             for k in keys:
    #                 self.chat_session.set_missing_ans_with_cref(k.strip(), val, cref)
    #         else:
    #             self.chat_session.set_missing_ans_with_cref(key, val, cref)
    #     return '', 0

    # def get_missing_ans(self, sentence):
    #     need_retry = 0
    #     ret_text = None
    #
    #     item_list = re.findall(r'\[(.*?)\]', sentence)
    #     if len(item_list) == 2 and item_list[0] and item_list[1]:
    #         key = item_list[0].replace(' ', '').strip()
    #
    #         # Try to use stored information
    #         ret_text = self.chat_session.get_missing_ans(key)
    #         if ret_text:  # Found the missing answer
    #             # Assume we do not have a chance to use the context ref, therefore, keep it if any
    #             if key == '谁唱的':
    #                 song_mat = re.search(r'《.*》', ret_text)
    #                 if song_mat:
    #                     song_name = to_dense_text(ret_text[song_mat.start():song_mat.end()])
    #                     self.chat_session.set_pronoun('ta3', song_name)
    #             self.chat_session.keep_context = True
    #         else:
    #             ret_text = item_list[1].strip()
    #             if self.chat_session.has_context():
    #                 # Did not find the missing answer, and will try the context ref since it is available
    #                 # print("coming here: No missing answer, but context ref is available.")
    #                 self.chat_session.keep_context = True
    #                 need_retry = 1
    #
    #     if not ret_text:  # No any useful information available
    #         ret_text = "我最好还是不要乱说哈。"
    #     return self._output_sentence_with_inner_function_executed(ret_text), need_retry

    """
    # Rule 8.1: Context topic (based on context cat 7.1: User favorite items)
    """
    def set_favor_item(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 2 and item_list[0] and item_list[1]:
            key = item_list[0].replace(' ', '').strip()
            val = item_list[1].strip()
            self.chat_session.set_favor_item(key, val)
            if not re.search(r'(歌手|演员)$', key):
                if key in ['电视剧', '电影', '小说', '诗歌', '歌曲']:
                    self.chat_session.set_pronoun('ta3', "{}《{}》".format(key, val))
                    print("t3 = {}".format("{}《{}》".format(key, val)))
                else:
                    self.chat_session.set_pronoun('ta3', val)
                    print("t3 = {}".format(val))
        return '', 0

    def get_favor_item(self, sentence):
        key, ret_text = None, None

        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            key = item_list[0].replace(' ', '').strip()
            ret_text = self.chat_session.get_favor_item(key)

        if not ret_text:  # No any useful information available
            ret_text = "不过......真不好意思，我还确实猜不出呢。"
        elif key in ['电视剧', '电影', '小说', '诗歌', '歌曲']:
            ret_text = "是《{}》，我一直记在心里呢。".format(ret_text)
        else:
            ret_text = "是{}，我记在心里呢。".format(ret_text)
        return self._output_sentence_with_inner_function_executed(ret_text), 0

    """
    # Rule 8.2: Attr Context (based on context cat 7.2)
    """
    def set_attr_context(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 2:
            attr_key = item_list[0].replace(' ', '').strip()
            attr_val = item_list[1].replace(' ', '').strip()
            time, subject, action, object = self.ner_predictor.predict_attr_ctxt_ents(attr_key)
            if subject == '':
                subject = '我'  # 此类上下问不适用于祈使句，所以可以设定缺省值=我
            elif subject == '他' and self.chat_session.pronoun['ta1']:
                subject = self.chat_session.pronoun['ta1']
            elif subject == '她' and self.chat_session.pronoun['ta2']:
                subject = self.chat_session.pronoun['ta2']
            elif subject == '他们' and self.chat_session.pronoun['tamen1']:
                subject = self.chat_session.pronoun['tamen1']
            elif subject == '她们' and self.chat_session.pronoun['tamen2']:
                subject = self.chat_session.pronoun['tamen2']
            subject = re.sub(r'_kehu_的?', '我', subject)
            new_key = "{}={}={}={}".format(time, subject, action, object)
            print("new_key = {}".format(new_key))
            self.chat_session.set_attr_context(new_key, attr_val)
        return '', 0

    def retry_attr_context(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1:
            attr_key = item_list[0].replace(' ', '').strip()
            time, subject, action, object = self.ner_predictor.predict_attr_ctxt_ents(attr_key)
            if subject == '':
                subject = '我'  # 此类上下问不适用于祈使句，所以可以设定缺省值=我
            new_key = "{}={}={}={}".format(time, subject, action, object)
            attr_context = self.chat_session.get_attr_context_desc(new_key)
            if attr_context:
                self.chat_session.attr_emo_context_ref = attr_context
                return '', 5
            elif subject == '我' and action in ['心情', '情绪']:
                print("Coming here to retrieve possible events that may impact client's emotion ...")
                emo_context = self.chat_session.get_emo_context_desc('未知')
                if emo_context:
                    self.chat_session.attr_emo_context_ref = emo_context
                    return '', 5
        return '', 0

    """
    # Rule 8.3: Emo Context (based on context cat 7.3)
    """
    def set_emo_context(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 2:
            emo_key = item_list[0].replace(' ', '').strip()
            emo_val = item_list[1].replace(' ', '').strip()
            self.chat_session.set_emo_context(emo_key, emo_val)
        return '', 0

    def retry_emo_context(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1:
            emo_key = item_list[0].replace(' ', '').strip()
            emo_context = self.chat_session.get_emo_context_desc(emo_key)
            if emo_context:
                self.chat_session.attr_emo_context_ref = emo_context
                return '', 5
        return '', 0
    """
    # Rule 9: Context topic (based on context cat 8: context topic)
    """
    def add_context_topic(self, sentence):
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            topic = item_list[0].replace(' ', '')
            self.chat_session.add_context_topic(topic)
        return '', 0

    def get_context_topic(self, sentence):
        # self.chat_session.update_pair = False
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            deft = item_list[0].replace(' ', '').strip()
            extra = self.chat_session.get_context_topic()
            if extra:
                return extra, 0
            elif deft:
                if deft.find('|') > 0:
                    deft_opts = deft.split('|')
                    deft = random.choice(deft_opts)
                return deft, 0
        return "谢谢啊，你真是善良体贴的好童鞋呢！", 0

    def get_context_topic_with_retry(self, sentence):
        # self.chat_session.update_pair = False
        if self.chat_session.has_context():
            self.chat_session.keep_context = True
            need_retry = 1
        else:
            need_retry = 0

        extra = self.chat_session.get_context_topic()
        item_list = re.findall(r'\[(.*?)\]', sentence)
        if len(item_list) == 1 and item_list[0]:
            deft = item_list[0].replace(' ', '')
            if deft.find('|') > 0:
                deft_opts = deft.split('|')
                deft = random.choice(deft_opts)
            if extra:
                return extra, need_retry
            else:
                return deft, need_retry
        return "谢谢啊，你真是善良体贴的好童鞋呢！", need_retry

    def get_where_we_are(self, sentence):
        self.chat_session.update_pair = False

        ret = self.chat_session.get_context_topic(allow_default=False)
        if ret:
            return ret, 0

        last_question = self.chat_session.last_question
        last_answer = self.chat_session.last_answer
        if last_question and last_answer:
            last_answer = re.sub(r'_nl_|_np_|_nr_', ' ', last_answer)
            return "你刚刚说了，“" + last_question + "”，然后我回复说，“" + last_answer + "”", 0

        return "咱们才刚开始聊天呀。在此之前，你还什么也没说呢。", 0

    """
    # Others
    """
    # @staticmethod
    # def todo_smart_home():
    #     return '（系统信息：暂无智能家居接口。）', 0
    #
    # @staticmethod
    # def todo_addon_service():
    #     return '（系统信息：暂无外部或线下服务接口。）', 0

    def _output_sentence_with_inner_function_executed(self, output):
        if output.strip() == '':
            return ''

        last_word, pre_last_word = None, None
        word_list = []
        for word in output.split():  # output has to be already separated
            word = word.strip()
            if word and word.startswith('_func_'):
                ret_text = ''
                func_info = word[6:]
                # Does not support para0 in this case, or para_list
                para1_index = func_info.find('_para1_')
                para2_index = func_info.find('_para2_')
                if para1_index == -1:  # No parameter at all
                    func_name = func_info
                    if func_name in self.inner_fun_dict:
                        ret_text = self.inner_fun_dict[func_name]()[0]
                elif para2_index == -1:  # para1 only
                    func_name = func_info[:para1_index]
                    func_para = func_info[para1_index+7:]
                    # The parameter value was embedded in the text (part of the string) of the training example.
                    if func_name in self.inner_fun_dict:
                        ret_text = self.inner_fun_dict[func_name](func_para)[0]
                else:
                    func_name = func_info[:para1_index]
                    func_para1 = func_info[para1_index+7:para2_index]
                    func_para2 = func_info[para2_index+7:]
                    if func_name in self.inner_fun_dict:
                        ret_text = self.inner_fun_dict[func_name](func_para1, func_para2)[0]
                word = ret_text

            if word:  # word might be the output of the previous function: _func_
                word, new_last = get_formatted_word(word, last_word, pre_last_word)
                if word not in BPE_TAG_SET:
                    word_list.append(word)
                pre_last_word = last_word
                last_word = new_last if new_last else word.strip()

        return join_formatted_list(word_list)


def call_function(func_info, sess_func, out_sentence, para_list=None, in_question=None):
    """
    Args:
        func_info: the (rest) part of the text in _func_rest token
        sess_func: an instance of SessionFunction object
        out_sentence: the whole sentence or the part in between _nt_ tokens so that the parameter
                content inside brackets [] can be extracted
        para_list: the list of parameters extracted using regular expression from the user input
        in_question: the user input
    """
    fun_dict = {
        # 'set_use_simp': sess_func.set_use_simp,
        # 'skip_tradit_convert': sess_func.skip_tradit_convert,
        'get_random_out': SessionFunction.get_random_out,
        'greet_if_not_yet': sess_func.greet_if_not_yet,
        'check_and_set_question_asked': sess_func.check_and_set_question_asked,
        # 'get_xgg_birth_year': sess_func.get_xgg_birth_year,
        # 'get_boshao_contact': SessionFunction.get_boshao_contact,
        'show_random_ads': sess_func.show_random_ads,
        'reset_session': sess_func.reset_session,

        # 'get_date_time': sess_func.get_date_time,
        # 'get_time': sess_func.get_time,
        # 'get_area_time':sess_func.get_area_time,
        # 'get_prev_timezone_area_may_retry': sess_func.get_prev_timezone_area_may_retry,
        # 'get_xgg_local_area_info': sess_func.get_xgg_local_area_info,
        # 'get_today': sess_func.get_today,
        # 'get_today_lunar': sess_func.get_today_lunar,
        # 'get_today_jieqi_desc': SessionFunction.get_today_jieqi_desc,
        # 'get_today_season_desc': SessionFunction.get_today_season_desc,
        # 'get_date_weekday': sess_func.get_date_weekday,
        # 'get_weekday_or_lunar_date_from_date': sess_func.get_weekday_or_lunar_date_from_date,
        # 'get_weekday_or_lunar_date_from_month_day': sess_func.get_weekday_or_lunar_date_from_month_day,
        # 'get_date_with_span_from_weekday': sess_func.get_date_with_span_from_weekday,
        # 'get_date_prev_next_from_weekday': sess_func.get_date_prev_next_from_weekday,
        # 'get_current_year': sess_func.get_current_year,
        # 'get_year': sess_func.get_year,
        # 'get_jieri_date': sess_func.get_jieri_date,
        # 'get_jieri_date_of_year': sess_func.get_jieri_date_of_year,
        # 'get_xgg_age_in_year': sess_func.get_xgg_age_in_year,

        'comp_pre_res_with_0': sess_func.comp_pre_res_with_0,
        'equal_pre_ress': sess_func.equal_pre_ress,
        'evaluate_mixed_arith_exp': sess_func.evaluate_mixed_arith_exp,
        # 'get_converted_units': sess_func.get_converted_units,
        'get_huge_most_data': sess_func.get_huge_most_data,
        # 'get_huge_obj_data': sess_func.get_huge_obj_data,
        # 'get_area_attr': sess_func.get_area_attr,
        # 'extract_prev_unit_with_amount_and_convert': sess_func.extract_prev_unit_with_amount_and_convert,

        'get_story_any': sess_func.get_story_any,
        'get_story_cat': sess_func.get_story_cat,
        'continue_story': sess_func.continue_story,
        'get_joke_any': sess_func.get_joke_any,
        'get_duanzi_any': sess_func.get_duanzi_any,

        'get_poem_any': sess_func.get_poem_any,
        'get_poem_by_writer': sess_func.get_poem_by_writer,
        'get_poem_by_title': sess_func.get_poem_by_title,
        'get_prev_next_poem_line': sess_func.get_prev_next_poem_line,
        'get_poem_from_line': sess_func.get_poem_from_line,
        'set_context_poem_pair': sess_func.set_context_poem_pair,
        'set_context_poem_pair_by_line': sess_func.set_context_poem_pair_by_line,
        # 'get_poem_line_by_context': sess_func.get_poem_line_by_context,
        # 'get_poem_by_context': sess_func.get_poem_by_context,
        # 'get_poem_info': sess_func.get_poem_info,
        # 'get_poem_mugua_info': sess_func.get_poem_mugua_info,
        'compose_poem_any': sess_func.compose_poem_any,
        'compose_poem_type': sess_func.compose_poem_type,
        'compose_ci_poem_any': sess_func.compose_ci_poem_any,
        'compose_ci_poem': sess_func.compose_ci_poem,
        'get_lyric_by_title': sess_func.get_lyric_by_title,
        # 'get_song_info': sess_func.get_song_info,
        'get_pmsg_context': sess_func.get_pmsg_context,

        'keep_topic': sess_func.keep_topic,
        'propose_topic': sess_func.propose_topic,
        'beg_for': sess_func.beg_for,
        'beg_for_if_not': sess_func.beg_for_if_not,
        'run_last_topic': sess_func.run_last_topic,

        'set_context_ref': sess_func.set_context_ref,
        'set_context_ref_cache': sess_func.set_context_ref_cache,
        'set_categ': sess_func.set_categ,
        'set_context_by_quest': sess_func.set_context_by_quest,
        'keep_context_if_any': sess_func.keep_context_if_any,
        'get_pending_yes_with_context': sess_func.get_pending_yes_with_context,
        'retry_context_ref': sess_func.retry_context_ref,
        'retry_context_ref_with_base': sess_func.retry_context_ref_with_base,
        'retry_pron_ctxt_ref': sess_func.retry_pron_ctxt_ref,
        # 'retry_pron_ctxt_ref_with_base': sess_func.retry_pron_ctxt_ref_with_base,
        'retry_pta3_ctxt_ref': sess_func.retry_pta3_ctxt_ref,
        'retry_context_or_last_quest': sess_func.retry_context_or_last_quest,
        'set_pronoun': sess_func.set_pronoun,
        'retry_def_pronoun': sess_func.retry_def_pronoun,

        'ask_name_if_not_yet': sess_func.ask_name_if_not_yet,
        'set_user_name': sess_func.set_user_name,
        # 'set_user_name_rejected': sess_func.set_user_name_rejected,
        'set_iam_name': sess_func.set_iam_name,
        'get_user_name': sess_func.get_user_name,
        # 'get_user_name_full': sess_func.get_user_name_full,
        # 'get_user_name_xing': sess_func.get_user_name_xing,
        # 'get_user_name_with_title_updated': sess_func.get_user_name_with_title_updated,
        # 'set_gender': sess_func.set_gender,
        'output_if_gender': sess_func.output_if_gender,
        # 'clear_user_name': sess_func.clear_user_name,

        # 'get_celeb_info': sess_func.get_celeb_info,
        'get_nation_cap_info': sess_func.get_nation_cap_info,
        'get_province_city_info': sess_func.get_province_city_info,
        # 'get_meaning_from_entry': sess_func.get_meaning_from_entry,
        # 'get_whatis_from_entry': sess_func.get_whatis_from_entry,
        # 'get_whowhatis_from_entry': sess_func.get_whowhatis_from_entry,
        # 'get_lg_person_by_key': sess_func.get_lg_person_by_key,
        # 'get_lg_thing_by_key': sess_func.get_lg_thing_by_key,
        # 'get_lg_work_by_key': sess_func.get_lg_work_by_key,
        # 'get_maybe_person_by_key': sess_func.get_maybe_person_by_key,
        # 'get_maybe_thing_by_key': sess_func.get_maybe_thing_by_key,
        # 'get_maybe_work_by_key': sess_func.get_maybe_work_by_key,
        'get_niobj_info': sess_func.get_niobj_info,
        # 'get_val_from_key': sess_func.get_val_from_key,
        # 'get_celeb_name_attr': sess_func.get_celeb_attr_info,
        # 'get_celeb_attr_info': sess_func.get_celeb_attr_info,
        # 'get_celeb_relation_info': sess_func.get_celeb_attr_info,
        # 'get_celeb_work_info': sess_func.get_celeb_attr_info,
        # 'get_celeb_other_relation': sess_func.get_celeb_other_relation,
        # 'get_work_attr': sess_func.get_work_attr,
        # 'get_org_leader': sess_func.get_org_leader,
        # 'get_animal_attr_info': sess_func.get_animal_attr_info,
        'get_hist_event_time': sess_func.get_hist_event_time,
        'get_sport_event_info': sess_func.get_sport_event_info,

        'get_en_trans_from_cn': sess_func.get_en_trans_from_cn,

        # 'start_chengyu_jielong': sess_func.start_chengyu_jielong,
        # 'get_init_jielong_chengyu': sess_func.get_init_jielong_chengyu,
        # 'get_jielong_chengyu': sess_func.get_jielong_chengyu,
        # 'get_jielong_chengyu_for_user': sess_func.get_jielong_chengyu_for_user,
        # 'end_chengyu_jielong': sess_func.end_chengyu_jielong,

        'ask_weather_city': sess_func.ask_weather_city,
        # 'get_weather_from_desc': sess_func.get_weather_from_desc,

        # 'start_mimic_role': sess_func.start_mimic_role,
        # 'check_mimic_step': sess_func.check_mimic_step,
        # 'end_mimic_role': sess_func.end_mimic_role,

        # 'set_pending_ans': sess_func.set_pending_ans,
        # 'set_pending_ans_with_cref': sess_func.set_pending_ans_with_cref,
        'propose_pending_quest': sess_func.propose_pending_quest,
        'get_pending_ans': sess_func.get_pending_ans,
        'get_pending_ans_with_base': sess_func.get_pending_ans_with_base,
        'get_pending_ans_with_extra_output': sess_func.get_pending_ans_with_extra_output,

        # 'set_missing_ans': sess_func.set_missing_ans,
        # 'set_missing_ans_with_cref': sess_func.set_missing_ans_with_cref,
        # 'get_missing_ans': sess_func.get_missing_ans,

        'set_favor_item': sess_func.set_favor_item,
        'get_favor_item': sess_func.get_favor_item,
        'set_attr_context': sess_func.set_attr_context,
        'retry_attr_context': sess_func.retry_attr_context,
        'set_emo_context': sess_func.set_emo_context,
        'retry_emo_context': sess_func.retry_emo_context,

        # 'add_context_topic': sess_func.add_context_topic,
        'get_context_topic': sess_func.get_context_topic,
        'get_context_topic_with_retry': sess_func.get_context_topic_with_retry,
        'get_where_we_are': sess_func.get_where_we_are,

        # 'todo_smart_home': SessionFunction.todo_smart_home,
        # 'todo_addon_service': SessionFunction.todo_addon_service,
    }

    para0_index = func_info.find('_para0_')
    para1_index = func_info.find('_para1_')
    para2_index = func_info.find('_para2_')
    if para0_index == -1 and para1_index == -1:  # No parameter at all
        func_name = func_info
        if func_name in fun_dict:
            return fun_dict[func_name]() + (0,)
    elif para0_index > 0:
        func_name = func_info[:para0_index]
        if para1_index == -1:  # para0 only
            func_para0 = func_info[para0_index+7:]
        else:
            func_para0 = func_info[para0_index + 7:para1_index]

        if func_para0.startswith('non'):
            this_only = 0  # Do nothing to the text
        elif func_para0.startswith('pre'):
            this_only = 1  # Remove all texts before this
        elif func_para0.startswith('flw'):
            this_only = 2  # Remove all texts after this
        elif func_para0.startswith('slf'):
            this_only = 3  # Use the output of this function only, and re-scan the output
        elif func_para0.startswith('cat'):
            this_only = 4  # Concat the existing part with the output of this function, and re-scan the whole.
        elif func_para0.startswith('ton'):
            this_only = 0  # supports para0 only, and define this_only by itself (0 will be discarded)
        else:
            return "矮油，我认输，这个问题真把我难住了。", 0, 0

        if para1_index == -1:  # para0 only
            if func_para0.endswith('_ab'):  # sentence a and sentence b in the pair
                return fun_dict[func_name](out_sentence, in_question) + (this_only,)
            elif func_para0.startswith('ton'):
                return fun_dict[func_name](out_sentence)
            else:
                return fun_dict[func_name](out_sentence) + (this_only,)
        elif para2_index == -1:  # para0 and para1
            func_para1 = func_info[para1_index+7:]
            if para_list and len(para_list) >= 1 and func_para1[-1] == '_':
                para1_val1 = para_list[0]
                if func_para0.endswith('_ab'):  # sentence a and sentence b in the pair
                    return fun_dict[func_name](out_sentence, in_question, para1_val1) + (this_only,)
                else:
                    return fun_dict[func_name](out_sentence, para1_val1) + (this_only,)
            elif func_para0.endswith('_ab'):  # sentence a and sentence b in the pair
                return fun_dict[func_name](out_sentence, in_question, func_para1) + (this_only,)
            else:
                return fun_dict[func_name](out_sentence, func_para1) + (this_only,)
        else:  # para0, para1, and para2
            func_para1 = func_info[para1_index+7:para2_index]
            func_para2 = func_info[para2_index+7:]
            if para_list and len(para_list) >= 2 and func_para1[-1] == func_para2[-1] == '_':  # _num1_, _num2_
                para1_val = para_list[0]
                para2_val = para_list[1]
                if len(para_list) == 3 and func_para2[-1] == '_':
                    para3_val = para_list[2]
                    return fun_dict[func_name](out_sentence, para1_val, para2_val, para3_val) + (this_only,)
                elif func_para0.endswith('_ab'):  # sentence a and sentence b in the pair
                    return fun_dict[func_name](out_sentence, in_question, para1_val, para2_val) + (this_only,)
                else:
                    return fun_dict[func_name](out_sentence, para1_val, para2_val) + (this_only,)
            else:
                return fun_dict[func_name](out_sentence, func_para1, func_para2) + (this_only,)
    else:  # No para0, but at least para1
        func_name = func_info[:para1_index]
        if para2_index == -1:  # para1 only
            func_para = func_info[para1_index+7:]
            if para_list and len(para_list) >= 1 and func_para[-1] == '_':  # such as _cipai_, _mixed_arith_exp_
                para1_val = para_list[0]
                return fun_dict[func_name](para1_val) + (0,)
            else:
                # The parameter value was embedded in the text (part of the string) of the training example.
                return fun_dict[func_name](func_para) + (0,)
        else:  # para1 and para2
            func_para1 = func_info[para1_index+7:para2_index]
            func_para2 = func_info[para2_index+7:]
            if para_list and len(para_list) >= 2 and func_para1[-1] == func_para2[-1] == '_':  # _num1_, _num2_
                para1_val = para_list[0]
                para2_val = para_list[1]
                return fun_dict[func_name](para1_val, para2_val) + (0,)
            else:
                return fun_dict[func_name](func_para1, func_para2) + (0,)

    return "矮油，我认输，这个问题确实把我难住了。", 0, 0


if __name__ == "__main__":
    from settings import PROJECT_ROOT
    from chatbot.tokenizeddata import TokenizedData
    from addons.rules.knowledgebase import KnowledgeBase
    from addons.rules.chatsession import ChatSession
    from chatbot.datautil import cut_text_line

    corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')
    knbs_dir = os.path.join(PROJECT_ROOT, 'Data', 'KnowledgeBase')

    # td = TokenizedData(corpus_dir=corp_dir, training=False)

    # hparams = td.hparams

    knowledge_base = KnowledgeBase()
    # knowledge_base.load_knbase(knbs_dir)
    func_data = FunctionData(params=None, knowledge_base=knowledge_base, addons_dict=None)

    cs = ChatSession(1)
    sf = SessionFunction(func_data, cs)

    in_text_list = [
        '设鸡（雉）有x只，兔有y只。 _np_ 根据鸡兔（头）的数量，我们有： <mat_equ>方程1：x+y=二十五</mat_equ> _np_ 根据它们腿的数量（另单只鸡的腿数2和单个兔的腿数4的事实），我们有： <mat_equ>方程2：2*x+4*y=七十六</mat_equ> _np_ 解该方程组得：<mat_equ>方程1，2的解x，y=结果1，2</mat_equ>。所以鸡有<mat_equ>结果1</mat_equ>只，兔有<mat_equ>结果2</mat_equ>只。',
        '设钢笔为x盒，则铅笔为(27-x)盒。 _np_ 根据钢笔每盒10支，铅笔每盒12支，共计300支的条件列出方程： <mat_equ>方程1：10*x+12*(27-x)=300</mat_equ> _np_ 解方程得：<mat_equ>方程1的解x=结果1</mat_equ>。所以钢笔为<mat_equ>结果1</mat_equ>盒，铅笔为<mat_equ>27-结果1=结果2</mat_equ>盒。',
        '设蜘蛛为x只，蜻蜓y只，蝉z只。 _np_ 根据动物的数量，我们有： <mat_equ>方程1：x+y+z=18</mat_equ> _np_ 根据它们腿的数量，我们有： <mat_equ>方程2：8*x+6*y+6*z=118</mat_equ> _np_ 根据它们翅膀的数量，我们有： <mat_equ>方程2：2*y+z=20</mat_equ> _np_ 解该方程组得：<mat_equ>方程1，2，3的解x，y，z=结果1，2，3</mat_equ>。所以蜻蜓有<mat_equ>结果2</mat_equ>只。',
    ]
    for in_text in in_text_list:
        print(in_text)
        cs.before_prediction("AA")
        input_txt = ' '.join(cut_text_line(in_text)).strip()
        print(sf.scan_math_equations(input_txt))
        cs.after_prediction("BB", "cc")
        print("===")