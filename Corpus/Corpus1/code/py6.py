import re

import datetime as dt
import pytz as tz

from chatbot.datautil import cut_text_line

WEEK_DAYS = ["一", "二", "三", "四", "五", "六", "日"]
RELATIVE_CAL_DAYS = ['大前天', '前天', '昨天', '今天', '明天', '后天', '大后天']
RELATIVE_CAL_YEARS = ['大前年', '前年', '去年', '今年', '明年', '后年', '大后年']

# ============================== _cal_year_ ===========================================================
# 大前年，前年，去年，今年，明年，后年，大后年
# 3年前，5年后，18年前，50年后，100年前，100年后；两年前，八年前，十年后，八十八年前，九十九年后，百年前，一百年后
# 1998，1998年，公元2002年，公历2008年；一九八八，一九八八年，公元二〇〇八年，阳历二〇〇八
_gongli_options = r'(公\s?元\s?|公\s?历\s?|阳\s?历\s?)?'
_hanzi_0_9 = r'[〇零一二三四五六七八九]'
_hanzi_1_9 = r'[一二三四五六七八九]'
_hanzi_19_20_years = r'(一\s?九|二\s?[〇零])\s?{0}\s?{0}'.format(_hanzi_0_9)
_hanzi_years_before_after = r'({0}|两|({1}\s?)?十(\s?{1})?|(一\s?)?百)\s?年(\s?[之以])?\s?[前后]'.format(_hanzi_0_9, _hanzi_1_9)
_years_before_after = r'([0-9]|[1-9][0-9]|100)\s?年(\s?[之以])?\s?[前后]'
_cal_year_cat = re.compile(r'(({0})|{1}|{2}|{4}(19|20)[0-9][0-9](\s?年)?|{4}{3}(\s?年)?)'.format(
    '|'.join([r'\s?'.join(list(yr)) for yr in RELATIVE_CAL_YEARS]),
    _years_before_after, _hanzi_years_before_after, _hanzi_19_20_years, _gongli_options))

# ============================== _cal_month_day_ ======================================================
# 1月10日，2月28，三月十九号，十一月十一
_month_12_pat = r'(1[012]|[1-9]|{}|十(\s?[一二])?)'.format(_hanzi_1_9)
_month_day_31_pat = r'(3[01]|[12][0-9]|[1-9]|三\s?十(\s?一)?|(二\s?)?十(\s?{0})?|(初\s?)?{0})'.format(_hanzi_1_9)
_cal_month_day_cat = re.compile(r'{0}\s?月\s?{1}(\s?[日号])?'.format(_month_12_pat, _month_day_31_pat))

# ============================== _cal_month_ and _day_in_month_ =======================================
# 上上个月，大上个月，上月，上个月，这个月，本月，下月，下个月， 下下个月，大下个月
# 最后一天，最后1日，正数第十一天，倒数第9日，第12天，第二十二日，12日，29号，三十日，十八号
_month_span_opts =['大上', '大下', '上上', '下下', '这', '本', '此', '上', '下']
_relative_cal_month_cat = re.compile(r'({})(\s?一)?(\s?个)?\s?月'.format(
    '|'.join([r'\s?'.join(list(ms)) for ms in _month_span_opts])))
_day_in_month_cat = re.compile(
    r'最\s?后\s?[一1]\s?[天日]|([正倒]\s?数\s?)?第\s?{0}\s?[天日]|{0}\s?[日号]'.format(_month_day_31_pat))

# ============================== _cal_week_and_day_ ======================================================
# 上个周一，下个礼拜二，下一个星期的星期三，上一周的周四，下下个礼拜的礼拜五，上周周六，下星期星期天
_week_span_opts = ['大上', '大下', '上上', '下下', '这', '本', '此', '上', '下']
_week_desc_opts = ['星期', '礼拜', '周']
_week_day_ids = ['一', '二', '三', '四', '五', '六', '日', '天']

_prev_next_weekday_cat = re.compile(r'(过\s?去|未\s?来)\s?(的\s?)?({WD})\s?({WI})'.format(
    WD='|'.join([r'\s?'.join(list(wd)) for wd in _week_desc_opts]),
    WI='|'.join(_week_day_ids)
))

# the week part may or may not show up
_week_and_day_cat = \
    re.compile(r'(({WS})\s?(一\s?[个次]\s?(的\s?)?|次\s?(的\s?)?|[一个]\s?)?)?({WD})((\s?的)?\s?({WD}))?\s?({WI})'.format(
    WS='|'.join([r'\s?'.join(list(ws)) for ws in _week_span_opts]),
    WD='|'.join([r'\s?'.join(list(wd)) for wd in _week_desc_opts]),
    WI='|'.join(_week_day_ids)
))
_non_week_and_day_cat = re.compile(
    r'[刚已上].{0,4}\s?(过\s?去|出\s?现|度\s?过|结\s?束)|[要将下].{0,4}\s?(来\s?临|出\s?现|[要到]\s?来)'
)

# the week part in the pattern must show up
_week_and_day_cat2 = \
    re.compile(r'({WS})\s?(一\s?[个次]\s?(的\s?)?|次\s?(的\s?)?|[一个]\s?)?({WD})((\s?的)?\s?({WD}))?\s?({WI})'.format(
    WS='|'.join([r'\s?'.join(list(ws)) for ws in _week_span_opts]),
    WD='|'.join([r'\s?'.join(list(wd)) for wd in _week_desc_opts]),
    WI='|'.join(_week_day_ids)
))

_week_span_cat = re.compile(r'({WS})\s?(一\s?个\s?|[一个]\s?)?({WD})'.format(
    WS='|'.join([r'\s?'.join(list(ws)) for ws in _week_span_opts]),
    WD='|'.join([r'\s?'.join(list(wd)) for wd in _week_desc_opts]),
))

_day_span_cat = re.compile(r'({WD})\s?({WI})'.format(
    WD='|'.join([r'\s?'.join(list(wd)) for wd in _week_desc_opts]),
    WI='|'.join(_week_day_ids)
))

_nongli_cat = re.compile(r'[农阴]\s?历')
_gongli_cat = re.compile(r'[公阳]\s?历')

_hanzi_days_before_after = r'({0}|两|({1}\s?)?十(\s?{1})?|(一\s?)?百)\s?[天日](\s?[之以])?\s?[前后]'.format(_hanzi_0_9, _hanzi_1_9)
_days_before_after = r'([0-9]|[1-9][0-9]|100)\s?[天日](\s?[之以])?\s?[前后]'
_relative_cal_day_cat = re.compile(r'(({0})\s?[天日]|{1}|{2})'.format(
    '|'.join([r'\s?'.join(list(dy[:-1])) for dy in RELATIVE_CAL_DAYS]), _hanzi_days_before_after, _days_before_after))
_relative_cal_year_cat = re.compile(r'{}'.format('|'.join([r'\s?'.join(list(yr)) for yr in RELATIVE_CAL_YEARS])))
_dist_cal_year_cat = re.compile(r'(({0})|{1}|{2})'.format(
    '|'.join([r'\s?'.join(list(yr)) for yr in RELATIVE_CAL_YEARS]), _years_before_after, _hanzi_years_before_after))


class CalendarUtils(object):
    # Test if a given datetime is using Day Light Saving time
    @staticmethod
    def is_dst(tz_id, when=None):
        ask_tz = tz.timezone(tz_id)
        ask_dt = ask_tz.localize(when) if when else ask_tz.localize(dt.datetime.now())
        return ask_dt.dst() != dt.timedelta(0)

    # Test if this timezone has Day Light Saving time or not
    @staticmethod
    def has_dst(tz_id):
        now = dt.datetime.now()
        is_dst1 = CalendarUtils.is_dst(tz_id, dt.datetime(year=now.year, month=1, day=1))
        is_dst2 = CalendarUtils.is_dst(tz_id, dt.datetime(year=now.year, month=7, day=1))
        if is_dst1 or is_dst2:
            return True
        return False

    @staticmethod
    def get_utc_offset(tz_id, when=None):
        ask_tz = tz.timezone(tz_id)
        ask_dt = ask_tz.localize(when) if when else ask_tz.localize(dt.datetime.now())
        offset_text = ask_dt.strftime('%z')
        return "{}:{}".format(offset_text[:-2], offset_text[-2:])

    @staticmethod
    def get_now_in_timezone(params, tz_id=None, now_in_sys_tz=None):
        when = now_in_sys_tz if now_in_sys_tz else dt.datetime.now()
        if tz_id is None:
            tz_id = params['default_tz']

        if tz_id == params['system_tz']:
            ask_dt = when
        else:
            sys_tz = tz.timezone(params['system_tz'])
            ask_tz = tz.timezone(tz_id)
            ask_dt = sys_tz.localize(when).astimezone(ask_tz)
        return ask_dt

    # return a tuple: (标准时区,当前夏令时时区偏移)
    @staticmethod
    def get_timezone_offsets(tz_id, now_in_sys_tz=None):
        now = now_in_sys_tz if now_in_sys_tz else dt.datetime.now()
        cur_offset = CalendarUtils.get_utc_offset(tz_id, now)
        if not CalendarUtils.is_dst(tz_id, now):  # std_offset is the same as cur_offset
            return cur_offset, None
        else:
            year = now.year
            is_dst1 = CalendarUtils.is_dst(tz_id, dt.datetime(year=year, month=1, day=1))
            if not is_dst1:
                std_offset = CalendarUtils.get_utc_offset(tz_id, dt.datetime(year=year, month=1, day=1))
            else:  # not is_dst2
                std_offset = CalendarUtils.get_utc_offset(tz_id, dt.datetime(year=year, month=7, day=1))
            return std_offset, cur_offset

    @staticmethod
    def get_timezone_texts(tz_id):
        hanzi_list = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二"]
        std_offset, cur_offset = CalendarUtils.get_timezone_offsets(tz_id)
        if std_offset == "+00:00":
            std_text = "中时区（GMT{}）".format(std_offset)
        elif std_offset.endswith(":00"):
            hour_offset = int(std_offset[1:-3])
            if std_offset.startswith("+"):
                std_text = "东{}区（GMT{}）".format(hanzi_list[hour_offset-1], std_offset)
            else:
                std_text = "西{}区（GMT{}）".format(hanzi_list[hour_offset-1], std_offset)
        else:
            std_text = "GMT{}".format(std_offset)

        if cur_offset is None:
            return std_text, None
        else:
            return std_text, "GMT{}".format(cur_offset)

    @staticmethod
    def is_leap_year(year):
        if (year % 4) != 0:
            return False
        if ((year % 100) == 0) & ((year % 400) != 0):
            return False
        return True


class CalendarPatterns(object):
    @staticmethod
    def get_cal_year_cat():
        return _cal_year_cat

    @staticmethod
    def get_cal_month_day_cat():
        return _cal_month_day_cat

    @staticmethod
    def get_prev_next_weekday_cat():
        return _prev_next_weekday_cat

    @staticmethod
    def get_week_and_day_cat():
        return _week_and_day_cat

    @staticmethod
    def get_non_week_and_day_cat():
        return _non_week_and_day_cat

    @staticmethod
    def get_weekday_only_cat():
        return _day_span_cat

    @staticmethod
    def get_relative_cal_day_cat():
        return _relative_cal_day_cat

    @staticmethod
    def get_relative_cal_year_cat():
        return _relative_cal_year_cat

    @staticmethod
    def get_relative_cal_month_cat():
        return _relative_cal_month_cat

    @staticmethod
    def get_day_in_month_cat():
        return _day_in_month_cat

    @staticmethod
    def fullmatch_cal_year_cat(str_ent: str):
        cs = ' '.join(cut_text_line(str_ent)).strip()
        if re.fullmatch(_cal_year_cat, cs):
            return True
        return False

    @staticmethod
    def fullmatch__dist_cal_year_cat(str_ent: str):
        cs = ' '.join(cut_text_line(str_ent)).strip()
        if re.fullmatch(_dist_cal_year_cat, cs):
            return True
        return False

    @staticmethod
    def is_for_nongli(sentence):
        if re.search(_nongli_cat, sentence) and not re.search(_gongli_cat, sentence):
            return True
        else:
            return False

    @staticmethod
    def for_gongli_or_nongli(sentence):
        if re.search(_gongli_cat, sentence) and not re.search(r'[农阴]', sentence):
            return "GONGLI"
        elif re.search(_nongli_cat, sentence) and not re.search(r'[公阳]', sentence):
            return "NONGLI"
        else:
            return None

    @staticmethod
    def parse_cal_year_text(base_year: int, cal_year_text: str):
        cal_year = re.sub(r'[之以](?=[前后])', '', re.sub(r'公元|公历|阳历', '', cal_year_text))
        if cal_year in RELATIVE_CAL_YEARS:
            # RELATIVE_CAL_YEARS = ['大前年', '前年', '去年', '今年', '明年', '后年', '大后年']
            c_year = base_year - 3 + RELATIVE_CAL_YEARS.index(cal_year)
            y_name = "{}（{}年）".format(cal_year, c_year)
        elif cal_year.endswith('年前'):
            c_year = base_year - CalendarPatterns._convert_year_text_2_int(cal_year[:-2])
            y_name = "{}（{}年）".format(cal_year, c_year)
        elif cal_year.endswith('年后'):
            c_year = base_year + CalendarPatterns._convert_year_text_2_int(cal_year[:-2])
            y_name = "{}（{}年）".format(cal_year, c_year)
        elif cal_year[-1] == '年':
            c_year = CalendarPatterns._convert_year_text_2_int(cal_year[:-1])
            y_name = "{}年".format(c_year)
        else:
            c_year = CalendarPatterns._convert_year_text_2_int(cal_year)
            y_name = "{}年".format(c_year)
        return c_year, y_name

    @staticmethod
    def _convert_year_text_2_int(year_text):  # 1909, 2001, 一九〇九，二零一二
        if year_text.isdigit():
            return int(year_text)

        hanzi_list = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
        y_text = year_text.replace('〇', '零')
        if y_text.startswith('一九') or y_text.startswith('二零'):
            out = [str(hanzi_list.index(t)) for t in y_text]
            return int(''.join(out[:]).strip())
        else:
            if y_text in ['百', '一百']:
                return 100
            elif len(y_text) == 1:
                if y_text == '十':
                    return 10
                elif y_text == '两':
                    return 2
                else:
                    return hanzi_list.index(y_text)
            elif len(y_text) == 2:
                if y_text.startswith('十'):
                    return 10 + hanzi_list.index(y_text[1])
                else:  # y_text.endswith('十')
                    return hanzi_list.index(y_text[0]) * 10
            else:  # len(y_text) == 3
                return hanzi_list.index(y_text[0]) * 10 + hanzi_list.index(y_text[2])

    @staticmethod
    def parse_cal_month_day_text(cal_month_day: str):
        month_day_list = re.sub(r'[月日号]', ' ', cal_month_day).strip().split()
        month_text, day_text = month_day_list[0], month_day_list[1]
        if month_text.isdigit():
            c_month = int(month_text)
        else:
            month_hanzi_list = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二"]
            if month_text in month_hanzi_list:
                c_month = month_hanzi_list.index(month_text)
            else:
                c_month = None

        c_day = CalendarPatterns._convert_day_in_month_text_2_int(day_text)
        return c_month, c_day

    @staticmethod
    def _convert_day_in_month_text_2_int(dim_text):
        if dim_text.isdigit():
            return int(dim_text)

        day_hanzi_list = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        if dim_text in day_hanzi_list:
            c_day = day_hanzi_list.index(dim_text)
        elif dim_text.startswith('初') and len(dim_text) == 2:
            c_day = day_hanzi_list.index(dim_text[1])
        elif dim_text.startswith('十') and len(dim_text) == 2:
            c_day = 10 + day_hanzi_list.index(dim_text[1])
        elif dim_text.startswith('二十') and len(dim_text) == 3:
            c_day = 20 + day_hanzi_list.index(dim_text[2])
        elif dim_text == '二十':
            c_day = 20
        elif dim_text == '三十':
            c_day = 30
        elif dim_text == '三十一':
            c_day = 31
        else:
            c_day = None
        return c_day

    @staticmethod
    def parse_cal_month_text(cal_month: str):
        month_span_opts = ['前', '上', '这', '下', '后']
        cm = cal_month.replace('个', '').replace('本', '这').replace('此', '这')
        cm = re.sub(r'(?<=[上这下])一', '', cm)
        cm = re.sub(r'大上|上上', '前', re.sub(r'大下|下下', '后', cm))
        if cm[0] in month_span_opts:
            month_span_idx = month_span_opts.index(cm[0]) - 2
            month_desc = ['上上', '上', '这', '下', '下下'][month_span_idx+2] + '个月'
            return month_span_idx, month_desc
        return None, None

    @staticmethod
    def parse_day_in_month_text(day_in_month: str):
        dim = re.sub(r'[天日号]', '', day_in_month)
        if dim in ['最后一', '最后1']:
            day_in_month_idx = -1
        elif dim.startswith('倒数'):
            c_day = CalendarPatterns._convert_day_in_month_text_2_int(dim[3:])
            if c_day:
                day_in_month_idx = -1 * c_day
            else:
                return None
        else:
            dim = dim.replace('正数', '').replace('第', '')
            day_in_month_idx = CalendarPatterns._convert_day_in_month_text_2_int(dim)
        return day_in_month_idx

    @staticmethod
    def parse_week_and_day_text(cal_week_and_day: str):
        wd_desc0 = cal_week_and_day.replace('次', '个').replace('个的', '个').replace('此', '这').strip()
        if wd_desc0.find("周") >= 0:  # 但凡有周字出现
            wd_desc0 = wd_desc0.replace("星期", "周").replace("礼拜", "周").replace('周天', '周日')
        elif wd_desc0.find("星期") >= 0 and wd_desc0.find("礼拜") >= 0:
            wd_desc0 = wd_desc0.replace("礼拜", "星期")

        tmp = wd_desc0.replace('一个', '一').replace('个', '一')
        tmp = tmp.replace('星期', '周').replace('礼拜', '周').replace('周天', '周日')

        if tmp.count('周') == 1 and tmp.find('一周') >= 0:
            week_span_idx = None
            repeat_idx = CalendarPatterns._get_week_span_idx_from_text(tmp.replace('一周', '周'))
        else:
            week_span_idx = CalendarPatterns._get_week_span_idx_from_text(tmp.replace('一周', '周'))
            repeat_idx = None
        week_day_idx = WEEK_DAYS.index(tmp[-1]) if tmp[-1] in WEEK_DAYS else None

        return wd_desc0, week_span_idx, repeat_idx, week_day_idx

    @staticmethod
    def extract_week_and_or_day_pattern(question: str):
        wd_ind_list = [(m.start(0), m.end(0)) for m in re.finditer(_week_and_day_cat2, question)]
        if len(wd_ind_list) == 1:
            start, end = wd_ind_list[0]
            text = ''.join(question[start:end].split())
            if not text.endswith('周天'):
                _, span_idx, repeat_idx, day_idx = CalendarPatterns.parse_week_and_day_text(text)
                if (span_idx is not None or repeat_idx is not None) and day_idx is not None:
                    return 3, span_idx, repeat_idx, day_idx
        else:
            ws_ind_list = [(m.start(0), m.end(0)) for m in re.finditer(_week_span_cat, question)]
            if len(ws_ind_list) == 1:
                s1, e1 = ws_ind_list[0]
                text1 = ''.join(question[s1:e1].split())
                _, span_idx, repeat_idx, _ = CalendarPatterns.parse_week_and_day_text(text1)
                if span_idx is not None:
                    return 2, span_idx, None, None
                elif repeat_idx is not None:  # convert repeat_idx to span_idx
                    return 2, repeat_idx, None, None
            else:
                ds_ind_list = [(m.start(0), m.end(0)) for m in re.finditer(_day_span_cat, question)]
                if len(ds_ind_list) == 1:
                    s2, e2 = ds_ind_list[0]
                    text2 = ''.join(question[s2:e2].split())
                    tmp = text2.replace('星期', '周').replace('礼拜', '周').replace('周天', '周日')
                    day_idx = WEEK_DAYS.index(tmp[-1]) if tmp[-1] in WEEK_DAYS else None
                    if day_idx is not None:
                        return 1, None, None, day_idx

        return 0, None, None, None

    @staticmethod
    def _get_week_span_idx_from_text(text: str):
        if text.startswith("大上周") or text.startswith("上上周"):
            week_span_idx = -2
        elif text.startswith("上周"):
            week_span_idx = -1
        elif text.startswith("这周") or text.startswith("本周") or text.startswith("周"):
            week_span_idx = 0
        elif text.startswith("下周"):
            week_span_idx = 1
        elif text.startswith("大下周") or text.startswith("下下周"):
            week_span_idx = 2
        else:  # not supposed to come here
            week_span_idx = None

        return week_span_idx

    @staticmethod
    def parse_relative_cal_day_text(cal_day_text: str):
        day_text = re.sub(r'[之以](?=[前后])', '', cal_day_text.replace('日', '天'))
        if day_text in RELATIVE_CAL_DAYS:
            # RELATIVE_CAL_DAYS = ['大前天', '前天', '昨天', '今天', '明天', '后天', '大后天']
            delta_days = RELATIVE_CAL_DAYS.index(day_text) - 3
        elif day_text.endswith('天前'):
            delta_days = -1 * CalendarPatterns._convert_rel_day_text_2_int(day_text[:-2])
        else:  # day_text.endswith('天后'):
            delta_days = CalendarPatterns._convert_rel_day_text_2_int(day_text[:-2])
        return delta_days

    @staticmethod
    def _convert_rel_day_text_2_int(day_text):
        if day_text.isdigit():
            return int(day_text)

        hanzi_list = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
        d_text = day_text.replace('〇', '零')
        if d_text in ['百', '一百']:
            return 100
        elif len(d_text) == 1:
            if d_text == '十':
                return 10
            elif d_text == '两':
                return 2
            else:
                return hanzi_list.index(d_text)
        elif len(d_text) == 2:
            if d_text.startswith('十'):
                return 10 + hanzi_list.index(d_text[1])
            else:  # y_text.endswith('十')
                return hanzi_list.index(d_text[0]) * 10
        else:  # len(y_text) == 3
            return hanzi_list.index(d_text[0]) * 10 + hanzi_list.index(d_text[2])


if __name__ == '__main__':
    strs = [
        "上 个 周 一 呢 ",
        "下 一 个 星 期 的 星 期 三 呢",
        "下 下 个 礼 拜 的 礼 拜 二 呢",
        "上 周 呢",
        "下 周 五 呢",
        "那 周 日 呢",
        "星 期 五 呢",
        "那 下 星 期 呢 ？",
        "那 下 周 呢",
        "下 星 期 呢 ？",
        "下 下 个 礼 拜 呢 ？",
    ]

    for s in strs:
        print("{} = {}".format(s, CalendarPatterns.extract_week_and_or_day_pattern(s)))

    ss = [
        "上个月的最后一天是周几",
        "这月的倒数第4天是星期几",
        "上上个月的最后1天是礼拜几",
        "下个月的正数第3天是周几",
        "下下个月30号的农历日期",
        "上月20日的阴历日期",
        "这月29号是阴历几号？",
        "下个月第14天是星期几",
        "下下月第29日周几",
    ]

    cm_cat = CalendarPatterns.get_relative_cal_month_cat()
    dm_cat = CalendarPatterns.get_day_in_month_cat()

    for s in ss:
        ts = ' '.join(cut_text_line(s)).strip()
        cm_mat = re.search(cm_cat, ts)
        dm_mat = re.search(dm_cat, ts)

        if cm_mat or dm_cat:
            print(ts)
            if cm_mat:
                print("CM_MAT = {}".format(''.join(ts[cm_mat.start():cm_mat.end()].split())))
            if dm_mat:
                dm_text = ''.join(ts[dm_mat.start():dm_mat.end()].split())
                print("DM_MAT = {} = {}".format(dm_text, CalendarPatterns.parse_day_in_month_text(dm_text)))
        else:
            print("{} = NOT FOUND".format(ts))

    tz_ids = [
        "Europe/London",
        "America/New_York",
        "America/Los_Angeles",
        "America/Chicago",
        "America/Denver",
        "America/Tijuana",
        "America/Indiana/Indianapolis",
        "America/Detroit",
        "America/Anchorage",
        "America/Phoenix",
        "Pacific/Honolulu",
        "Asia/Shanghai",
        "Asia/Kolkata",
        "Asia/Kabul",
        "Europe/Dublin",
    ]
    for tz_id in tz_ids:
        print("{} = Has DST: {}".format(tz_id, CalendarUtils.has_dst(tz_id)))

    for tz_id in tz_ids:
        print("{} = UTC Offset: {}".format(tz_id, CalendarUtils.get_timezone_texts(tz_id)))
