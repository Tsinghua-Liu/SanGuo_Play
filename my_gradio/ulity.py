from pypinyin import lazy_pinyin

def hanzi_to_pinyin(hanzi):
    """
    将汉字转换为拼音的拼接形式
    :param hanzi: 汉字字符串
    :return: 拼音的拼接字符串
    """
    # 使用 lazy_pinyin 获取不带声调的拼音列表
    pinyin_list = lazy_pinyin(hanzi)
    # 将拼音列表拼接成一个字符串
    pinyin_result = ''.join(pinyin_list)
    return pinyin_result