import json
import random
import sys
import os
from tqdm import tqdm
from time import sleep


abspath = os.path.dirname(os.path.abspath(__file__))
classes_file = abspath + "/classes_file.json"

categories = [
    "节日",
    "书籍",
    "交通工具",
    "家具",
    "工具",
    "电器",
    "名人",
    "电影",
    "音乐",
    "体育运动",
    "食物/饮品",
    "服饰",
    "科技设备",
    "地理",
    "职业",
    "学科",
    "建筑",
    "品牌",
    "动漫",
    "文具和办公用品",
    "植物",
    "玩具",
    "游戏",
    "基础设施",
    "歌手",
    "演员",
    "作家",
    "体育明星",
    "名人"
]
classes_list = ['自行车', '轿车', '公交车', '摩托车', '飞机', '轮船', '滑板车', '火车', '地铁', '货车', '拖拉机', '电动车', '直升机', '皮划艇', '帆船', '消防车', '救护车', '出租车', '客车', '游轮', '三轮车', '高铁', '磁悬浮列车', '雪地摩托', '卡车', '椅子', '沙发', '床', '餐桌', '衣柜', '书架', '茶几', '电视柜', '鞋架', '电脑桌', '吊柜', '书桌', '梳妆台', '床头柜', '储物柜', '藤椅', '折叠桌', '躺椅', '婴儿床', '餐椅', '摇椅', '酒柜', '屏风', '挂衣架', '折叠椅', '猫', '狗', '兔子', '金鱼', '鹦鹉', '乌龟', '马', '牛', '羊', '鸡', '鸭', '鹅', '仓鼠', '蛇', '蜥蜴', '猴子', '狐狸', '狼', '袋鼠', '海豚', '大象', '老虎', '狮子', '企鹅', '熊猫', '斑马', '河马', '孔雀', '蜗牛', '蚂蚁', '螺丝刀', '锤子', '扳手', '电钻', '锯子', '尺子', '钳子', '焊接机', '螺母', '榔头', '抹泥刀', '油漆刷', '切割机', '砂纸', '卷尺', '电烙铁', '打钉枪', '铲子', '园艺剪', '刨子', '起子', '电锤', '割草机', '探测仪', '钻头', '微波炉', '洗衣机', '冰箱', '电视机', '吸尘器', '吹风机', '烤箱', '电饭锅', '空气净化器', '电扇', '电热水壶', '榨汁机', '搅拌机', '电热毯', '路由器', '咖啡机', '电暖器', '扫地机器人', '洗碗机', '投影仪', '饮水机', '加湿器', '空气炸锅', '灭蚊灯', '智能音箱', '激光打印机', '牙刷', '雨伞', '背包', '水瓶', '眼镜', '手机', '毛巾', '筷子', '勺子', '碗', '锅', '盘子', '沐浴露', '洗发水', '肥皂', '拖鞋', '闹钟', '笔记本', '纸巾', '垃圾桶', '剪刀', '梳子', '手电筒', '钥匙', '挂钟', '地毯', '台灯', '充电宝', '充电器', '毛刷', '暖水袋', '胶带']

CLASS_LIST = []

with open(classes_file, 'r', encoding='utf-8') as f:
    CLASS_LIST = json.load(f)

classes_list = classes_list+CLASS_LIST

print(CLASS_LIST[0])
def main():
    # #根据初始的类别，来生成很多名词
    #

    num = 100  # 总循环次数
    # tqdm 包装循环以显示进度条
    reso_once = []
    for index in tqdm(range(num), desc="处理进度", unit="步骤"):
        prompt = f"""你需要根据在已有的元素列表中继续生成一些新的元素。已有的元素列表：{random.shuffle(classes_list)}
        生成的元素必须满足类别：{random.sample(categories, 2)}。每个元素都必须在国内被广泛知道。
        你生成的元素如果含义很宽泛（例如'书籍'、'电影'），那么你需要生成具体的书名、电影名称。
        你只需要输出一个包含新的元素的列表，格式按照:['比萨斜塔', '战斗机', '牙刷', '连衣裙', ...]
        注意：新的元素不允许出现在已有的元素列表中。你只需要回复一个列表即可，不要回复额外的信息。
        你至少需要生成一个拥有10个新的元素的列表。每个元素都必须广为人知，是一个大众都会知道的词语。
        """
        try:
            # 模拟调用生成函数（可以是耗时操作）
            reso = qwen7b_generate(prompt=prompt)
            reso = eval(reso)

            # 检查和更新 classes_list
            for res in reso:
                if res in classes_list:
                    # print(f"元素 {res} 已存在")
                    print(random.sample(categories, 2))
                else:
                    reso_once.append(res)
                    classes_list.append(res)
                    once_list.append(res)
        except Exception as e:
            print(f"出现异常：{e}")
            pass

        # 将更新后的列表写入文件
        with open(classes_file, 'w', encoding='utf-8') as f:
            json.dump(once_list, f, ensure_ascii=False, indent=4)

    print(f"共有 {len(classes_list)} 个元素")
    print(f"新增 {len(reso_once)} 个元素")