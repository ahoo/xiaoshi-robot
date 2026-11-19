"""Labelled Chinese utterances for the voice tests (test_home.py voice). Each: (what is meant to be said after the wake word, intent, slots that must come out).
Written by hand in several registers: a plain command, a polite one, a child's, a long-winded one. `ITEMS` and the learned names are the ones the tests set up."""

COMMANDS = [
    # tidy
    ('收拾一下玩具', 'tidy', {}), ('把地上的玩具收起来', 'tidy', {}), ('帮我整理一下房间', 'tidy', {}), ('快把玩具收一收', 'tidy', {}), ('请你捡玩具', 'tidy', {}),
    ('把地上的东西收一下', 'tidy', {}), ('你来收拾玩具吧', 'tidy', {}), ('整理房间', 'tidy', {}),
    # fetch one thing
    ('把小球放进筐里', 'fetch_item', {'item': 'ball'}), ('把袜子捡起来', 'fetch_item', {'item': 'sock'}), ('帮我把毛绒玩具收到筐里', 'fetch_item', {'item': 'plush'}),
    ('把玩具车放进筐里', 'fetch_item', {'item': 'car'}), ('把积木放到筐里', 'fetch_item', {'item': 'lbrick'}), ('捡一下那个石头', 'fetch_item', {'item': 'rock'}),
    ('把香蕉玩具收好', 'fetch_item', {'item': 'banana'}), ('请把方块放进筐里', 'fetch_item', {'item': 'cube'}),
    # where is
    ('小球在哪里', 'where_is', {'item': 'ball'}), ('袜子去哪了', 'where_is', {'item': 'sock'}), ('帮我找一下毛绒玩具', 'where_is', {'item': 'plush'}), ('玩具车在哪', 'where_is', {'item': 'car'}),
    ('我的积木在哪里呢', 'where_is', {'item': 'lbrick'}), ('找找小球', 'where_is', {'item': 'ball'}), ('石头在什么地方', 'where_is', {'item': 'rock'}), ('方块哪去了', 'where_is', {'item': 'cube'}),
    # list / what's new
    ('地上有什么东西', 'list_items', {}), ('现在地上都有什么', 'list_items', {}), ('盘点一下玩具', 'list_items', {}), ('地上有什么玩具', 'list_items', {}),
    ('今天多了什么新东西', 'what_new', {}), ('有什么新东西吗', 'what_new', {}), ('少了什么', 'what_new', {}), ('什么东西被挪过位置了', 'what_new', {}), ('有什么变化', 'what_new', {}),
    # scan
    ('把家里扫描一遍', 'scan_home', {}), ('给家里建个三维地图', 'scan_home', {}), ('去看看家里', 'scan_home', {}), ('扫一扫房间', 'scan_home', {}), ('画一张家里的地图', 'scan_home', {}),
    # go to
    ('去沙发那边', 'go_to', {'place': '沙发'}), ('去桌子那里', 'go_to', {'place': '桌子'}), ('走到架子旁边', 'go_to', {'place': '架子'}), ('去椅子那边', 'go_to', {'place': '椅子'}),
    ('过去收纳筐那里', 'go_to', {'place': '收纳筐'}), ('去盆栽那边', 'go_to', {'place': '盆栽'}),
    # charge
    ('去充电', 'charge', {}), ('回去充电吧', 'charge', {}), ('该充电了', 'charge', {}), ('回充电座', 'charge', {}),
    # people
    ('谁在家', 'who_here', {}), ('有小孩吗', 'who_here', {}), ('你看到谁了', 'who_here', {}), ('有没有人', 'who_here', {}),
    ('记住这个孩子，他叫小明', 'enrol', {'newname': '小明'}), ('认识一下我女儿，她叫小红', 'enrol', {'newname': '小红'}), ('记住这个孩子', 'enrol', {}),
    ('忘掉小明', 'forget', {'person': '小明'}), ('删除小红的信息', 'forget', {'person': '小红'}),
    # naming
    ('这个积木叫做大黄', 'name_item', {'newname': '大黄'}), ('把小球叫做球球', 'name_item', {'newname': '球球'}),
    # status, privacy, small talk
    ('你在干什么', 'status', {}), ('电量还有多少', 'status', {}), ('你现在在忙什么', 'status', {}),
    ('你在录音吗', 'privacy', {}), ('你会不会一直听着', 'privacy', {}), ('别听了，静音', 'privacy', {'mute': True}),
    ('你好', 'greet', {}), ('晚上好', 'greet', {}), ('你是谁', 'identity', {}), ('你叫什么名字', 'identity', {}), ('你会做什么', 'help', {}), ('你能干什么', 'help', {}),
    # stop / cancel / resume
    ('停下', 'stop', {}), ('别动', 'stop', {}), ('先别扫了', 'stop', {}), ('算了', 'cancel', {}), ('继续', 'resume', {}), ('接着干吧', 'resume', {}),
]

# what people say in the room that is not for the robot: it must never be acted on without the wake word
CHATTER = [
    '今天晚上吃什么', '过了一小时就回来', '你把作业写完了没有', '电视里在播新闻', '妈妈在哪里', '那个球是我的', '快去洗手准备吃饭', '明天幼儿园要带什么',
    '我去充电宝那里拿一下', '这个玩具真好玩', '别跑了小心摔倒', '把门关上', '我要去睡觉了', '周末我们去公园吧', '小明你过来', '收拾东西的时候要小心', '你在干什么呢宝贝',
    '谁把杯子放在这里了', '沙发上有个枕头', '哎呀停一下我接个电话', '房间里太乱了', '这个地图好看', '记住明天要交费', '我在找手机',
]

# multi-turn scripts: (speaker role, said, expectation on the reply/action)
