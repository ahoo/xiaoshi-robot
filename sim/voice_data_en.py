"""Labelled English and mixed Chinese-English utterances for the voice tests (test_sound.py lang), the counterpart of voice_data.py.
Each: (what is meant to be said after the wake word, intent, slots that must come out). Items: ball, cube, car, banana, lbrick (building brick), rock, sock, plush (teddy bear); places are the Chinese keys
the robot uses inside (沙发 sofa, 桌子 table, 架子 shelf, 盆栽 plant, 椅子 chair, 收纳筐 basket, 充电座 charger).

Three sets, written one after the other. DEV is what the phrase lists in lang.py were grown against. HOLDOUT was written afterwards in other words: on its first look the phrase lists alone got 45 of 60 right,
and the misses led to the sentence frames in lang.py (a verb of a kind and a thing of a kind), so it counts as development now. TEST was written after that, again in other words, and is run once without
any change to the lists: its numbers are the ones to read.
"""

COMMANDS_EN_DEV = [
    # tidy
    ('tidy up the toys', 'tidy', {}), ('could you please clean up the floor', 'tidy', {}), ('pick up all the toys', 'tidy', {}), ('put the toys away', 'tidy', {}), ('tidy the room', 'tidy', {}),
    ('can you collect the toys', 'tidy', {}), ('clean up', 'tidy', {}),
    # fetch one thing
    ('put the red ball in the basket', 'fetch_item', {'item': 'ball'}), ('pick up the sock', 'fetch_item', {'item': 'sock'}), ('put the toy car in the basket please', 'fetch_item', {'item': 'car'}),
    ('put the teddy bear in the basket', 'fetch_item', {'item': 'plush'}), ('put the brick in the basket', 'fetch_item', {'item': 'lbrick'}), ('pick up the rock', 'fetch_item', {'item': 'rock'}),
    ('put the banana in the basket', 'fetch_item', {'item': 'banana'}), ('pick up the cube and put it in the basket', 'fetch_item', {'item': 'cube'}),
    # where is
    ('where is my ball', 'where_is', {'item': 'ball'}), ("where's the sock", 'where_is', {'item': 'sock'}), ('can you find the teddy bear', 'where_is', {'item': 'plush'}),
    ('have you seen the toy car', 'where_is', {'item': 'car'}), ('where is the brick', 'where_is', {'item': 'lbrick'}), ('find my sock please', 'where_is', {'item': 'sock'}),
    ('look for the cube', 'where_is', {'item': 'cube'}), ('where is the rock', 'where_is', {'item': 'rock'}),
    # list / what's new
    ('what is on the floor', 'list_items', {}), ('what toys are there', 'list_items', {}), ('what do you see on the floor', 'list_items', {}), ('list the toys', 'list_items', {}),
    ("what's new today", 'what_new', {}), ('what has changed since yesterday', 'what_new', {}), ("what's missing", 'what_new', {}), ('anything moved', 'what_new', {}),
    # scan
    ('scan the house', 'scan_home', {}), ('make a 3d map of the home', 'scan_home', {}), ('map the house', 'scan_home', {}), ('please scan the room', 'scan_home', {}), ('look around the house', 'scan_home', {}),
    # go to
    ('go to the sofa', 'go_to', {'place': '沙发'}), ('go over to the table', 'go_to', {'place': '桌子'}), ('drive to the shelf', 'go_to', {'place': '架子'}), ('move to the chair', 'go_to', {'place': '椅子'}),
    ('go to the basket', 'go_to', {'place': '收纳筐'}), ('go to the plant', 'go_to', {'place': '盆栽'}),
    # charge
    ('go charge', 'charge', {}), ('go and recharge', 'charge', {}), ('time to charge', 'charge', {}), ('go to the charger', 'charge', {}),
    # people
    ('who is home', 'who_here', {}), ('is anyone home', 'who_here', {}), ('who do you see', 'who_here', {}), ('is there a child here', 'who_here', {}),
    ('remember this child, his name is Tom', 'enrol', {'newname': 'Tom'}), ('remember my face', 'enrol', {}), ('forget Tom', 'forget', {}), ('delete everything about Emma', 'forget', {}),
    # naming
    ('call this brick Bruno', 'name_item', {'newname': 'Bruno'}), ('name the ball Lucky', 'name_item', {'newname': 'Lucky'}),
    # status, privacy, small talk
    ('what are you doing', 'status', {}), ('how much battery do you have left', 'status', {}), ('are you busy', 'status', {}),
    ('are you recording', 'privacy', {}), ('are you always listening', 'privacy', {}), ('mute', 'privacy', {'mute': True}),
    ('hello', 'greet', {}), ('good morning', 'greet', {}), ('good evening', 'greet', {}), ('who are you', 'identity', {}), ("what's your name", 'identity', {}), ('what can you do', 'help', {}),
    # stop / cancel / resume / come here / language / profile
    ('never mind', 'cancel', {}), ('cancel that', 'cancel', {}), ('continue', 'resume', {}), ('carry on', 'resume', {}), ('come here', 'come_here', {}), ('come to me', 'come_here', {}),
    ('speak Chinese', 'set_lang', {'lang': 'zh'}), ('reply in English', 'set_lang', {'lang': 'en'}), ('what do you know about Tom', 'profile_show', {}),
]

COMMANDS_EN_HOLDOUT = [
    ('would you mind picking the toys up', 'tidy', {}), ('get all the toys off the floor', 'tidy', {}), ('I would like the floor cleared', 'tidy', {}), ('time to tidy up', 'tidy', {}),
    ('please put the building block into the basket', 'fetch_item', {'item': 'lbrick'}), ('get the teddy into the basket', 'fetch_item', {'item': 'plush'}), ('pick up that sock for me', 'fetch_item', {'item': 'sock'}),
    ('can you pick up the toy banana', 'fetch_item', {'item': 'banana'}), ('put the little ball in the basket', 'fetch_item', {'item': 'ball'}),
    ('where could my teddy bear be', 'where_is', {'item': 'plush'}), ('do you know where the ball is', 'where_is', {'item': 'ball'}), ("I can't find my sock", 'where_is', {'item': 'sock'}),
    ('where did the toy car go', 'where_is', {'item': 'car'}), ('can you look for the pebble', 'where_is', {'item': 'rock'}), ('where are my socks', 'where_is', {'item': 'sock'}),
    ('show me what is lying around', 'list_items', {}), ('what can you see on the floor right now', 'list_items', {}), ('what toys do you see', 'list_items', {}),
    ('has anything changed', 'what_new', {}), ('is anything missing', 'what_new', {}), ('what moved since last time', 'what_new', {}),
    ('please map the whole home', 'scan_home', {}), ('can you scan the house for me', 'scan_home', {}), ('explore the house', 'scan_home', {}),
    ('walk over to the couch', 'go_to', {'place': '沙发'}), ('head to the shelf', 'go_to', {'place': '架子'}), ('go near the table', 'go_to', {'place': '桌子'}), ('go over to the plant', 'go_to', {'place': '盆栽'}),
    ('go and charge yourself', 'charge', {}), ('please go to your charging dock', 'charge', {}), ('recharge now', 'charge', {}),
    ("who's here", 'who_here', {}), ('is somebody home', 'who_here', {}), ('are the kids here', 'who_here', {}),
    ('please remember my voice', 'enrol', {}), ('remember her, her name is Mia', 'enrol', {'newname': 'Mia'}), ('forget everyone', 'forget', {'all': True}), ('erase Tom', 'forget', {}),
    ('name this teddy Fluffy', 'name_item', {'newname': 'Fluffy'}), ('call it Rex', 'name_item', {'newname': 'Rex'}),
    ('what are you up to', 'status', {}), ('battery level please', 'status', {}), ('what is your status', 'status', {}),
    ('do you record us', 'privacy', {}), ('turn off the microphone', 'privacy', {'mute': True}), ('stop listening', 'privacy', {'mute': True}),
    ('hi there', 'greet', {}), ('good afternoon', 'greet', {}), ('introduce yourself', 'identity', {}), ('what are you able to do', 'help', {}), ('how do you work', 'help', {}),
    ('forget it', 'cancel', {}), ('nevermind', 'cancel', {}), ('keep going', 'resume', {}), ('go on', 'resume', {}), ('come over here', 'come_here', {}), ('come closer', 'come_here', {}),
    ('talk in Chinese', 'set_lang', {'lang': 'zh'}), ('switch to English', 'set_lang', {'lang': 'en'}), ('tell me about Emma', 'profile_show', {}),
]

# Chinese with English words in it, and English with Chinese words in it (what people do at home)
COMMANDS_MIX_DEV = [
    ('把 ball 放进筐里', 'fetch_item', {'item': 'ball'}), ('go to the 沙发', 'go_to', {'place': '沙发'}), ('帮我 find 一下 teddy bear', 'where_is', {'item': 'plush'}), ('去 charger 那里', 'go_to', {'place': '充电座'}),
    ('put the 袜子 in the basket', 'fetch_item', {'item': 'sock'}), ('tidy up 一下玩具', 'tidy', {}), ('收拾一下 the toys', 'tidy', {}), ('请 scan 一下家里', 'scan_home', {}),
    ('where is 我的小球', 'where_is', {'item': 'ball'}), ('把 plush toy 收好', 'fetch_item', {'item': 'plush'}), ('go charge 吧', 'charge', {}), ('去 sofa 那边', 'go_to', {'place': '沙发'}),
    ('what is 地上有什么', 'list_items', {}), ('who is 在家', 'who_here', {}), ('把玩具车 put in the basket', 'fetch_item', {'item': 'car'}), ('找一下 my sock', 'where_is', {'item': 'sock'}),
    ('go to the 桌子那边', 'go_to', {'place': '桌子'}), ('帮我 pick up the rock', 'fetch_item', {'item': 'rock'}), ('今天 what is new', 'what_new', {}), ('去 basket 那里', 'go_to', {'place': '收纳筐'}),
]
COMMANDS_MIX_HOLDOUT = [
    ('把 red ball 放到筐里', 'fetch_item', {'item': 'ball'}), ('walk over to the 沙发', 'go_to', {'place': '沙发'}), ('帮我 look for 一下 the car', 'where_is', {'item': 'car'}), ('回 charging dock 充电', 'charge', {}),
    ('put the 积木 away please', 'fetch_item', {'item': 'lbrick'}), ('clean up 一下房间', 'tidy', {}), ('请 map 一下 the house', 'scan_home', {}), ('teddy bear 在哪里', 'where_is', {'item': 'plush'}),
    ('请把 banana 捡起来', 'fetch_item', {'item': 'banana'}), ('go over to the 架子', 'go_to', {'place': '架子'}), ('地上有什么 toys', 'list_items', {}), ('谁 is home', 'who_here', {}),
    ('去 chair 那边', 'go_to', {'place': '椅子'}), ('把 sock 捡起来', 'fetch_item', {'item': 'sock'}), ('what changed 了吗', 'what_new', {}), ('我的 cube 在哪', 'where_is', {'item': 'cube'}),
]

# what people say in the room that is not for the robot, in English, with the words that make a robot misfire: stop, wait, find, go to, charge, come here, where is
CHATTER_EN = [
    'dinner is ready', 'did you finish your homework', 'where is the bus stop', 'I will stop by the store later', 'wait for me by the door', 'turn the television down please', 'can you find my phone',
    'the ball is red', 'put your shoes on', 'let me go to the bathroom', 'what time is it', 'I need to charge my phone', 'Tom come here', 'who is at the door', 'we should go to the park this weekend',
    'that is a nice toy', 'be careful with the stairs', 'I think the sofa is too big', 'it is going to rain tomorrow', 'what are we having for lunch', 'do not run in the house', 'where did I put my keys',
    'the news is on', 'stop it you two', 'please be quiet the baby is asleep', 'hold on I am on the phone', 'she is scanning the document', 'the table is set', 'good night sweetheart', 'my battery is almost dead',
]
# the same kind of sentences in Chinese-English mixture
CHATTER_MIX = ['今天 dinner 吃什么', '那个 ball 是我的', '快 go to bed', '你把 homework 写完了没有', '电视里的 news 好无聊', '等一下 wait for me', '我 stop by 超市买点东西', '把 TV 关掉']


# written after the sentence frames, and not tuned on
COMMANDS_EN_TEST = [
    ('could you get the toys off the floor', 'tidy', {}), ('let us clean up the mess', 'tidy', {}), ('pick the toys up for me', 'tidy', {}), ('I want the toys put away', 'tidy', {}), ('the floor needs tidying', 'tidy', {}),
    ('toss the sock in the basket', 'fetch_item', {'item': 'sock'}), ('bring the teddy bear to the basket', 'fetch_item', {'item': 'plush'}), ('could you put the toy banana away', 'fetch_item', {'item': 'banana'}),
    ('drop the rock in the basket', 'fetch_item', {'item': 'rock'}), ('move the red ball into the basket', 'fetch_item', {'item': 'ball'}), ('take the toy car to the basket', 'fetch_item', {'item': 'car'}),
    ('where is the teddy', 'where_is', {'item': 'plush'}), ('any idea where my ball is', 'where_is', {'item': 'ball'}), ('have you seen my socks', 'where_is', {'item': 'sock'}),
    ('I am looking for the building block', 'where_is', {'item': 'lbrick'}), ('where did you last see the car', 'where_is', {'item': 'car'}), ('could you tell me where the rock is', 'where_is', {'item': 'rock'}),
    ('tell me what is on the floor', 'list_items', {}), ('what are the things on the ground', 'list_items', {}), ('what toys can you see', 'list_items', {}),
    ('what is different from yesterday', 'what_new', {}), ('did anything move', 'what_new', {}), ('has anything gone missing', 'what_new', {}), ('is there anything new', 'what_new', {}),
    ('could you scan the whole house', 'scan_home', {}), ('go around and make a map', 'scan_home', {}), ('I want a 3d map of the apartment', 'scan_home', {}), ('map out the place', 'scan_home', {}),
    ('go to the shelf', 'go_to', {'place': '架子'}), ('please drive over to the sofa', 'go_to', {'place': '沙发'}), ('head over to the chair', 'go_to', {'place': '椅子'}), ('roll over to the table', 'go_to', {'place': '桌子'}),
    ('you need to charge', 'charge', {}), ('go back to your dock', 'charge', {}), ('please recharge', 'charge', {}), ('head to the charging station', 'charge', {}),
    ('is somebody at home', 'who_here', {}), ('who is in the house', 'who_here', {}), ('are there any children around', 'who_here', {}), ('do you see anyone', 'who_here', {}),
    ('learn my face', 'enrol', {}), ('remember this person', 'enrol', {}), ('her name is Mia, remember her', 'enrol', {'newname': 'Mia'}),
    ('forget Mia', 'forget', {}), ('please delete Tom', 'forget', {}), ('remove everything about Emma', 'forget', {}),
    ('call this plush Bear', 'name_item', {'newname': 'Bear'}), ("let's call the ball Bouncy", 'name_item', {'newname': 'Bouncy'}),
    ('how is your battery', 'status', {}), ('what are you working on', 'status', {}), ('are you charged', 'status', {}),
    ('are you listening to us', 'privacy', {}), ('do you keep recordings', 'privacy', {}), ('please mute yourself', 'privacy', {'mute': True}),
    ('hello there', 'greet', {}), ('hi', 'greet', {}), ('what is your name', 'identity', {}), ('what are you capable of', 'help', {}), ('what can you help me with', 'help', {}),
    ('forget about it', 'cancel', {}), ('never mind that', 'cancel', {}), ('please continue', 'resume', {}), ('resume', 'resume', {}),
    ('come over to me', 'come_here', {}), ('please come here', 'come_here', {}), ('please speak English', 'set_lang', {'lang': 'en'}), ('can you talk in Chinese', 'set_lang', {'lang': 'zh'}),
    ('what do you know about Mia', 'profile_show', {}), ('show me the profile of Tom', 'profile_show', {}),
]
COMMANDS_MIX_TEST = [
    ('把 teddy 放进 basket', 'fetch_item', {'item': 'plush'}), ('go to 沙发 please', 'go_to', {'place': '沙发'}), ('帮我 find the ball', 'where_is', {'item': 'ball'}), ('去 table 那边', 'go_to', {'place': '桌子'}),
    ('where is 小袜子', 'where_is', {'item': 'sock'}), ('clean up 玩具', 'tidy', {}), ('请 scan the house', 'scan_home', {}), ('what is on 地板上', 'list_items', {}), ('把 car 放进筐里', 'fetch_item', {'item': 'car'}),
    ('收拾 the toys 吧', 'tidy', {}), ('come here 一下', 'come_here', {}), ('go charge 好吗', 'charge', {}), ('put 积木 in the basket', 'fetch_item', {'item': 'lbrick'}), ('去 shelf 那里', 'go_to', {'place': '架子'}),
    ('请 put the sock 放进筐里', 'fetch_item', {'item': 'sock'}), ('找找 my teddy bear', 'where_is', {'item': 'plush'}),
]
