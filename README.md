# 小拾 xiaoshi-robot

一台在房间里自己找东西、夹起来、送到收纳筐的小机器人，以及它的 MuJoCo 物理仿真和实物验收方案。地上的东西可以是硬的、软的、形状不规则的，机器人事先不知道它们是什么、有多大。

硬件全部是能买到的开源套件：**LeKiwi 三全向轮底盘 + SO-101 机械臂（5 轴 + 夹爪）+ LD19 激光雷达**，外加前置和腕部两个深度相机。

在家里还能做的（第十四轮，**只在仿真里做了我们自己写的那部分，认人和语音识别在仿真里是替身/模拟**）：扫描家里建三维地图并导出，记住东西放在哪、换了场景后认出挪过的，靠头部相机分大人和小孩并在人靠近时放慢和停下，听中文命令（唤醒词、停止词、谁能让它做什么）。孩子的数据怎么处理：[`docs/privacy.md`](docs/privacy.md)。三维查看器：[`results/home_scan_seed1.html`](results/home_scan_seed1.html)（下载后用浏览器打开）。

第十五轮加了**声音**（同样只在仿真里做了我们自己写的那部分）：用麦克风阵列听声音从哪个方向来（4 麦中位误差约 8°）、用声纹（音色）认出是谁在说话、把方向、声纹和摄像头里的人对起来（电视的声音和录音不当成人、只凭声音拿不到“主人”权限）、**中文、英文和中英夹杂都能听懂、用提问的语言回答**，并给登记过的人建一张本机的、能看能删的画像（语言偏好、常在的时段、常叫它做的事；不存原话，孩子只存最少的）。仿真里没有声音，我们用自己写的房间声学和 espeak-ng 合成的中英文说话声把声音渲染出来再走真算法，**声学和声音都是仿真，不是实测**。

第十六轮把**相机从头部立杆挪到了机械臂上**（装在前臂上、向上翻到仰 32.5°，行进姿势下就朝前向上看，不用抬手臂，不要立杆；仿真里找人、分大人孩子、认人（替身识别器）、人和机器人同场干活都和立杆差不多，代价是手臂干活时约 12% 的时间看不到人，以及**相机会影响起身**（75 g 装上去，原来存着的起身动作向左侧倒就起不来了；已带着相机重搜，新场景上起身 23–24/24，机器人有前臂相机时自动用新动作；但侧倒时相机会压在地上，要加保护挡条；整机摔倒试验里还抓到并修了起身模式里一个等不完的等待，没有相机也会遇到），见 [`docs/status.md`](docs/status.md)）；接上了**大模型 API**（`sim/llm.py`，**不限哪一家**：Claude、任何 OpenAI 兼容的服务——DeepSeek、通义、Kimi、智谱、OpenAI、Azure……以及家里电脑上的 Ollama / LM Studio / vLLM，还能几个串起来；默认关，只发唤醒词之后这一句话的文字，孩子和认不出的人的话不发，动作仍要先口头确认；真服务没有测过）；唤醒词和停止词改成**在声音上直接听**（`sim/wakeword.py`，sherpa-onnx 的中英双语关键词模型，词用文字写进去；真录音上验证了流程，**真人喊“小拾小拾”有多灵没有测**）。

- 当前状态和已知限制：[`docs/status.md`](docs/status.md)；下一步计划：[`docs/roadmap.md`](docs/roadmap.md)
- 试验结果和实物验收表：[`docs/validation.md`](docs/validation.md)
- 要买的东西：[`docs/bom.md`](docs/bom.md)
- 原始数据：[`results/`](results/)（每次试验的结果和事件日志）

## 目录

| 路径 | 内容 |
|---|---|
| `sim/model.py` | 整机物理模型：带滚子的全向轮、底盘、SO-101、房间。硬件参数集中在顶部的 `P` |
| `sim/objects.py` | 物体库：方块、小球、玩具车、香蕉、L 形积木、随机形状的石头、袜子（软）、毛绒玩具（软）。每次出场尺寸和重量随机 |
| `sim/robot.py` | 驱动层接口：底盘速度、关节目标、雷达、里程计 |
| `sim/arm.py` | 5 轴机械臂逆运动学 |
| `sim/skills.py` | 抓取、搬运、放置，以及把贴着障碍的东西扒出来。下手位置只从相机量到的表面点里选，不用物体类别 |
| `sim/controller.py` | 自主控制器：建图、A* 规划、全向跟踪、防撞层、任务调度；只靠空洞簇看见的「看不清的东西」（确认两帧、端面看不出长度就换侧面再看）；雷达和里程计对不上（轮子转了车没跟上）时后退并记一个看不见的障碍。决策时不读仿真真值 |
| `sim/localization.py` | 雷达扫描匹配定位：里程计预测，对自建地图做相关匹配（稳健对数似然 + 里程计先验，墙按局部峰值描细）；分数突然掉下去时宽范围重找，且不写进地图 |
| `sim/lidar_models.py` | 雷达误差模型：高斯噪声；按 LD19 规格书的模型（每个表面固定偏差、1.5 cm 量化、混合点、野点） |
| `sim/perturb.py` | 敏感度扫描用的误差开关（摩擦、重量、舵机零位、相机外参、里程计、雷达、深度；透明物体深度图里没读数的比例和成片程度） |
| `sim/perception.py` | 前置深度相机：拟合地面、去掉家具，把剩下的点分成一件件东西，量轮廓、表面点和高度；地板上连成片没有读数的地方（透明塑料）单独找出来，只凭它估出近侧边缘、粗细、朝向和长度（`clear_detection`）；腕部深度相机：看两片夹爪之间有没有东西 |
| `sim/modes.py` | 模式管理层 `Supervisor`：自动任务 / 自主导航 / 遥控 / 充电 / 起身 / 待命 / 故障，以及谁能打断谁（摔倒 > 电量见底 > 操作员 > 电量偏低 > 任务结束）。还管 IMU 摔倒检测、电池记账、遥控的死人开关 |
| `sim/power.py` | 电池和充电模型：容量、耗电（算力常数 + 每个舵机的电流）、库仑计读数误差、恒流恒压充电（充电时仿真快进，物理时间不走） |
| `sim/dock.py` | 回充电座：规划到座前 0.4 m，倒车、按位姿估计修正横向和朝向，碰到触片就停，没接上就前进离开再来，最多 3 次 |
| `sim/righting.py` | 摔倒后起身：等静止、按 IMU 重力方向分类躺姿、执行对应的手臂+轮子动作、检查、重试；起来以后用整张地图重新定位（多视角一致才接受） |
| `sim/righting_maneuvers.json` | 起身动作库（每种躺姿一组 4 段的关节目标和轮速），由 `sim/search_righting.py` 搜出来 |
| `sim/righting_maneuvers_armcam.json` | 带前臂相机（75 g，盒子碰地）的起身动作库（第十六轮）：`righting.library_for(机器人)` 在机器人有前臂相机时自动用它，没有相机或相机在夹爪上用上面那套 |
| `sim/search_righting.py` | 起身动作的搜索工具：在带扰动的场景里（零位、摩擦、重量、落地姿态）用交叉熵法找，按新场景测成功率；`ARM_CAM=forearm_up`（加 `ARM_CAM_COLLIDE=1`）带着相机的重量和盒子搜，适应度里有“压相机”的惩罚；`init` 把已有的动作作为起点 |
| `sim/teleop_server.py` | 遥控的 TCP 接口（一行一个 JSON）：速度、手臂（关节或手端位置）、切模式、去某个点、状态 |
| `sim/teleop_client.py` | 终端键盘遥控客户端（连 `teleop_server.py`；w/s/a/d/q/e 开，1–4 手臂，t 接管、r 交还、c 去充电） |
| `sim/mapping3d.py` | 三维家庭地图（第十四轮）：4 cm 体素 + 对数几率，沿相机射线清空，墙 = 体素列上的长直线（RANSAC + 外侧检验），家具 = 三维连通块的盒子，下一个最佳视点规划 `ScanPlanner`，PLY 导出，不用任何库的网页查看器，对仿真真值的评估 |
| `sim/home.py` | `Home`：把三维地图和物件记录挂到前置相机的每一帧上（`Agent.depth_hook`，不挂的时候旧路径不变） |
| `sim/inventory.py` | 物件记录：颜色直方图 + 尺寸 → 种类猜测；跨次会话再识别（挪过位置的认得出是同一件）；「不见了」「还没核实」；按名字问「在哪」；命名；存本机 JSON，不存图片 |
| `sim/people.py` | 人和孩子：仿真里的人（五种身高、走、站、坐、追着机器人跑）；头部深度相机（立杆）；深度图检测、跟踪、身高 → 大人/小孩/不确定；激光雷达上会动的东西（相机看不到的方向）；登记簿（要大人同意、可删、只存特征向量）；安全层 `Guard`（放慢、停下、手臂不动）；**认人那一层是替身，见 `docs/privacy.md`** |
| `sim/voice.py`、`sim/voice_data.py`、`sim/voice_data_en.py` | 中文和英文语音（第十五轮起两种语言和夹杂）：唤醒词和停止词、模拟的语音识别（中文：同音字、声调、平翘舌、前后鼻音、掉字；英文：同音词、差一个音素的词、中国人说英文的口音、掉词；夹杂时另一种语言的词会被写成对方的字）、拼音/音素模糊匹配的理解、多轮对话、谁能让它做什么、执行器、可选大模型的接口、声纹登记；`voice_data.py` 是带标注的中文说法，`voice_data_en.py` 是英文和夹杂的（开发集、第一次看过的保留集、没调过的测试集） |
| `sim/lang.py` | 双语工具：语种判断、英文音素（CMU 词典）和音素距离匹配、英文唤醒词/停止词/意图/物品/地点/是否、句型（动词+东西）、拼音写的名字（`xiaoshi`）、英文识别错误模型 |
| `sim/acoustics.py` | 声学仿真：长方体房间（镜像声源法 3 阶 + 扩散尾巴）、2/4/6 麦阵列、espeak-ng 合成的中英文说话声、电视、机器人自己的风扇和轮子声、背景和麦克风自噪声 |
| `sim/doa.py` | 声源方向：STFT + SRP-PHAT，唤醒词前一秒做背景协方差扣除，延迟求和和 MVDR 波束形成；**文件名不叫 ssl.py，免得盖住标准库** |
| `sim/speaker.py` | 声纹：MFCC + 音高统计 → 在背景声音上训练的 LDA 投影；登记（要同意、只存向量）、识别（认出/不认识/拿不准）、每人分数归一化、陌生人匿名编号（只在内存、一天清掉）、忘掉 |
| `sim/assoc.py` | 声音和人对上号：方向 + 声纹 + 摄像头 → 谁说的（看得见的人 / 视野外的人 / 设备）；用听到的声音学会电视在哪；**只凭声音不给任何高于陌生人的权限，“主人”要摄像头里认得出的大人在声音的方向上、声纹不否定、方向上没有已知的电视/手机** |
| `sim/soundfront.py` | 把仿真里的一句话渲染成麦克风信号，走方向、波束、声纹、关联，交给 `VoiceSystem` |
| `sim/llm.py` | 大模型接口（第十六轮）：`Gateway`（默认关；只发唤醒词之后这一句话的文字，去人名/长数字/邮箱/网址；孩子和认不出的人的话不发；限额、熔断、不记原话的日志）+ 各家的接法：`ClaudeProvider`（Anthropic SDK）、`OpenAICompatProvider`（OpenAI 聊天接口，标准库 HTTPS，`PRESETS` 里有 DeepSeek、通义、Kimi、智谱、硅基流动、豆包、混元、Groq、OpenRouter、Together、Mistral、xAI、Gemini 兼容、Ollama、LM Studio、vLLM、llama.cpp，另有 Azure 和任意地址）、`ChainProvider`（串联）、`make_provider(名字, model=…)`、`from_env()`（环境变量 `XIAOSHI_LLM=ollama:型号,deepseek:型号`）、`from_config('llm.json')`（把选择写在文件里，文件里不许放密钥）；模型只能把话**改写**成机器人听得懂的一句或回一句闲聊，机器人自己再解析、先口头确认，记人/忘人/静音/换语言/停止永远不从模型接受 |
| `sim/wakeword.py` | 唤醒词和停止词在声音上直接听（第十六轮）：`SherpaWake`（sherpa-onnx 流式关键词识别，中英双语 3 M 参数；词用文字写进去；冷却时间；机器人自己说话时只压唤醒词不压停止词）、`keyword_lines`、`evaluate`（在你自己的录音上扫阈值，量命中和每小时误唤醒）、`fetch_model` |
| `sim/test_llm.py`、`sim/test_wake.py`、`sim/test_armcam.py` | 第十六轮的测试：大模型接口 145 项（真 SDK 加假网络）；唤醒词 16 项（真模型加真录音）；手臂相机的位姿、认人、起身（相机重量扫描）、麦克风装低了的声源方向 |
| `sim/profiles.py` | 人物画像：只存计数（语言、时段、房间、常叫它做的事、自己的东西），孩子最少，能看（中英文卡片）、能删、衰减 |
| `sim/store.py` | 人脸、声纹、画像三个库共用的两件事：文件整份写入（先写临时文件再换，断电不留半份）、读回日志（“忘掉”也清掉前几次启动留下的名字，最多 2000 条） |
| `sim/sound_lab.py`、`sim/test_sound.py`、`sim/test_lang.py`、`sim/test_assoc.py`、`sim/test_day2.py` | 第十五轮台架：声源方向、阵列误差、声纹；中英文听懂；声音和人对上号、画像；整机的第二天 → `results/sound_*.json`、`results/lang.json` |
| `sim/test_home.py` | 第十四轮台架：`scan`、`inventory`、`people_static`、`safety`、`identity`、`voice`、`day`、`export` → `results/home_*.json` |
| `sim/test_modes.py` | 模式、充电座、遥控、导航、摔倒的台架 → `results/modes_*.json` |
| `sim/test_righting.py` | 起身台架：每种躺姿的动作在新场景里的成功率，以及 28 次推倒 → `results/righting.json`；`unit`：起身前“等静止”的 6 项检查（不跑物理） |
| `sim/episode.py` | 一次完整试验（随机布局 → 自主运行 → 成绩单）；`run_supervised` 是同一个试验放在模式管理层下（带电池、充电座、摔倒、操作员） |
| `sim/run_trials.py` | 批量随机试验 → `results/trials.json` |
| `sim/test_base.py`、`sim/test_grasp.py`、`sim/test_perception.py`、`sim/test_pull.py` | 底盘、抓取、相机、扒出来的台架测试 |
| `sim/test_localization.py` | 定位台架：录一局真实轨迹，离线换不同雷达误差模型重放，一次约 20 秒 |
| `sim/test_clear.py` | 透明瓶台架：空洞簇给出的位置、粗细、朝向、长度准不准（远处、近处、下手站位，机身转着看），空房间误报，只凭空洞抓一次 → `results/clear_bench.json` |
| `sim/run_clear.py` | 透明瓶单独在房间里：两个办法分开开关的对照、强制各种朝向、更苛刻的深度图、没参与开发的新种子（含整机难材质）→ `results/clear_bottle.json` |
| `sim/render_video.py` | 把一次试验渲染成录像 |
| `third_party/so101/` | SO-101 官方 MuJoCo 模型（Apache-2.0，来自 TheRobotStudio/SO-ARM100） |
| `web/` | 结果页面的模板和生成脚本；`web/concept-v0.1/` 是第一版概念演示，不是物理仿真 |
| `web/dist/` | 页面的构建产物，由 `web/build_page.py` 生成：`index.html`（完整文档，可直接用浏览器打开）、`artifact.html`（同一份页面去掉外层文档标签，线上发布的逐字节就是它，平台再套自己的外壳）、两段录像和封面图（与 `results/` 里的是同一份文件） |

## 运行

```bash
pip install mujoco numpy scipy imageio imageio-ffmpeg pillow pypinyin cmudict
sudo apt-get install espeak-ng      # 第十五轮：合成说话声，只有声音台架用
pip install anthropic sherpa-onnx sentencepiece   # 第十六轮，可选：Claude 那一路要 anthropic（其他家只用标准库）；唤醒词的真模型测试要后两个（没装就跳过，不算通过）
cd sim
python test_base.py          # 底盘台架，几十秒
python test_grasp.py 10      # 抓取台架，每种物体 10 次 × 两种夹爪摩擦
python test_pull.py 2        # 扒出来台架：八种物体离书架 2、4、6 cm
python test_perception.py     # 相机台架：八种物体 × 四个距离，挨着放的两件，贴着家具，空房间误报
python episode.py 4          # 第 4 号随机布局跑一次，打印事件日志和成绩
python run_sensitivity.py 2  # 敏感度：每次只让一项误差变差，标准条件 10 局（--only 名字,名字 --out 文件 只跑其中几项）
python test_localization.py record 5    # 录第 5 号布局的轨迹（一局，约 3 分钟）
python test_localization.py bench 5,9   # 两套定位 × 六种雷达误差模型 -> results/loc_bench.json
python profile_compute.py    # 机器人自己的算法各占多少 CPU
python test_clear.py all      # 透明瓶台架，几分钟 -> results/clear_bench.json
python run_clear.py ab 4      # 透明瓶单独在房间里（ab / forced / robust / heldout 四种：两个办法的对照、强制各种朝向、更苛刻的深度图、没参与开发的新种子），4 个进程 -> results/clear_bottle.json
python test_home.py scan 1      # 三维扫描：下一个最佳视点 / 覆盖巡视各一遍，对仿真真值评估（PROCS=2）-> results/home_scan.json
python test_home.py export 1    # 一次扫描导出点云（PLY）、能直接打开的三维查看器（HTML）和数字 -> results/home_scan_seed1.*
python test_home.py inventory 1 2 3 4 5 6   # 物件记录：换了场景（挪了两件、拿走一件、新来一件）以后再识别 -> results/home_inventory.json
python test_home.py people_static 150       # 人：站在 1–3 m 处能不能检出、身高量得准不准、分成大人还是小孩 -> results/home_people_static.json
python test_home.py pairs       # 孩子挨着大人站（并排/一前一后，间隔 10–70 cm）：分不分得开、孩子会不会被当成大人 -> results/home_people_pairs.json
python test_home.py identity    # 登记簿和认人的阈值（替身识别器，只测我们自己的逻辑）-> results/home_people_identity.json
python test_home.py safety 1 2  # 人和机器人干活同场：成年人走过、孩子冲过来、幼儿在筐边、一家三口；安全层开/关 -> results/home_people_safety.json
python test_home.py voice       # 中文命令：加噪声扫描、没喊唤醒词的闲聊、停止词、权限表、多轮对话、大模型接口 -> results/home_voice.json
python test_home.py day 3       # 一整天连起来：语音、扫描、登记孩子、孩子冲过来、停下和继续、忘掉 -> results/home_day.json
python test_sound.py ssl 100      # 声源方向：2/4/6 麦、混响、电视、机器人自己在开、孩子、两个人同时说话 -> results/sound_ssl.json
python test_sound.py ssl_calib 100  # 阵列自己的误差：麦克风位置偏、增益差 -> results/sound_ssl_calib.json
python test_sound.py train_voice    # 训练声纹的投影（60 个背景声音 + 20 个校准）-> results/sound_voicemodel.npz
python test_sound.py voiceprint     # 声纹：认对、认错、陌生人、电视、机器人在开、远处、两种语言、声音很像的两个人 -> results/sound_voiceprint.json
python test_lang.py lang            # 中文、英文、夹杂：听懂（开发集/保留集/测试集）、口音、只会中文的机器人、唤醒词和停止词、闲聊、多轮对话 -> results/lang.json
python test_armcam.py pose       # 手臂相机：离地多高、朝哪、机器人由关节读数算出的位姿差多少（立杆/前臂/潜望镜）-> results/armcam_pose.json
CAMS=mast,forearm,periscope python test_armcam.py faces 60   # 手臂相机认人：登记过的人 1–3 m 外，叫对名字的比例和用时（替身识别器）-> results/armcam_faces.json
python test_armcam.py righting   # 相机 0–75 g 装在前臂上，存着的起身动作还灵不灵 -> results/armcam_righting.json
RIGHTING_LIB=righting_maneuvers_armcam.json python test_armcam.py righting_camera 24   # 相机重量和盒子碰地：起身成功、相机压在地上的力和时间、离机身多近（RIGHTING_HARSH=1 苛刻扰动）
FALL_CAM=forearm_up python test_modes.py fall 16   # 整机摔倒试验带前臂相机（falljob 同理）-> results/modes_fall_forearm_up.json
python test_armcam.py job 12       # 相机的 75 g 对不摔倒的取放任务有没有影响：12 个布局有/无相机，加布局 4 的相机重量扫描（约半小时，4 核）-> results/armcam_job.json
python test_armcam.py mic        # 麦克风阵列离地 0.5/0.3/0.2 m：声源方向 -> results/armcam_mic_height.json
CAM=forearm python test_home.py people_static 150   # 同上台架，相机装在前臂（CAM=periscope|forearm_up 也行）；CAM=forearm python test_home.py safety 1：同场干活
python test_llm.py               # 大模型接口：145 项检查，不联网 -> results/llm.json；有密钥时 XIAOSHI_LLM=deepseek:型号 python test_llm.py live 跑六句真的（不设就用 Claude）
python test_wake.py              # 唤醒词/停止词：真关键词模型 + 真录音（需要 pip install sherpa-onnx sentencepiece，python test_wake.py fetch 下模型）-> results/wake.json
python test_assoc.py assoc 200      # 声音和人对上号：200 个房间、电视、录音重放、访客 -> results/sound_assoc.json
python test_assoc.py profile        # 人物画像：一个编出来的四口之家 14 天，隐私检查 -> results/sound_profile.json
MUJOCO_GL=egl python test_day2.py 5 # 整机的第二天：电视、厨房里的声音、访客、录音、孩子 -> results/sound_day2.json
python test_modes.py dock 10  # 充电座：10 个随机起点倒车对接（PROCS=3 用 3 个进程），-> results/modes_dock.json
python test_modes.py fall 16  # 摔倒：16 次随机位置、方向、力度的推倒，看能不能起来、起来后位置对不对
python test_modes.py nav 6    # 自主导航：6 个随机布局各 6 个可达目标和 2 个到不了的目标
python test_modes.py teleop   # 遥控：死人开关、安全层开/关、手臂指令、从自动任务接管再交还
python test_modes.py lowbatt 2  # 整套任务带低电量：软停→充电→续作，电量紧急线
python test_modes.py server   # 遥控 TCP 服务器：真的连一个 socket 去开机器人
python test_modes.py explore 3  # 自主导航的巡视建图（只看不捡）
python test_modes.py dockfail   # 充电座位置和机器人以为的偏 0/8/15/25 cm：小偏差照样对上，大偏差干净地失败
python test_righting.py       # 起身台架（每种躺姿 24 个新场景 + 推倒扫描），约 5 分钟 -> results/righting.json
python test_righting.py unit  # 起身模式里“等机器人静止 3 秒”的 6 项检查，不跑物理，1 秒（旧代码 3/6：机器人躺下后自己再滑 3° 就永远等下去）
MUJOCO_GL=egl python render_video.py modes 3 ../results/modes.mp4   # 一次任务里接管、摔倒、低电量充电连着发生的录像
python run_trials.py 2       # 全部 56 次随机试验，2 个进程（双核约 2–3 小时，渲染相机图像占了一半时间）；中断后重跑会接着做
MUJOCO_GL=egl python render_video.py 4    # 录像，需要能离屏渲染的环境
python ../web/build_page.py  # 用 results/ 里的数字重新生成页面和 docs/validation.md
```

### 把唤醒词、识别器、大模型接到真机上（第十六轮）

仿真里没有声音，下面这段是真机上的接法（**没有在真机上跑过**，接口都有测试，见 `test_llm.py`、`test_wake.py`）：

```python
from voice import VoiceSystem, TextASR
from llm import Gateway, make_provider, from_env   # 不传 llm 就是完全离线，只用本机能听懂的
from wakeword import SherpaWake                  # pip install sherpa-onnx sentencepiece；python test_wake.py fetch 下模型

vs = VoiceSystem(sup=sup, home=home, ppl=ppl, asr=TextASR(), llm=Gateway(make_provider('deepseek', model='…')))   # asr: 真识别器写下的字原样交进来；各家的密钥在各自的环境变量里；孩子和认不出的人的话不会发出去
# 别的接法：make_provider('claude') / ('ollama', model='…') / ('openai-compatible', base_url='…', model='…')；from_env()（读 XIAOSHI_LLM）；from_config('llm.json')；Gateway([本机, 云]) 串联
# VoiceSystem(…, today=lambda: datetime.date.today().toordinal())   # 家里的日历日（跟着系统时区）：陌生人的匿名声音在家里的午夜清掉；不给就按仿真的钟算
kws = SherpaWake()                               # 常开，在机器人本机上跑
while True:
    for e in kws.feed(mic_chunk):                # mic_chunk: 波束形成之后的 16 kHz 单声道，一小块一小块地喂
        if e['kind'] == 'stop': vs.hear('停', kws='stop')                       # 停止词不要唤醒词，也不经过任何网络
        else:                                                                    # 唤醒词：这时才开始录这一句话、交给识别器
            text = my_recogniser.listen_until_silence()                          # 本机识别器或别的，自己接
            vs.hear(text, ident=front.resolve(...), kws=True)                    # 声音这一侧的“谁说的”见 soundfront.py
    # 机器人自己说话时：kws.suppress(until_t) 只压唤醒词；回声消除要在麦克风芯片上做
```

开发环境是 MuJoCo 4.0.1、Python 3、无显示器的 Linux。带窗口的交互查看器（`python -m mujoco.viewer`）没有在这个环境里试过。

## 哪些是真物理，哪些是假设

**真物理和真算法**：轮子、滚子和地面的接触摩擦；舵机力矩上限（STS3215 12 V 堵转 2.94 N·m）；机械臂质量、惯量、限位；夹爪与物体的接触；雷达射线投射（名义 1 cm 噪声，另按 LD19 规格书的误差模型测过）；带轮径误差和打滑的里程计；雷达扫描匹配定位；只从渲染深度图里分出一件件东西、量轮廓和高度，不用已知类别或尺寸，也不用训练好的模型。控制器做决策时不读仿真真值。

**假设**（每条在验收表里都有对应实测项）：

- 深度图的毛病按规律近似：噪声随距离增长；黑色表面约一半、反光表面约四分之一成片没有读数；透明塑料约 60% 没有读数、30% 读到后面的地板、10% 读到瓶身（逐像素独立）；深度边缘有飞点。比例都是估的，要在实物上量。**透明瓶这一项最需要实测**：找到它靠的就是「没有读数的像素连成片」，真瓶子带瓶盖、标签、在强光或背光下的比例未必是这样（`run_clear.py robust` 测了成片、更少空洞的情形）。
- 雷达误差按 LD19 规格书的上限估：0.3–12 m 每个表面一个固定偏差（±45 mm 内），1.5 cm 量化，混合点和 1% 野点；没有按距离、颜色、入射角变化的规律，0.3 m 以内规格书不保证准，角度零位误差（规格 ±2°）没有加。防撞层对雷达点留的 2 cm 余量（`LIDAR_BIAS_MARGIN`）假设真机做了距离标定。
- 机器人自己手臂的像素直接屏蔽（真机按运动学和标定算）。
- 软物体是多段刚体加弹性铰链和柔性接触的近似，能复现压扁、垂挂和粘手，不是布料仿真。
- 开机时停在充电座上，起点已知；收纳筐大致位置事先已知，精确位置由相机看到后修正。
- 底盘、电池等的重量是估的；里程计误差和深度噪声的大小是按常见硬件估的。
- **电池和充电**（第十三轮）：容量 50 Wh、算力和传感器耗电 10 W、每个舵机的电流（待机、铜损、做功/效率）、电池板读数的固定偏差（1σ 3%）、3 A 恒流到 80% 再恒压收尾，都是按数据手册和常见硬件估的（`sim/power.py`）。充电时仿真把时间快进，物理时间不走。
- **充电座和后部充电探针**：设计出来的，没造过。「接上」在仿真里是探针几何上碰到触片并保持 1 秒，没有接触电阻、弹簧力、氧化。充电座位置事先已知（装的时候量好写进 `sim/model.py`），挪了要重新量。
- **摔倒和起身**：IMU 读数是重力方向加噪声（0.25°）和固定偏差（0.5°），没有加速度干扰；起身的手臂加轮子动作是在仿真里搜出来的，要 30 rad/s² 的关节加速度，**实物没试过**；机身上只有雷达头和立柱是实体，相机支架、线缆、电池位置没建。
- **遥控和导航**：网络延迟、丢包没有建模；导航只在 4×3 m 单房间里测过，没有会动的障碍物。
- **三维地图和物件记录**（第十四轮）：地图只记前置相机（离地 12 cm、向下 25°）看得到的面，所以家具只有下面 0.3–0.6 m 是实的，高处和顶面靠盒子估；靠墙那一面的深度看不到；盒子比真的大 4–14 cm（体素和噪声）。只在 4×3 m 单房间里测过，多房间没测。物件种类靠颜色直方图和尺寸，仿真里八种东西颜色各不相同，**这是最好情况**，黑色和反光的东西没有颜色，认不出种类；两个一模一样的东西分不开。
- **人和孩子**（第十四轮）：仿真里的人是五种身高的刚体人形（躯干、头、两腿、两臂），按脚本走、站、坐、追着机器人跑，走路会绕开机器人（脚本里写的规则）；真的孩子不会这么听话。头部相机和立杆是新加的硬件，**立杆的重量和重心仿真里没算**。大人/小孩只看身高（和坐着时的肩宽）：约 11 岁以下算小孩，12 岁的算大人；坐着的大人会被分成小孩或不确定（保守）。**认出是哪个孩子那一层是替身**（给每个人画一个固定的特征向量、按距离和朝向加噪声），只能测登记簿、阈值、同意和删除这些我们自己写的逻辑，不能说明真的人脸模型认得准不准。
- **声音**（第十五轮）：**仿真里没有声音，我们自己写了房间声学（镜像声源法，长方体，家具不挡声）、麦克风阵列、espeak-ng 合成的说话声（几种合成器音色 + 音高 + 声道长度，比真人更容易分开）、电视和机器人自己的噪声，再把渲染出来的麦克风信号交给真的方向、波束、声纹算法。** 所有声压级（说话 62 dB、电视 55–70 dB、风扇 38 dB）都是估的；声纹是经典特征 + LDA，不是神经网络；识别器和英文口音是模拟的。真阵列、真声音、真电视、真孩子，都没测。
- **语音**（第十四轮）：仿真里没有声音，语音识别是模拟的（同音字、声调、平翘舌、前后鼻音、掉字，孩子和远处更差），后面的理解、对话、权限、执行是真代码。真麦克风阵列、客厅里的电视声、六岁孩子的口音、TTS 的声音、真大模型怎么回答，都没测。

## 参考

- LeKiwi：https://github.com/SIGRobotics-UIUC/LeKiwi
- SO-101：https://github.com/TheRobotStudio/SO-ARM100
- LeRobot 的 LeKiwi 文档：https://www.mintlify.com/huggingface/lerobot/robots/lekiwi
- 给 LeKiwi 加装激光雷达的实测：https://foxglove.dev/blog/upgrading-the-lekiwi-into-a-lidar-equipped-explorer
