# 小拾 xiaoshi-robot

一台在房间里自己找东西、夹起来、送到收纳筐的小机器人，以及它的 MuJoCo 物理仿真和实物验收方案。地上的东西可以是硬的、软的、形状不规则的，机器人事先不知道它们是什么、有多大。

硬件全部是能买到的开源套件：**LeKiwi 三全向轮底盘 + SO-101 机械臂（5 轴 + 夹爪）+ LD19 激光雷达**，外加前置和腕部两个深度相机。

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
| `sim/episode.py` | 一次完整试验（随机布局 → 自主运行 → 成绩单） |
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
pip install mujoco numpy scipy imageio imageio-ffmpeg pillow
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
python run_trials.py 2       # 全部 56 次随机试验，2 个进程（双核约 2–3 小时，渲染相机图像占了一半时间）；中断后重跑会接着做
MUJOCO_GL=egl python render_video.py 4    # 录像，需要能离屏渲染的环境
python ../web/build_page.py  # 用 results/ 里的数字重新生成页面和 docs/validation.md
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

## 参考

- LeKiwi：https://github.com/SIGRobotics-UIUC/LeKiwi
- SO-101：https://github.com/TheRobotStudio/SO-ARM100
- LeRobot 的 LeKiwi 文档：https://www.mintlify.com/huggingface/lerobot/robots/lekiwi
- 给 LeKiwi 加装激光雷达的实测：https://foxglove.dev/blog/upgrading-the-lekiwi-into-a-lidar-equipped-explorer
