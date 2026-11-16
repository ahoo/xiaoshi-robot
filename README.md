# 小拾 xiaoshi-robot

一台在房间里自己找东西、夹起来、送到收纳筐的小机器人，以及它的 MuJoCo 物理仿真和实物验收方案。

硬件全部是能买到的开源套件：**LeKiwi 三全向轮底盘 + SO-101 机械臂（5 轴 + 夹爪）+ LD19 激光雷达**。

- 试验结果和实物验收表：[`docs/validation.md`](docs/validation.md)
- 要买的东西：[`docs/bom.md`](docs/bom.md)
- 原始数据：[`results/`](results/)（每次试验的结果和事件日志）

## 目录

| 路径 | 内容 |
|---|---|
| `sim/model.py` | 整机物理模型：带滚子的全向轮、底盘、SO-101、房间、物体。硬件参数集中在顶部的 `P` |
| `sim/robot.py` | 驱动层接口：底盘速度、关节目标、雷达、里程计 |
| `sim/arm.py` | 5 轴机械臂逆运动学 |
| `sim/skills.py` | 抓取、搬运、放置 |
| `sim/controller.py` | 自主控制器：雷达建图、A* 规划、全向跟踪、任务调度。传感器噪声在顶部 |
| `sim/episode.py` | 一次完整试验（随机布局 → 自主运行 → 成绩单） |
| `sim/run_trials.py` | 批量随机试验 → `results/trials.json` |
| `sim/test_base.py`、`sim/test_grasp.py` | 底盘和抓取的台架测试 |
| `sim/render_video.py` | 把一次试验渲染成录像 |
| `third_party/so101/` | SO-101 官方 MuJoCo 模型（Apache-2.0，来自 TheRobotStudio/SO-ARM100） |
| `web/` | 结果页面的模板和生成脚本；`web/concept-v0.1/` 是第一版概念演示，不是物理仿真 |

## 运行

```bash
pip install mujoco numpy scipy imageio imageio-ffmpeg pillow
cd sim
python test_base.py          # 底盘台架，几十秒
python test_grasp.py 10      # 抓取台架，每种物体 10 次 × 两种夹爪摩擦
python episode.py 4          # 第 4 号随机布局跑一次，打印事件日志和成绩
python run_trials.py         # 全部 40 次随机试验（双核约 45 分钟）
MUJOCO_GL=egl python render_video.py 4    # 录像，需要能离屏渲染的环境
python ../web/build_page.py  # 用 results/ 里的数字重新生成页面和 docs/validation.md
```

开发环境是 MuJoCo 4.0.1、Python 3、无显示器的 Linux。带窗口的交互查看器（`python -m mujoco.viewer`）没有在这个环境里试过。

## 哪些是真物理，哪些是假设

**真物理**：轮子、滚子和地面的接触摩擦；舵机力矩上限（STS3215 12 V 堵转 2.94 N·m）；机械臂质量、惯量、限位；夹爪与物体的接触；雷达射线投射。

**假设**（每条在验收表里都有对应实测项）：

- 定位用「真实位姿 + 1.5 cm / 1° 噪声」代替激光定位算法。
- 识别用「视野内且无遮挡就认出，位置带噪声」代替识别模型。
- 收纳筐位置事先已知。
- 底盘、电池等的重量是估的。

## 参考

- LeKiwi：https://github.com/SIGRobotics-UIUC/LeKiwi
- SO-101：https://github.com/TheRobotStudio/SO-ARM100
- LeRobot 的 LeKiwi 文档：https://www.mintlify.com/huggingface/lerobot/robots/lekiwi
- 给 LeKiwi 加装激光雷达的实测：https://foxglove.dev/blog/upgrading-the-lekiwi-into-a-lidar-equipped-explorer
