# D1 Platform：第 8200 轮 checkpoint

`model_8200.pt` 是用户要求停训时最新保存的 NP3O checkpoint，包含模型和优化器状态。
模型约 9.1 MB，直接保存在 Git 中；下载后在本目录运行 `sha256sum -c SHA256SUMS` 校验。
GitHub 不允许这个公开 fork 上传新的 LFS 对象，因此仅此 checkpoint 使用普通 Git 保存。

**代码中的上台课程已改为从 30 cm 开始，但这个模型尚未按新课程续训。**
该 checkpoint 的实际训练高度范围为 5 cm–1 m；当前代码的上台范围为 30 cm–1 m。
它是最新保存的模型，不是按评估选出的最佳模型，也没有第 8200 轮的完整固定高度评估。

- 实际训练配置：`training_env.yaml`、`training_agent.yaml`。
- 轮数、校验值、reward 权重、版本和机器人来源：`manifest.json`。
- 机器人来源：`DDTRobot/ddt_ros2_control`，`compress_v1` 分支，提交
  `9e45c8cea6a7517ae2df7064e2030391c382c161`。
- 后轮上台权重：提前避墙惩罚 `3.645`，抬轮奖励 `0.390625`，干净落台奖励 `1.0`，碰墙惩罚系数 `0.5`。

在配置好 Isaac Lab 环境并安装本项目后，从仓库根目录运行：

```bash
python scripts/np3o/play.py --task DDT-Velocity-Platform-D1-NP3O-v0 \
  --num_envs 16 --checkpoint checkpoints/d1_platform_rear_clearance/model_8200.pt
```

这条回放命令使用当前代码的 30 cm 起步课程。保存的 YAML 记录原训练设置及机器上的路径，
用于核对训练来源；跨机器运行时应使用本地安装和机器人模型路径。

训练、自动监测和补充评估均已按用户要求停止。新的 30/40 cm 成对基线尚未完成，不能将旧协议数据
直接用于新协议的自动调参。
