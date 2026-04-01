# 模型文件

本目录用于存放本地模型文件（可选）。

## 下载模型（如果有网络）

模型会自动从 HuggingFace 下载。

## 离线使用方案

### 方案1：复制模型文件

1. 从有网络的机器复制 `models--bert-base-chinese.tar.gz` 和 `models--bert-base-uncased.tar.gz`
2. 解压到项目根目录：
   ```bash
   mkdir -p local_models
   tar -xzvf models--bert-base-chinese.tar.gz -C local_models/
   tar -xzvf models--bert-base-uncased.tar.gz -C local_models/
   ```
3. 修改 `configs/config.yaml` 设置 `local_model_dir: "./local_models"`

### 方案2：使用镜像

如需从其他镜像站下载，请参考 HuggingFace 镜像。