# Avatar Agent

Avatar Agent 是一个基于AI的实时音视频交互系统，集成了语音识别、自然语言处理、语音合成等功能，支持通过Web界面进行实时音视频对话。

## 功能特点

- 实时音视频通信（基于WebRTC）
- 语音活动检测（VAD）
- 语言唤醒检测（KWS）
- 自然语言处理（LLM）
- 自动语音识别（ASR）
- 大语言模型对话（兼容OpenAI API）
- 文本转语音（TTS）
- Web界面交互（基于Gradio）
- 可配置的多模态交互流程

## 技术栈

- 后端框架：FastAPI、Uvicorn
- 前端界面：Gradio
- 实时通信：WebRTC (aiortc)
- 语音处理：Silero VAD、XiaoYun KWS SenseVoice ASR、CosyVoice TTS
- 自然语言处理：OpenAI兼容API
- 配置管理：Dynaconf、Pydantic
- 日志管理：Loguru
- 模型管理：ModelScope
- 数值计算：NumPy、PyTorch

## 安装步骤

### 1. 克隆仓库

```bash
git clone <仓库地址>
cd avatar_agent

# 更新子模块
git submodule update --init --recursive
```

### 2. 创建虚拟环境

```bash
conda create --name avatar-dify python=3.11
conda activate avatar-dify
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### MuseTalk必要依赖安装

##### 安装 MMLab 生态系统包：
```bash
pip install --no-cache-dir -U openmim
mim install mmengine
mim install mmcv==2.0.1
mim install mmdet==3.1.0
mim install mmpose==1.1.0
```

##### 设置安装 FFmpeg

ubuntu 系统下安装
```bash
apt update
apt install ffmpeg -y
```

conda 环境下安装
```bash
conda install -c conda-forge ffmpeg -y
```

##### 下载权重

linux系统：
```bash
sh ./download_weights.sh
```

权重组织结构
```text
./models/
├── musetalk
│   └── musetalk.json
│   └── pytorch_model.bin
├── musetalkV15
│   └── musetalk.json
│   └── unet.pth
├── syncnet
│   └── latentsync_syncnet.pt
├── dwpose
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── sd-vae
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── whisper
    ├── config.json
    ├── pytorch_model.bin
    └── preprocessor_config.json
    
```

### 4. 生成SSL证书（用于HTTPS）

```bash
bash scripts/create_ssl_certs.sh
```

## 配置说明

配置文件位于 `config/config.yaml`，主要配置项包括：

- 服务设置（host、port、SSL证书）
- 模型根目录
- 处理器配置（RTC客户端、VAD、ASR、LLM、TTS等）
- LLM模型设置（模型名称、API地址、API密钥、系统提示词）
- TTS模型设置（模型名称、语速、采样率等）

可根据需要修改配置文件，或通过环境变量覆盖配置。

## 使用方法

### 启动服务

```bash
# 在运行命令前设置PYTHONPATH
export PYTHONPATH=/mnt/19d1c476-ff03-4b9b-8605-718608679932/human/avatar_agent:$PYTHONPATH
python src/app.py --config config/config.yaml
```

可选参数：
- `--host`：指定服务地址（默认从配置文件读取）
- `--port`：指定服务端口（默认从配置文件读取）
- `--config`：指定配置文件路径
- `--env`：指定配置环境（默认：default）

### 访问界面

服务启动后，通过浏览器访问 `https://<host>:<port>` 即可使用Web界面进行交互。

## 项目结构

```
avatar_agent/
├── config/                 # 配置文件目录
├── src/
│   ├── chat_engine/        # 聊天引擎核心
│   ├── handlers/           # 各种处理器（ASR、LLM、TTS等）
│   ├── service/            # 服务相关（RTC、Web服务等）
│   ├── engine_utils/       # 工具函数
│   └── app.py              # 应用入口
├── scripts/                # 脚本文件
├── requirements.txt        # 依赖列表
└── .gitignore              # Git忽略文件
```

## 注意事项

1. 首次运行时会自动下载所需模型，可能需要较长时间
2. 部分模型需要网络访问权限
3. 实时音视频通信可能需要配置TURN服务器以支持跨网络通信
4. LLM功能需要有效的API密钥，请在配置文件中更新`api_key`

## 许可证

[请在此处填写许可证信息]
