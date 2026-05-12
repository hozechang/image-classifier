"""
图像分类推理网站 - 上传 .pth 模型 + 图片 → 输出 Top-2 标签及概率
支持摄像头拍照输入
"""
import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os
from pathlib import Path

# ---------- ImageNet 1000 类标签 ----------
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
IMAGENET_LABELS = None  # lazy load


def load_imagenet_labels():
    """加载 ImageNet 1000 类标签"""
    global IMAGENET_LABELS
    if IMAGENET_LABELS is not None:
        return IMAGENET_LABELS
    try:
        import urllib.request
        with urllib.request.urlopen(IMAGENET_LABELS_URL) as f:
            IMAGENET_LABELS = [line.decode().strip() for line in f.readlines()]
    except Exception:
        # fallback: 生成占位标签
        IMAGENET_LABELS = [f"Class_{i}" for i in range(1000)]
    return IMAGENET_LABELS


# ---------- 支持的模型架构 ----------
ARCHITECTURES = {
    "ResNet-18": lambda n: models.resnet18(num_classes=n),
    "ResNet-34": lambda n: models.resnet34(num_classes=n),
    "ResNet-50": lambda n: models.resnet50(num_classes=n),
    "ResNet-101": lambda n: models.resnet101(num_classes=n),
    "VGG-16": lambda n: models.vgg16(num_classes=n),
    "VGG-19": lambda n: models.vgg19(num_classes=n),
    "MobileNet V2": lambda n: models.mobilenet_v2(num_classes=n),
    "MobileNet V3 Small": lambda n: models.mobilenet_v3_small(num_classes=n),
    "MobileNet V3 Large": lambda n: models.mobilenet_v3_large(num_classes=n),
    "EfficientNet B0": lambda n: models.efficientnet_b0(num_classes=n),
    "DenseNet-121": lambda n: models.densenet121(num_classes=n),
    "AlexNet": lambda n: models.alexnet(num_classes=n),
    "SqueezeNet 1.0": lambda n: models.squeezenet1_0(num_classes=n),
}


# ---------- 图像预处理 ----------
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------- 模型加载 ----------
def load_model(architecture, num_classes, model_path):
    """动态加载 PyTorch 模型"""
    if architecture not in ARCHITECTURES:
        return None, f"不支持的架构: {architecture}"

    if model_path is None:
        return None, "请上传 .pth 模型文件"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = ARCHITECTURES[architecture](num_classes)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model, None
    except Exception as e:
        # 尝试非 weights_only 模式
        try:
            model = ARCHITECTURES[architecture](num_classes)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, nn.Module):
                model = checkpoint
                model.to(device)
                model.eval()
                return model, None
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            return model, None
        except Exception as e2:
            return None, f"加载模型失败。请确认:\n1. 架构选择正确\n2. 类别数正确\n3. .pth 文件是有效的 state_dict\n\n错误: {str(e2)[:200]}"


# ---------- 推理 ----------
def predict(image, architecture, num_classes, model_file, labels_file):
    """运行推理，返回 Top-2 预测"""
    if model_file is None:
        return "错误: 请上传 .pth 模型文件"

    if image is None:
        return "错误: 请上传图片或拍照"

    # 加载模型
    model, error = load_model(architecture, num_classes, model_file.name if hasattr(model_file, 'name') else model_file)
    if error:
        return error

    # 加载标签
    if labels_file is not None:
        try:
            labels_path = labels_file.name if hasattr(labels_file, 'name') else labels_file
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]
        except Exception:
            labels = load_imagenet_labels()
    else:
        labels = load_imagenet_labels()

    # 确保标签数量 >= num_classes
    if len(labels) < num_classes:
        labels = labels + [f"Class_{i}" for i in range(len(labels), num_classes)]
    labels = labels[:num_classes]

    # 预处理图片
    transform = get_transform()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probs, k=2, dim=1)

        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]

        # 构建结果
        lines = ["##  预测结果 (Top-2)\n"]
        medals = ["🥇", "🥈"]
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            idx_int = int(idx)
            label = labels[idx_int] if idx_int < len(labels) else f"Class_{idx_int}"
            percent = prob * 100
            bar_len = int(percent / 2)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            lines.append(f"### {medals[i]} **{label}**")
            lines.append(f"```\n{bar}  {percent:.3f}%\n```\n")

        # 处理 top-2 概率差异
        diff = abs(top_probs[0] - top_probs[1]) * 100
        if diff < 5:
            lines.append(f"> ⚠️ Top-2 概率仅差 **{diff:.2f}%**，结果不确定度较高\n")
        else:
            lines.append(f"> Top-2 概率差 **{diff:.2f}%**，第一名可信度较高\n")

        lines.append(f"---\n")
        lines.append(f"*推理设备: {'GPU' if torch.cuda.is_available() else 'CPU'}*")

        return "\n".join(lines)

    except Exception as e:
        return f"推理出错: {str(e)[:300]}"


# ---------- Gradio 界面 ----------
with gr.Blocks(title="图像分类推理", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    #  图像分类模型推理平台
    **上传你的 `.pth` 模型 + 图片 → 输出 Top-2 预测标签及概率**

    支持摄像头拍照。免费 CPU 推理 (Hugging Face Spaces)。
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("###  模型配置")
            architecture = gr.Dropdown(
                choices=list(ARCHITECTURES.keys()),
                value="ResNet-50",
                label="模型架构"
            )
            num_classes = gr.Slider(
                minimum=2, maximum=10000, value=1000, step=1,
                label="类别数量 (ImageNet 默认 1000)"
            )
            model_file = gr.File(
                label="上传 .pth 模型文件",
                file_types=[".pth", ".pt", ".pkl"],
            )
            labels_file = gr.File(
                label="上传标签文件 (可选，txt 每行一个标签)",
                file_types=[".txt"],
            )
            gr.Markdown("""
            <small>
              <b>默认使用 ImageNet-1K 标签</b>，如需自定义标签请上传 txt 文件。<br><br>
              <b>.pth 文件要求</b>: 标准 PyTorch state_dict，键名与所选架构匹配。<br>
              <b>数量标注</b>: ✅ 的数量 = 类别数量<br>
              如 ResNet-50 有 1000 个类别，需要 1000 个标记。
            </small>
            """)

        with gr.Column(scale=1):
            gr.Markdown("###  图片输入")
            image_input = gr.Image(
                label="上传图片或拍照",
                type="pil",
                sources=["upload", "webcam"],
            )
            btn = gr.Button(" 开始推理", variant="primary", size="lg")

    with gr.Row():
        output = gr.Markdown(
            value="等待推理...",
            label="推理结果",
            elem_id="output-box",
        )

    btn.click(
        fn=predict,
        inputs=[image_input, architecture, num_classes, model_file, labels_file],
        outputs=output,
    )

    gr.Markdown("""
    ---
    ###  使用说明
    1. **选择模型架构** — 与你的 .pth 文件对应的 torchvision 架构
    2. **设置类别数** — ImageNet 预训练填 1000，自定义数据集填实际类别数
    3. **上传 .pth 模型** — 训练好的 state_dict 文件
    4. **上传图片** — 或点击相机图标拍照
    5. **点击推理** — 获取 Top-2 预测结果
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
