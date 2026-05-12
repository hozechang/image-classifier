"""
将 PyTorch .pth 模型转换为 ONNX 格式，用于浏览器推理
用法:
  python convert_to_onnx.py --arch resnet50 --num_classes 1000 --weights model.pth --output model.onnx
"""
import argparse
import torch
import torchvision.models as models

ARCHITECTURES = {
    "resnet18":              lambda n: models.resnet18(num_classes=n),
    "resnet34":              lambda n: models.resnet34(num_classes=n),
    "resnet50":              lambda n: models.resnet50(num_classes=n),
    "resnet101":             lambda n: models.resnet101(num_classes=n),
    "vgg16":                 lambda n: models.vgg16(num_classes=n),
    "vgg19":                 lambda n: models.vgg19(num_classes=n),
    "mobilenet_v2":          lambda n: models.mobilenet_v2(num_classes=n),
    "mobilenet_v3_small":    lambda n: models.mobilenet_v3_small(num_classes=n),
    "mobilenet_v3_large":    lambda n: models.mobilenet_v3_large(num_classes=n),
    "efficientnet_b0":       lambda n: models.efficientnet_b0(num_classes=n),
    "densenet121":           lambda n: models.densenet121(num_classes=n),
    "alexnet":               lambda n: models.alexnet(num_classes=n),
    "squeezenet1_0":         lambda n: models.squeezenet1_0(num_classes=n),
    "shufflenet_v2_x1_0":    lambda n: models.shufflenet_v2_x1_0(num_classes=n),
}


def main():
    parser = argparse.ArgumentParser(description="Convert .pth to ONNX")
    parser.add_argument("--arch", required=True, choices=list(ARCHITECTURES.keys()),
                        help="Model architecture")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of output classes")
    parser.add_argument("--weights", required=True, help="Path to .pth weights file")
    parser.add_argument("--output", default="model.onnx", help="Output ONNX file path")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    args = parser.parse_args()

    print(f"Architecture: {args.arch}")
    print(f"Num classes:  {args.num_classes}")
    print(f"Weights:      {args.weights}")
    print(f"Output:       {args.output}")
    print()

    # Build model
    print("Building model...")
    model = ARCHITECTURES[args.arch](args.num_classes)

    # Load weights
    print("Loading weights...")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("Weights loaded successfully.")

    # Export to ONNX
    print("Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size)
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12,
    )
    print(f"Done! ONNX model saved to: {args.output}")
    print(f"File size: {__import__('os').path.getsize(args.output) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
