# Titans PyTorch 多模态扩展

这个扩展为Titans PyTorch库添加了多模态功能，使模型能够处理图像和文本输入。

## 功能特点

- Vision Transformer编码器，用于将图像编码为向量表示
- 多模态MAC Transformer模型，支持图像和文本的联合处理
- 图像预处理工具，简化图像输入流程
- 支持图像引导的文本生成

## 安装

确保已安装基本的Titans PyTorch库，然后安装额外的依赖：

```bash
pip install torch torchvision pillow einops
```

## 使用方法

### 1. 导入必要的模块

```python
from titans_pytorch import MultimodalMAC, VisionTransformerEncoder, ImageProcessor
```

### 2. 创建多模态模型

```python
model = MultimodalMAC(
    # Vision Transformer参数
    image_size=224,
    patch_size=16,
    vision_dim=768,
    vision_depth=12,
    vision_heads=12,
    vision_mlp_dim=3072,
    vision_dropout=0.1,
    
    # MAC Transformer参数
    num_tokens=256,
    text_dim=384,
    depth=8,
    segment_len=32,
    # 其他参数...
)
```

### 3. 处理图像

```python
from PIL import Image
import torch

# 加载图像
image = Image.open('example.jpg').convert('RGB')

# 使用模型的图像处理器进行预处理
image_processor = model.image_processor
image_tensor = image_processor.preprocess(image).unsqueeze(0)  # [1, 3, 224, 224]
```

### 4. 准备文本输入

```python
# 创建包含图像标记的输入
prompt_text = "This is an image of "
image_token_id = model.image_token_id

# 在提示中插入图像标记
prompt_tokens = [ord(c) for c in prompt_text] + [image_token_id]
prompt = torch.tensor([prompt_tokens], dtype=torch.long)

# 图像位置（在序列中的位置）
image_position = torch.tensor([[len(prompt_tokens) - 1]])
```

### 5. 生成文本

```python
# 生成文本
generated = model.generate(
    prompt,
    images=image_tensor.unsqueeze(1),  # [batch_size, num_images, channels, height, width]
    image_positions=image_position,
    max_length=100,
    temperature=0.8
)

# 解码生成的文本
def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

output_str = decode_tokens(generated[0])
print(f"{prompt_text}[IMAGE]{output_str}")
```

## 训练多模态模型

使用提供的训练脚本进行训练：

```bash
python train_multimodal_mac.py
```

训练脚本会自动创建模拟数据进行训练。要使用自己的数据集，请创建一个`config.json`文件，指定以下内容：

```json
{
  "text_data_path": "path/to/text_data.pt",
  "image_dir": "path/to/images",
  "annotations_file": "path/to/annotations.json"
}
```

注解文件应该是一个JSON文件，包含图像和文本对：

```json
[
  {
    "image": "image1.jpg",
    "text": "This is the description of image 1."
  },
  {
    "image": "image2.jpg",
    "text": "This is the description of image 2."
  }
]
```

## 推理

使用提供的推理脚本进行推理：

```bash
python inference_multimodal_mac.py --checkpoint path/to/model.pt --image path/to/image.jpg --prompt "This is an image of " --max_length 100 --temperature 0.8
```

## 自定义

### 使用不同的图像编码器

您可以通过修改`VisionTransformerEncoder`类或创建自己的编码器来自定义图像编码：

```python
class CustomImageEncoder(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 自定义实现
    
    def forward(self, x):
        # 自定义前向传播
        return image_features
```

### 自定义多模态融合

您可以通过修改`MultimodalMAC`类中的`forward`方法来自定义图像和文本的融合方式。

## 注意事项

- 图像和文本的维度应该匹配，或者通过投影层进行适当的转换
- 对于大型数据集，建议使用数据并行训练
- 使用预训练的视觉模型可以提高性能 