import os
import random
import tqdm
import numpy as np
from PIL import Image
import json

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from adam_atan2_pytorch import AdoptAtan2

from titans_pytorch import (
    MultimodalMAC,
    MemoryMLP
)

# 常量

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = True
SEQ_LEN = 512

# 神经记忆相关

NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 4
NEURAL_MEM_LAYERS = (2, 4, 6)
NEURAL_MEM_GATE_ATTN_OUTPUT = False
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-1
USE_MEM_ATTENTION_MODEL = False
WINDOW_SIZE = 32
NEURAL_MEM_SEGMENT_LEN = 4
NEURAL_MEM_BATCH_SIZE = 128
SLIDING_WINDOWS = True
STORE_ATTN_POOL_CHUNKS = True
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
NEURAL_MEM_WEIGHT_RESIDUAL = True
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True
NEURAL_MEM_SPEC_NORM_SURPRISES = True

# 图像相关参数

IMAGE_SIZE = 224
PATCH_SIZE = 16
VISION_DIM = 768
VISION_DEPTH = 12
VISION_HEADS = 12
VISION_MLP_DIM = 3072
VISION_DROPOUT = 0.1

# 实验相关

PROJECT_NAME = 'titans-multimodal-mac'
RUN_NAME = f'multimodal-mac - {NUM_LONGTERM_MEM} longterm mems, layers {NEURAL_MEM_LAYERS}'
WANDB_ONLINE = False  # 设置为True以将实验数据发送到云端

# 性能相关

USE_ACCELERATED_SCAN = True
USE_FLEX_ATTN = True
USE_FAST_INFERENCE = False

# wandb实验跟踪器

import wandb
wandb.init(project=PROJECT_NAME, mode='disabled' if not WANDB_ONLINE else 'online')
wandb.run.name = RUN_NAME
wandb.run.save()

# 辅助函数

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# 多模态数据集

class MultimodalDataset(Dataset):
    def __init__(self, text_data, image_dir, annotations_file, seq_len, tokenizer=None):
        super().__init__()
        self.text_data = text_data
        self.image_dir = image_dir
        self.seq_len = seq_len
        
        # 加载图像-文本对的注释
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # 简单的字符级tokenizer
        self.tokenizer = tokenizer or (lambda x: torch.tensor([ord(c) for c in x], dtype=torch.long))
        
        # 特殊token
        self.image_token_id = 255  # 使用255作为图像token的ID
        
    def __getitem__(self, index):
        # 随机选择一个样本
        sample = random.choice(self.annotations)
        
        # 获取文本和图像路径
        text = sample['text']
        image_path = os.path.join(self.image_dir, sample['image'])
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 使用空白图像作为替代
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='black')
        
        # 在文本中插入图像标记
        image_position = random.randint(0, min(len(text) - 1, self.seq_len - 20))
        text_with_image = text[:image_position] + f"<image>" + text[image_position:]
        
        # Tokenize文本
        tokens = []
        image_token_position = -1
        
        for i, part in enumerate(text_with_image.split("<image>")):
            part_tokens = self.tokenizer(part)
            tokens.extend(part_tokens)
            if i == 0:  # 第一个分割后插入图像token
                image_token_position = len(tokens)
                tokens.append(self.image_token_id)  # 图像token
        
        # 截断到序列长度
        if len(tokens) > self.seq_len:
            if image_token_position >= self.seq_len:
                # 图像位置超出范围，重新调整
                tokens = tokens[:self.seq_len-1]
                tokens[self.seq_len-2] = self.image_token_id
                image_token_position = self.seq_len-2
            else:
                tokens = tokens[:self.seq_len]
        
        # 转换为tensor
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        
        # 填充到固定长度
        if len(tokens_tensor) < self.seq_len:
            padding = torch.zeros(self.seq_len - len(tokens_tensor), dtype=torch.long)
            tokens_tensor = torch.cat([tokens_tensor, padding])
        
        return {
            'input_ids': tokens_tensor.cuda(),
            'image': image,
            'image_position': torch.tensor([image_token_position], dtype=torch.long).cuda()
        }
    
    def __len__(self):
        return len(self.annotations) * 10  # 每个样本可以多次使用

# 模拟数据

def create_mock_data():
    # 创建模拟文本数据
    mock_text = "This is a sample text for multimodal training. " * 1000
    mock_text_data = torch.tensor([ord(c) for c in mock_text], dtype=torch.long)
    
    # 创建模拟图像目录
    os.makedirs("mock_images", exist_ok=True)
    
    # 创建一些模拟图像
    for i in range(10):
        img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        img.save(f"mock_images/image_{i}.jpg")
    
    # 创建模拟注释
    mock_annotations = []
    for i in range(100):
        mock_annotations.append({
            "image": f"image_{random.randint(0, 9)}.jpg",
            "text": mock_text[i*100:(i+1)*100]
        })
    
    # 保存模拟注释
    with open("mock_annotations.json", "w") as f:
        json.dump(mock_annotations, f)
    
    return mock_text_data, "mock_images", "mock_annotations.json"

# 记忆模型

if USE_MEM_ATTENTION_MODEL:
    neural_memory_model = MemoryAttention(
        dim = 64
    )
else:
    neural_memory_model = MemoryMLP(
        dim = 64,
        depth = NEURAL_MEMORY_DEPTH
    )

# 实例化多模态MAC Transformer

model = MultimodalMAC(
    # Vision Transformer参数
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    vision_dim=VISION_DIM,
    vision_depth=VISION_DEPTH,
    vision_heads=VISION_HEADS,
    vision_mlp_dim=VISION_MLP_DIM,
    vision_dropout=VISION_DROPOUT,
    
    # MAC Transformer参数
    num_tokens=256,
    text_dim=384,
    depth=8,
    segment_len=WINDOW_SIZE,
    num_persist_mem_tokens=NUM_PERSIST_MEM,
    num_longterm_mem_tokens=NUM_LONGTERM_MEM,
    neural_memory_layers=NEURAL_MEM_LAYERS,
    neural_memory_segment_len=NEURAL_MEM_SEGMENT_LEN,
    neural_memory_batch_size=NEURAL_MEM_BATCH_SIZE,
    neural_mem_gate_attn_output=NEURAL_MEM_GATE_ATTN_OUTPUT,
    neural_mem_weight_residual=NEURAL_MEM_WEIGHT_RESIDUAL,
    neural_memory_qkv_receives_diff_views=NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
    use_flex_attn=USE_FLEX_ATTN,
    sliding_window_attn=SLIDING_WINDOWS,
    neural_memory_model=neural_memory_model,
    neural_memory_kwargs=dict(
        dim_head=64,
        heads=4,
        attn_pool_chunks=STORE_ATTN_POOL_CHUNKS,
        qk_rmsnorm=NEURAL_MEM_QK_NORM,
        momentum=NEURAL_MEM_MOMENTUM,
        momentum_order=NEURAL_MEM_MOMENTUM_ORDER,
        default_step_transform_max_lr=NEURAL_MEM_MAX_LR,
        use_accelerated_scan=USE_ACCELERATED_SCAN,
        per_parameter_lr_modulation=MEMORY_MODEL_PER_LAYER_LEARNED_LR,
        spectral_norm_surprises=NEURAL_MEM_SPEC_NORM_SURPRISES
    )
).cuda()

# 准备数据

# 如果有真实数据，使用真实数据
# 否则创建模拟数据
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    text_data = torch.load(config['text_data_path'])
    image_dir = config['image_dir']
    annotations_file = config['annotations_file']
except:
    print("使用模拟数据进行训练")
    text_data, image_dir, annotations_file = create_mock_data()

# 创建数据集和数据加载器
train_dataset = MultimodalDataset(text_data, image_dir, annotations_file, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))

# 优化器
optim = AdoptAtan2(model.parameters(), lr=LEARNING_RATE)

# 训练循环
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    
    total_loss = 0
    
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        batch = next(train_loader)
        
        # 处理图像
        images = [img for img in batch['image']]
        image_positions = batch['image_position']
        
        # 将图像转换为tensor
        image_processor = model.image_processor
        image_tensors = torch.stack([image_processor.preprocess(img) for img in images]).cuda()
        
        # 前向传播
        loss = model(
            batch['input_ids'], 
            images=image_tensors, 
            image_positions=image_positions,
            return_loss=True
        )
        
        loss = loss / GRADIENT_ACCUMULATE_EVERY
        loss.backward()
        total_loss += loss.item()
    
    print(f'training loss: {total_loss}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()
    wandb.log(dict(loss=total_loss))
    
    if i % GENERATE_EVERY == 0 and SHOULD_GENERATE:
        model.eval()
        
        # 获取一个样本用于生成
        sample_batch = next(train_loader)
        sample_image = sample_batch['image'][0]
        sample_image_tensor = image_processor.preprocess(sample_image).unsqueeze(0).cuda()
        
        # 创建提示
        prompt = torch.tensor([[ord(c) for c in "This is an image of "] + [model.image_token_id]], dtype=torch.long).cuda()
        
        # 生成文本
        generated = model.generate(
            prompt,
            images=sample_image_tensor.unsqueeze(1),
            image_positions=torch.tensor([[len(prompt[0])-1]]).cuda(),
            max_length=100,
            temperature=0.8
        )
        
        # 解码生成的文本
        output_str = decode_tokens(generated[0])
        print(f"\nGenerated text: {output_str}\n")
        
        # 保存checkpoint
        torch.save(model.state_dict(), f'checkpoints/multimodal_mac_{i}.pt')

print("Training completed!") 