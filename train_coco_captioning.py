import os
import argparse
import tqdm
import numpy as np
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from titans_pytorch import (
    MultimodalMAC,
    MemoryMLP,
    MSCOCOCaptionDataset,
    MSCOCOCaptionCollator
)

# 参数解析
parser = argparse.ArgumentParser(description='MSCOCO图像描述生成训练')
parser.add_argument('--coco_dir', type=str, required=True, help='MSCOCO数据集根目录')
parser.add_argument('--image_dir', type=str, help='图像目录，默认为[coco_dir]/train2014')
parser.add_argument('--ann_file', type=str, help='注释文件，默认为[coco_dir]/annotations/captions_train2014.json')
parser.add_argument('--val_image_dir', type=str, help='验证集图像目录，默认为[coco_dir]/val2014')
parser.add_argument('--val_ann_file', type=str, help='验证集注释文件，默认为[coco_dir]/annotations/captions_val2014.json')
parser.add_argument('--output_dir', type=str, default='./checkpoints', help='输出目录')
parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')
parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
parser.add_argument('--max_seq_len', type=int, default=77, help='最大序列长度')
parser.add_argument('--seed', type=int, default=42, help='随机种子')
parser.add_argument('--device', type=str, default='cuda', help='设备')
parser.add_argument('--wandb', action='store_true', help='是否使用wandb记录训练过程')
args = parser.parse_args()

# 设置随机种子
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# 设置设备
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置默认路径
if args.image_dir is None:
    args.image_dir = os.path.join(args.coco_dir, 'train2014')
if args.ann_file is None:
    args.ann_file = os.path.join(args.coco_dir, 'annotations', 'captions_train2014.json')
if args.val_image_dir is None:
    args.val_image_dir = os.path.join(args.coco_dir, 'val2014')
if args.val_ann_file is None:
    args.val_ann_file = os.path.join(args.coco_dir, 'annotations', 'captions_val2014.json')

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# 常量
IMAGE_SIZE = 224
PATCH_SIZE = 16
VISION_DIM = 768
VISION_DEPTH = 12
VISION_HEADS = 12
VISION_MLP_DIM = 3072
VISION_DROPOUT = 0.1

# 神经记忆相关
NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 4
NEURAL_MEM_LAYERS = (2, 4, 6)
WINDOW_SIZE = 32
NEURAL_MEM_SEGMENT_LEN = 4
NEURAL_MEM_BATCH_SIZE = 128
SLIDING_WINDOWS = True
NEURAL_MEM_WEIGHT_RESIDUAL = True
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True
USE_FLEX_ATTN = True

# wandb实验跟踪器
if args.wandb:
    import wandb
    wandb.init(project='titans-coco-captioning')
    wandb.config.update(args)

# 辅助函数
def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# 创建记忆模型
neural_memory_model = MemoryMLP(
    dim=64,
    depth=NEURAL_MEMORY_DEPTH
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
    num_tokens=256,  # 字符级tokenization
    text_dim=384,
    depth=8,
    segment_len=WINDOW_SIZE,
    num_persist_mem_tokens=NUM_PERSIST_MEM,
    num_longterm_mem_tokens=NUM_LONGTERM_MEM,
    neural_memory_layers=NEURAL_MEM_LAYERS,
    neural_memory_segment_len=NEURAL_MEM_SEGMENT_LEN,
    neural_memory_batch_size=NEURAL_MEM_BATCH_SIZE,
    neural_mem_weight_residual=NEURAL_MEM_WEIGHT_RESIDUAL,
    neural_memory_qkv_receives_diff_views=NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
    use_flex_attn=USE_FLEX_ATTN,
    sliding_window_attn=SLIDING_WINDOWS,
    neural_memory_model=neural_memory_model,
    neural_memory_kwargs=dict(
        dim_head=64,
        heads=4,
        attn_pool_chunks=True,
        qk_rmsnorm=True,
        momentum=True,
        momentum_order=1,
        default_step_transform_max_lr=1e-1,
        use_accelerated_scan=True,
        per_parameter_lr_modulation=True,
        spectral_norm_surprises=True
    )
).to(device)

# 创建数据集
train_dataset = MSCOCOCaptionDataset(
    root_dir=args.image_dir,
    ann_file=args.ann_file,
    max_seq_len=args.max_seq_len,
    image_token_id=model.image_token_id
)

val_dataset = MSCOCOCaptionDataset(
    root_dir=args.val_image_dir,
    ann_file=args.val_ann_file,
    max_seq_len=args.max_seq_len,
    image_token_id=model.image_token_id
)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")

# 创建数据加载器
train_collator = MSCOCOCaptionCollator(image_processor=model.image_processor, device=device)
val_collator = MSCOCOCaptionCollator(image_processor=model.image_processor, device=device)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=train_collator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=val_collator
)

# 优化器
optimizer = AdamW(model.parameters(), lr=args.lr)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=len(train_loader) * args.epochs
)

# 训练循环
best_val_loss = float('inf')

for epoch in range(args.epochs):
    # 训练
    model.train()
    train_loss = 0
    train_steps = 0
    
    train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
    for batch in train_pbar:
        # 前向传播
        loss = model(
            batch['input_ids'],
            images=batch['images'],
            image_positions=batch['image_positions'],
            return_loss=True
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # 更新统计
        train_loss += loss.item()
        train_steps += 1
        train_pbar.set_postfix(loss=loss.item())
        
        # 记录到wandb
        if args.wandb:
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': scheduler.get_last_lr()[0]
            })
    
    avg_train_loss = train_loss / train_steps
    print(f"Epoch {epoch+1}/{args.epochs} - 平均训练损失: {avg_train_loss:.4f}")
    
    # 验证
    model.eval()
    val_loss = 0
    val_steps = 0
    
    val_pbar = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
    with torch.no_grad():
        for batch in val_pbar:
            # 前向传播
            loss = model(
                batch['input_ids'],
                images=batch['images'],
                image_positions=batch['image_positions'],
                return_loss=True
            )
            
            # 更新统计
            val_loss += loss.item()
            val_steps += 1
            val_pbar.set_postfix(loss=loss.item())
    
    avg_val_loss = val_loss / val_steps
    print(f"Epoch {epoch+1}/{args.epochs} - 平均验证损失: {avg_val_loss:.4f}")
    
    # 记录到wandb
    if args.wandb:
        wandb.log({
            'epoch': epoch + 1,
            'avg_train_loss': avg_train_loss,
            'avg_val_loss': avg_val_loss
        })
    
    # 保存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
        print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
    
    # 保存每个epoch的模型
    torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pt'))
    
    # 生成一些样例
    model.eval()
    with torch.no_grad():
        # 获取一个验证样本
        val_batch = next(iter(val_loader))
        
        # 选择第一个图像
        sample_image = val_batch['images'][0:1]
        
        # 创建提示（只包含图像标记）
        prompt = torch.tensor([[model.image_token_id]], dtype=torch.long).to(device)
        
        # 生成描述
        generated = model.generate(
            prompt,
            images=sample_image,
            image_positions=torch.tensor([[0]]).to(device),
            max_length=30,
            temperature=0.7
        )
        
        # 解码生成的文本
        output_str = decode_tokens(generated[0])
        original_caption = val_batch['captions'][0]
        
        print(f"\n生成的描述: {output_str}")
        print(f"原始描述: {original_caption}\n")
        
        # 记录到wandb
        if args.wandb:
            wandb.log({
                'example_generation': output_str,
                'example_original': original_caption
            })

print("训练完成!") 