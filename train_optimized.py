import os
import sys
import argparse
import tqdm
import numpy as np
from pathlib import Path

# 将当前目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.multiprocessing as mp
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from titans_pytorch import (
    MultimodalMAC,
    MemoryMLP,
    MSCOCOCaptionDataset,
    MSCOCOCaptionCollator
)

# 参数解析
parser = argparse.ArgumentParser(description='MSCOCO图像描述生成训练(优化版)')
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
parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
parser.add_argument('--grad_accum', type=int, default=1, help='梯度累积步数')
parser.add_argument('--fp16', action='store_true', help='是否使用混合精度训练')
parser.add_argument('--data_ratio', type=float, default=0.33, help='使用数据集的比例')
parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
parser.add_argument('--patience', type=int, default=3, help='早停耐心值')
args = parser.parse_args()

# 设置随机种子
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# 设置多进程启动方式
if args.num_workers > 0:
    mp.set_start_method('spawn', force=True)

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
USE_FLEX_ATTN = False

# wandb实验跟踪器
if args.wandb:
    import wandb
    wandb.init(project='titans-coco-captioning-optimized')
    wandb.config.update(args)

# 辅助函数
def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def main():
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
            use_accelerated_scan=False,  # 禁用加速扫描以兼容FP16
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
    
    # 减少数据集大小
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    train_indices = list(range(train_dataset_size))
    val_indices = list(range(val_dataset_size))
    
    import random
    random.seed(args.seed)
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    
    # 按指定比例选择数据
    train_indices = train_indices[:int(train_dataset_size * args.data_ratio)]
    val_indices = val_indices[:int(val_dataset_size * args.data_ratio)]
    
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    
    print(f"缩减后训练集大小: {len(train_dataset)}")
    print(f"缩减后验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_collator = MSCOCOCaptionCollator(image_processor=model.image_processor, device=device)
    val_collator = MSCOCOCaptionCollator(image_processor=model.image_processor, device=device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collator,
        pin_memory=False  # 设置为False，因为collator已经将数据移到设备上
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_collator,
        pin_memory=False  # 设置为False
    )
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) // args.grad_accum * args.epochs
    )
    
    # 混合精度训练
    scaler = GradScaler() if args.fp16 and args.device == 'cuda' else None
    
    # 训练状态
    start_epoch = 0
    best_val_loss = float('inf')
    no_improve = 0
    
    # 从检查点恢复
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> 从检查点恢复训练 '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"=> 成功加载检查点 '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> 未找到检查点 '{args.resume}'")
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        # 训练
        model.train()
        train_loss = 0
        train_steps = 0
        
        # 重置梯度
        optimizer.zero_grad()
        
        train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for i, batch in enumerate(train_pbar):
            # 前向传播
            if args.fp16 and args.device == 'cuda':
                with autocast(device_type='cuda'):
                    loss = model(
                        batch['input_ids'],
                        images=batch['images'],
                        image_positions=batch['image_positions'],
                        return_loss=True
                    )
                    # 梯度累积
                    loss = loss / args.grad_accum
                
                # 反向传播
                scaler.scale(loss).backward()
                
                # 梯度累积步数
                if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # 优化器步骤
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # 学习率调度器
                    scheduler.step()
            else:
                loss = model(
                    batch['input_ids'],
                    images=batch['images'],
                    image_positions=batch['image_positions'],
                    return_loss=True
                )
                # 梯度累积
                loss = loss / args.grad_accum
                
                # 反向传播
                loss.backward()
                
                # 梯度累积步数
                if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # 优化器步骤
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 学习率调度器
                    scheduler.step()
            
            # 更新统计
            train_loss += loss.item() * args.grad_accum
            train_steps += 1
            
            # 进度条更新
            lr = scheduler.get_last_lr()[0]
            train_pbar.set_postfix(loss=loss.item() * args.grad_accum, lr=f"{lr:.6f}")
            
            # 记录到wandb
            if args.wandb and (i + 1) % args.grad_accum == 0:
                wandb.log({
                    'train_loss': loss.item() * args.grad_accum,
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
                if args.fp16 and args.device == 'cuda':
                    with autocast(device_type='cuda'):
                        loss = model(
                            batch['input_ids'],
                            images=batch['images'],
                            image_positions=batch['image_positions'],
                            return_loss=True
                        )
                else:
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
            no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"早停：{args.patience}个epoch未改善")
                break
        
        # 保存每个epoch的检查点
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
        }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
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
                max_length=70,  # 增加最大长度以生成更详细描述
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

if __name__ == "__main__":
    main()