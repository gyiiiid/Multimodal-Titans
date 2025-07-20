import os
import argparse
from PIL import Image

import torch
from titans_pytorch import MultimodalMAC, MemoryMLP

# 参数解析
parser = argparse.ArgumentParser(description='图像描述生成推理')
parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
parser.add_argument('--image', type=str, required=True, help='输入图像路径')
parser.add_argument('--output', type=str, help='输出文件路径')
parser.add_argument('--max_length', type=int, default=30, help='生成的最大长度')
parser.add_argument('--temperature', type=float, default=0.7, help='采样温度')
parser.add_argument('--device', type=str, default='cuda', help='设备')
args = parser.parse_args()

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

# 辅助函数
def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def main():
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
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
        num_tokens=256,
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
    
    # 加载检查点
    print(f"加载检查点: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # 加载和预处理图像
    print(f"加载图像: {args.image}")
    image = Image.open(args.image).convert('RGB')
    image_processor = model.image_processor
    image_tensor = image_processor.preprocess(image).unsqueeze(0).to(device)
    
    # 创建提示（只包含图像标记）
    prompt = torch.tensor([[model.image_token_id]], dtype=torch.long).to(device)
    
    # 生成描述
    print("生成描述中...")
    with torch.no_grad():
        generated = model.generate(
            prompt,
            images=image_tensor.unsqueeze(1),
            image_positions=torch.tensor([[0]]).to(device),
            max_length=args.max_length,
            temperature=args.temperature
        )
    
    # 解码生成的文本
    output_str = decode_tokens(generated[0])
    print(f"\n生成的描述: {output_str}")
    
    # 保存输出
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_str)
        print(f"描述已保存到: {args.output}")
    
    return output_str

if __name__ == "__main__":
    main() 