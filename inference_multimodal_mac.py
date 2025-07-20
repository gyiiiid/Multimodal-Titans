import os
import argparse
from PIL import Image

import torch
from titans_pytorch import MultimodalMAC, MemoryMLP, ImageProcessor

# 解析命令行参数
parser = argparse.ArgumentParser(description='多模态MAC Transformer推理')
parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
parser.add_argument('--image', type=str, required=True, help='输入图像路径')
parser.add_argument('--prompt', type=str, default='This is an image of ', help='文本提示')
parser.add_argument('--max_length', type=int, default=100, help='生成的最大长度')
parser.add_argument('--temperature', type=float, default=0.8, help='采样温度')
args = parser.parse_args()

# 常量 - 与训练时保持一致
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
NEURAL_MEM_GATE_ATTN_OUTPUT = False
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-1
WINDOW_SIZE = 32
NEURAL_MEM_SEGMENT_LEN = 4
NEURAL_MEM_BATCH_SIZE = 128
SLIDING_WINDOWS = True
STORE_ATTN_POOL_CHUNKS = True
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
NEURAL_MEM_WEIGHT_RESIDUAL = True
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True
NEURAL_MEM_SPEC_NORM_SURPRISES = True
USE_ACCELERATED_SCAN = True
USE_FLEX_ATTN = True

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def main():
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建记忆模型
    neural_memory_model = MemoryMLP(
        dim=64,
        depth=NEURAL_MEMORY_DEPTH
    )
    
    # 创建模型
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
    
    # 准备提示
    prompt_text = args.prompt
    image_token_id = model.image_token_id
    
    # 在提示中插入图像标记
    prompt_tokens = [ord(c) for c in prompt_text] + [image_token_id]
    prompt = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    
    # 图像位置
    image_position = torch.tensor([[len(prompt_tokens) - 1]]).to(device)
    
    print(f"提示: {prompt_text}[IMAGE]")
    print(f"生成中...")
    
    # 生成文本
    with torch.no_grad():
        generated = model.generate(
            prompt,
            images=image_tensor.unsqueeze(1),
            image_positions=image_position,
            max_length=args.max_length,
            temperature=args.temperature
        )
    
    # 解码生成的文本
    output_str = decode_tokens(generated[0])
    print("\n生成的文本:")
    print(f"{prompt_text}[IMAGE]{output_str}")

if __name__ == "__main__":
    main() 