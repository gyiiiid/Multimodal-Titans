import os
import argparse
import random
import json
from PIL import Image

import torch
from titans_pytorch import MultimodalMAC, MemoryMLP

# 参数解析
parser = argparse.ArgumentParser(description='批量图像描述生成推理')
parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
parser.add_argument('--val_image_dir', type=str, required=True, help='验证集图像目录')
parser.add_argument('--ann_file', type=str, required=True, help='注释文件路径')
parser.add_argument('--num_samples', type=int, default=5, help='要测试的图像数量')
parser.add_argument('--max_length', type=int, default=100, help='生成的最大长度')
parser.add_argument('--temperature', type=float, default=0.7, help='采样温度')
parser.add_argument('--output_dir', type=str, default='./inference_results', help='结果输出目录')
parser.add_argument('--device', type=str, default='cuda', help='设备')
parser.add_argument('--seed', type=int, default=42, help='随机种子')
args = parser.parse_args()

# 设置随机种子
random.seed(args.seed)
torch.manual_seed(args.seed)

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

# 辅助函数
def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def load_annotations(ann_file):
    """加载MSCOCO注释文件"""
    with open(ann_file, 'r') as f:
        return json.load(f)

def get_random_images(annotations, num_samples):
    """从注释文件中随机选择图像"""
    images = annotations['images']
    return random.sample(images, min(num_samples, len(images)))

def get_captions_for_image(annotations, image_id):
    """获取图像的所有描述"""
    captions = []
    for ann in annotations['annotations']:
        if ann['image_id'] == image_id:
            captions.append(ann['caption'])
    return captions

def main():
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 加载注释文件
    print(f"加载注释文件: {args.ann_file}")
    annotations = load_annotations(args.ann_file)
    
    # 随机选择图像
    selected_images = get_random_images(annotations, args.num_samples)
    print(f"随机选择了 {len(selected_images)} 张图像进行测试")
    
    # 结果保存文件
    results_file = os.path.join(args.output_dir, 'batch_inference_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"模型检查点: {args.checkpoint}\n")
        f.write(f"测试图像数量: {len(selected_images)}\n")
        f.write(f"最大生成长度: {args.max_length}\n")
        f.write(f"采样温度: {args.temperature}\n\n")
        f.write("="*50 + "\n\n")
    
    # 逐个处理选定的图像
    for i, image_info in enumerate(selected_images):
        image_id = image_info['id']
        filename = image_info['file_name']
        image_path = os.path.join(args.val_image_dir, filename)
        
        print(f"\n处理图像 {i+1}/{len(selected_images)}: {filename}")
        
        # 获取原始描述
        original_captions = get_captions_for_image(annotations, image_id)
        
        # 加载和预处理图像
        try:
            image = Image.open(image_path).convert('RGB')
            image_processor = model.image_processor
            image_tensor = image_processor.preprocess(image).unsqueeze(0).to(device)
            
            # 创建提示（只包含图像标记）
            prompt = torch.tensor([[model.image_token_id]], dtype=torch.long).to(device)
            
            # 生成描述
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
            
            # 打印结果
            print(f"\n===== 图像 {i+1}: {filename} =====")
            print(f"生成的描述 ({len(output_str)} 字符):\n{output_str}")
            print("\n原始描述:")
            for j, caption in enumerate(original_captions):
                print(f"{j+1}. {caption}")
            
            # 保存结果
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(f"图像 {i+1}: {filename}\n")
                f.write(f"图像ID: {image_id}\n\n")
                f.write(f"生成的描述 ({len(output_str)} 字符):\n{output_str}\n\n")
                f.write("原始描述:\n")
                for j, caption in enumerate(original_captions):
                    f.write(f"{j+1}. {caption}\n")
                f.write("\n" + "="*50 + "\n\n")
            
            # 保存图像和结果
            image_output_path = os.path.join(args.output_dir, f"image_{i+1}_{image_id}.jpg")
            image.save(image_output_path)
            
        except Exception as e:
            print(f"处理图像 {filename} 时出错: {e}")
    
    print(f"\n批量推理完成! 结果已保存到 {results_file}")

if __name__ == "__main__":
    main()