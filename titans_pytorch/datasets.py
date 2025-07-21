import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class MSCOCOCaptionDataset(Dataset):
    """
    MSCOCO图像描述数据集加载器
    """
    def __init__(
        self,
        root_dir,
        ann_file,
        transform=None,
        max_seq_len=77,
        image_token_id=255,
        tokenizer=None
    ):
        """
        初始化MSCOCO数据集
        
        Args:
            root_dir (str): 图像目录路径
            ann_file (str): 注释文件路径
            transform (callable, optional): 图像转换函数
            max_seq_len (int): 最大序列长度
            image_token_id (int): 图像标记ID
            tokenizer (callable, optional): 文本标记化函数
        """
        self.root_dir = root_dir
        self.max_seq_len = max_seq_len
        self.transform = transform
        self.image_token_id = image_token_id
        
        # 加载注释
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        
        # 处理注释格式
        if 'annotations' in self.annotations:
            self.captions = self.annotations['annotations']
        else:
            self.captions = self.annotations
        
        # 创建图像ID到文件名的映射
        self.image_id_to_filename = {}
        if 'images' in self.annotations:
            for img in self.annotations['images']:
                self.image_id_to_filename[img['id']] = img['file_name']
        
        # 简单的字符级tokenizer
        self.tokenizer = tokenizer or (lambda x: [ord(c) for c in x])
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        caption_info = self.captions[idx]
        
        # 获取图像ID和文件名
        if 'image_id' in caption_info:
            image_id = caption_info['image_id']
            if self.image_id_to_filename:
                image_filename = self.image_id_to_filename[image_id]
            else:
                # 如果没有映射，假设文件名格式为COCO_[split]_[id].jpg
                image_filename = f"COCO_train2014_{image_id:012d}.jpg"
        else:
            image_filename = caption_info['file_name']
        
        # 获取图像路径
        image_path = os.path.join(self.root_dir, image_filename)
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 使用空白图像作为替代
            image = Image.new('RGB', (224, 224), color='black')
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        # 获取描述文本
        caption = caption_info['caption']
        
        # 标记化文本
        tokens = self.tokenizer(caption)
        
        # 准备输入序列：[图像标记] + 文本
        input_tokens = [self.image_token_id] + tokens
        
        # 截断到最大长度-1（为了留出EOS标记的空间）
        if len(input_tokens) > self.max_seq_len - 1:
            input_tokens = input_tokens[:self.max_seq_len - 1]
        
        # 添加EOS标记（这里使用0作为EOS）
        input_tokens.append(0)
        
        # 转换为tensor
        input_ids = torch.tensor(input_tokens, dtype=torch.long)
        
        # 填充到固定长度
        if len(input_ids) < self.max_seq_len:
            padding = torch.zeros(self.max_seq_len - len(input_ids), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
        
        return {
            'input_ids': input_ids,
            'image': image,
            'image_position': torch.tensor([0], dtype=torch.long),  # 图像位于序列开头
            'caption': caption
        }

class MSCOCOCaptionCollator:
    """
    MSCOCO数据集的批处理整理器
    """
    def __init__(self, image_processor=None, device='cuda'):
        self.image_processor = image_processor
        self.device = device
    
    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        image_positions = torch.stack([item['image_position'] for item in batch])
        
        # 处理图像
        if self.image_processor:
            images = torch.stack([
                self.image_processor.preprocess(item['image'])
                for item in batch
            ])
        else:
            # 如果没有提供处理器，假设图像已经是tensor
            images = torch.stack([item['image'] for item in batch])
        
        # 收集原始描述文本
        captions = [item['caption'] for item in batch]
        
        # 移动到设备
        input_ids = input_ids.to(self.device)
        images = images.to(self.device)
        image_positions = image_positions.to(self.device)
        
        return {
            'input_ids': input_ids,
            'images': images.unsqueeze(1),  # 添加num_images维度 [B, 1, C, H, W]
            'image_positions': image_positions,
            'captions': captions
        } 