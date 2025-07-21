import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import random

class HuggingFaceMSCOCODataset(Dataset):
    """
    使用Hugging Face datasets库加载MSCOCO数据集
    """
    def __init__(
        self,
        split="train",
        max_seq_len=77,
        image_token_id=255,
        tokenizer=None,
        transform=None,
        dataset_name="nlphuji/coco_karpathy",  # 默认使用Karpathy分割的MSCOCO
        cache_dir=None
    ):
        """
        初始化数据集
        
        Args:
            split (str): 数据集分割，可以是'train', 'validation', 'test'
            max_seq_len (int): 最大序列长度
            image_token_id (int): 图像标记ID
            tokenizer (callable, optional): 文本标记化函数
            transform (callable, optional): 图像转换函数
            dataset_name (str): Hugging Face数据集名称
            cache_dir (str, optional): 缓存目录
        """
        self.max_seq_len = max_seq_len
        self.transform = transform
        self.image_token_id = image_token_id
        
        # 加载数据集
        print(f"加载Hugging Face数据集: {dataset_name} ({split})")
        self.dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        print(f"数据集大小: {len(self.dataset)}")
        
        # 简单的字符级tokenizer
        self.tokenizer = tokenizer or (lambda x: [ord(c) for c in x])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 获取图像
        image = item["image"]
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        # 获取描述文本
        if "sentences" in item and isinstance(item["sentences"], dict):
            # Karpathy格式
            caption = item["sentences"]["raw"]
        elif "sentences" in item and isinstance(item["sentences"], list):
            # 随机选择一个描述
            caption = random.choice(item["sentences"])["raw"]
        elif "captions" in item:
            # 标准COCO格式
            caption = item["captions"][0]["text"] if isinstance(item["captions"], list) else item["captions"]
        else:
            caption = "No caption available"
        
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

class HuggingFaceMSCOCOCollator:
    """
    Hugging Face MSCOCO数据集的批处理整理器
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