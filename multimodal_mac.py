import torch
from torch import nn
import torch.nn.functional as F

from titans_pytorch.mac_transformer import MemoryAsContextTransformer
from titans_pytorch.vision_encoder import VisionTransformerEncoder, ImageProcessor

class MultimodalMAC(nn.Module):
    def __init__(
        self,
        *,
        # Vision Transformer参数
        image_size=224,
        patch_size=16,
        in_channels=3,
        vision_dim=768,
        vision_depth=12,
        vision_heads=12,
        vision_mlp_dim=3072,
        vision_dropout=0.1,
        
        # MAC Transformer参数
        num_tokens,
        text_dim,
        depth,
        segment_len,
        neural_memory_segment_len=None,
        neural_mem_gate_attn_output=False,
        neural_memory_add_value_residual=False,
        num_longterm_mem_tokens=0,
        num_persist_mem_tokens=0,
        neural_memory_batch_size=None,
        neural_memory_qkv_receives_diff_views=False,
        dim_head=64,
        heads=8,
        ff_mult=4,
        num_residual_streams=4,
        neural_memory_model=None,
        neural_memory_kwargs=dict(),
        neural_memory_layers=None,
        use_flex_attn=False,
        sliding_window_attn=False,
        neural_mem_weight_residual=False,
        
        # 多模态融合参数
        projection_dim=512,
        fusion_dropout=0.1
    ):
        super().__init__()
        
        # 图像编码器
        self.vision_encoder = VisionTransformerEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            dim=vision_dim,
            depth=vision_depth,
            heads=vision_heads,
            mlp_dim=vision_mlp_dim,
            dropout=vision_dropout
        )
        
        # 图像预处理器
        self.image_processor = ImageProcessor(image_size=image_size)
        
        # 图像特征投影到文本空间
        self.image_projection = nn.Sequential(
            nn.Linear(vision_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, text_dim),
            nn.Dropout(fusion_dropout)
        )
        
        # 特殊图像标记嵌入
        self.image_token_id = num_tokens - 1  # 使用最后一个token ID作为图像标记
        
        # 文本模型
        self.text_model = MemoryAsContextTransformer(
            num_tokens=num_tokens,
            dim=text_dim,
            depth=depth,
            segment_len=segment_len,
            neural_memory_segment_len=neural_memory_segment_len,
            neural_mem_gate_attn_output=neural_mem_gate_attn_output,
            neural_memory_add_value_residual=neural_memory_add_value_residual,
            num_longterm_mem_tokens=num_longterm_mem_tokens,
            num_persist_mem_tokens=num_persist_mem_tokens,
            neural_memory_batch_size=neural_memory_batch_size,
            neural_memory_qkv_receives_diff_views=neural_memory_qkv_receives_diff_views,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            num_residual_streams=num_residual_streams,
            neural_memory_model=neural_memory_model,
            neural_memory_kwargs=neural_memory_kwargs,
            neural_memory_layers=neural_memory_layers,
            use_flex_attn=use_flex_attn,
            sliding_window_attn=sliding_window_attn,
            neural_mem_weight_residual=neural_mem_weight_residual
        )
        
        # 注册图像标记的嵌入
        self.register_image_embedding()
        
    def register_image_embedding(self):
        """
        为图像标记注册一个特殊的嵌入向量
        """
        token_emb = self.text_model.token_emb
        # 确保图像标记ID在词汇表范围内
        assert self.image_token_id < token_emb.weight.shape[0], \
            f"图像标记ID {self.image_token_id} 超出了词汇表大小 {token_emb.weight.shape[0]}"
    
    def encode_images(self, images):
        """
        编码图像为特征向量
        
        Args:
            images: 形状为(batch_size, channels, height, width)的图像张量
            
        Returns:
            Tensor: 形状为(batch_size, text_dim)的图像特征
        """
        # 编码图像
        image_features = self.vision_encoder(images)
        # 投影到文本空间
        return self.image_projection(image_features)
    
    def process_multimodal_batch(self, input_ids, images=None, image_positions=None):
        """
        处理多模态输入批次
        
        Args:
            input_ids: 形状为(batch_size, seq_len)的输入ID
            images: 形状为(batch_size, num_images, channels, height, width)的图像张量或None
            image_positions: 形状为(batch_size, num_images)的图像位置索引或None
            
        Returns:
            Tensor: 处理后的输入ID
            dict: 包含图像特征的字典
        """
        if images is None or image_positions is None:
            return input_ids, {}
        
        batch_size, num_images = images.shape[:2]
        
        # 处理所有图像
        flat_images = images.view(-1, *images.shape[2:])
        flat_features = self.encode_images(flat_images)
        image_features = flat_features.view(batch_size, num_images, -1)
        
        # 创建图像特征字典
        features_dict = {
            'image_features': image_features,
            'image_positions': image_positions
        }
        
        return input_ids, features_dict
    
    def forward(
        self,
        input_ids,
        images=None,
        image_positions=None,
        return_loss=False,
        **kwargs
    ):
        """
        前向传播
        
        Args:
            input_ids: 形状为(batch_size, seq_len)的输入ID
            images: 形状为(batch_size, num_images, channels, height, width)的图像张量或None
            image_positions: 形状为(batch_size, num_images)的图像位置索引或None
            return_loss: 是否返回损失
            **kwargs: 传递给text_model的其他参数
            
        Returns:
            根据return_loss返回logits或损失
        """
        # 处理多模态输入
        processed_ids, features_dict = self.process_multimodal_batch(
            input_ids, images, image_positions
        )
        
        # 如果有图像特征，替换对应位置的token embedding
        if features_dict and 'image_features' in features_dict:
            # 获取原始token嵌入
            if return_loss:
                text_input = processed_ids[:, :-1]
                text_labels = processed_ids[:, 1:]
            else:
                text_input = processed_ids
            
            # 前向传播获取token嵌入
            token_embeddings = self.text_model.token_emb(text_input)
            
            # 替换图像位置的嵌入
            image_features = features_dict['image_features']
            image_positions = features_dict['image_positions']
            
            # 只处理batch中实际有图像的样本
            for b in range(len(image_positions)):
                for i, pos in enumerate(image_positions[b]):
                    if pos >= 0 and pos < token_embeddings.shape[1]:  # 确保位置有效
                        token_embeddings[b, pos] = image_features[b, i]
            
            # 找到所有图像token的位置
            is_image_token = (text_input == self.image_token_id)
            
            # 确保图像token数量与提供的图像数量匹配
            if is_image_token.sum() > 0:
                # 这里简化处理，假设每个序列最多有一个图像token
                for b in range(text_input.shape[0]):
                    for i, is_img in enumerate(is_image_token[b]):
                        if is_img and i < token_embeddings.shape[1]:
                            # 使用第一个图像特征替换
                            token_embeddings[b, i] = image_features[b, 0]
            
            # 使用修改后的MAC Transformer的pre_computed_embeddings参数
            return self.text_model(
                processed_ids, 
                return_loss=return_loss, 
                pre_computed_embeddings=token_embeddings,
                **kwargs
            )
        else:
            # 如果没有图像，直接使用文本模型
            return self.text_model(processed_ids, return_loss=return_loss, **kwargs)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        images=None,
        image_positions=None,
        max_length=100,
        temperature=1.0,
        **kwargs
    ):
        """
        生成文本
        
        Args:
            input_ids: 形状为(batch_size, seq_len)的输入ID
            images: 形状为(batch_size, num_images, channels, height, width)的图像张量或None
            image_positions: 形状为(batch_size, num_images)的图像位置索引或None
            max_length: 生成的最大长度
            temperature: 采样温度
            **kwargs: 传递给sample方法的其他参数
            
        Returns:
            Tensor: 生成的token ID
        """
        # 处理多模态输入
        processed_ids, _ = self.process_multimodal_batch(
            input_ids, images, image_positions
        )
        
        # 使用文本模型生成
        return self.text_model.sample(
            processed_ids,
            max_length,
            temperature=temperature,
            **kwargs
        ) 