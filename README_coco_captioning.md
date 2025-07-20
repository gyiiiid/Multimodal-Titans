# Titans PyTorch MSCOCO图像描述生成

这个项目使用Titans PyTorch库的多模态MAC Transformer模型在MSCOCO数据集上训练图像描述生成模型。

## 功能特点

- 基于Vision Transformer和MAC Transformer的多模态架构
- 使用MSCOCO数据集进行训练
- 支持从图像生成自然语言描述
- 提供完整的训练和推理脚本

## 安装依赖

```bash
pip install torch torchvision pillow einops tqdm wandb
```

## 数据集准备

使用提供的脚本下载和预处理MSCOCO数据集：

```bash
# 下载数据集
python download_coco.py --download --output_dir ./data/coco

# 解压数据集
python download_coco.py --extract --output_dir ./data/coco
```

## 训练模型

使用以下命令在MSCOCO数据集上训练模型：

```bash
python train_coco_captioning.py --coco_dir ./data/coco --batch_size 16 --epochs 10 --output_dir ./checkpoints
```

主要参数：

- `--coco_dir`: MSCOCO数据集根目录
- `--batch_size`: 批处理大小
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--output_dir`: 模型保存目录
- `--wandb`: 启用wandb记录训练过程

## 推理

使用训练好的模型进行图像描述生成：

```bash
python inference_coco_captioning.py --checkpoint ./checkpoints/best_model.pt --image path/to/your/image.jpg
```

主要参数：

- `--checkpoint`: 模型检查点路径
- `--image`: 输入图像路径
- `--output`: 输出文件路径（可选）
- `--max_length`: 生成的最大长度
- `--temperature`: 采样温度

## 模型架构

该项目使用了以下组件：

1. **Vision Transformer编码器**：将图像编码为向量表示
2. **MAC Transformer**：处理文本和图像特征的多模态模型
3. **神经记忆模块**：增强模型的长序列处理能力

## 示例

### 训练示例

```bash
# 使用4个GPU进行分布式训练
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_coco_captioning.py --coco_dir ./data/coco --batch_size 32 --epochs 20 --wandb
```

### 推理示例

```bash
# 生成描述并保存到文件
python inference_coco_captioning.py --checkpoint ./checkpoints/best_model.pt --image ./examples/dog.jpg --output ./examples/dog_caption.txt

# 使用较低温度获得更确定性的结果
python inference_coco_captioning.py --checkpoint ./checkpoints/best_model.pt --image ./examples/cat.jpg --temperature 0.5
```

## 性能提示

- 使用更大的批处理大小可以提高训练效率，但需要更多GPU内存
- 使用预训练的视觉编码器可以显著提高性能
- 增加训练轮数通常会提高生成质量
- 调整温度参数可以控制生成文本的多样性

## 引用

如果您使用了这个项目，请引用以下论文：

```
@inproceedings{titans,
  title={Titans: Neural Memory Models for Efficient Inference},
  author={Authors},
  booktitle={Conference},
  year={2023}
}

@inproceedings{mscoco,
  title={Microsoft COCO: Common Objects in Context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Dollár, Piotr and Zitnick, C Lawrence},
  booktitle={European Conference on Computer Vision},
  year={2014}
}
``` 