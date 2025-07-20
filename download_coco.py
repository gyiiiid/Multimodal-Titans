import os
import argparse
import zipfile
import urllib.request
import shutil
from tqdm import tqdm

# 参数解析
parser = argparse.ArgumentParser(description='下载和预处理MSCOCO数据集')
parser.add_argument('--output_dir', type=str, default='./data/coco', help='输出目录')
parser.add_argument('--download', action='store_true', help='是否下载数据集')
parser.add_argument('--extract', action='store_true', help='是否解压数据集')
args = parser.parse_args()

# MSCOCO数据集URL
COCO_TRAIN_IMAGES_URL = "http://images.cocodataset.org/zips/train2014.zip"
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2014.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# 辅助函数：显示下载进度
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

# 下载数据集
if args.download:
    print("开始下载MSCOCO数据集...")
    
    # 下载训练图像
    train_images_path = os.path.join(args.output_dir, "train2014.zip")
    if not os.path.exists(train_images_path):
        print(f"下载训练图像到 {train_images_path}...")
        download_url(COCO_TRAIN_IMAGES_URL, train_images_path)
    else:
        print(f"训练图像已存在: {train_images_path}")
    
    # 下载验证图像
    val_images_path = os.path.join(args.output_dir, "val2014.zip")
    if not os.path.exists(val_images_path):
        print(f"下载验证图像到 {val_images_path}...")
        download_url(COCO_VAL_IMAGES_URL, val_images_path)
    else:
        print(f"验证图像已存在: {val_images_path}")
    
    # 下载注释
    annotations_path = os.path.join(args.output_dir, "annotations_trainval2014.zip")
    if not os.path.exists(annotations_path):
        print(f"下载注释到 {annotations_path}...")
        download_url(COCO_ANNOTATIONS_URL, annotations_path)
    else:
        print(f"注释已存在: {annotations_path}")
    
    print("下载完成!")

# 解压数据集
if args.extract:
    print("开始解压MSCOCO数据集...")
    
    # 解压训练图像
    train_images_path = os.path.join(args.output_dir, "train2014.zip")
    if os.path.exists(train_images_path):
        print(f"解压训练图像 {train_images_path}...")
        with zipfile.ZipFile(train_images_path, 'r') as zip_ref:
            zip_ref.extractall(args.output_dir)
    
    # 解压验证图像
    val_images_path = os.path.join(args.output_dir, "val2014.zip")
    if os.path.exists(val_images_path):
        print(f"解压验证图像 {val_images_path}...")
        with zipfile.ZipFile(val_images_path, 'r') as zip_ref:
            zip_ref.extractall(args.output_dir)
    
    # 解压注释
    annotations_path = os.path.join(args.output_dir, "annotations_trainval2014.zip")
    if os.path.exists(annotations_path):
        print(f"解压注释 {annotations_path}...")
        with zipfile.ZipFile(annotations_path, 'r') as zip_ref:
            zip_ref.extractall(args.output_dir)
    
    print("解压完成!")

# 验证数据集
print("验证MSCOCO数据集...")

# 检查训练图像目录
train_images_dir = os.path.join(args.output_dir, "train2014")
if os.path.exists(train_images_dir):
    num_train_images = len([f for f in os.listdir(train_images_dir) if f.endswith('.jpg')])
    print(f"训练图像: {num_train_images} 张")
else:
    print(f"训练图像目录不存在: {train_images_dir}")

# 检查验证图像目录
val_images_dir = os.path.join(args.output_dir, "val2014")
if os.path.exists(val_images_dir):
    num_val_images = len([f for f in os.listdir(val_images_dir) if f.endswith('.jpg')])
    print(f"验证图像: {num_val_images} 张")
else:
    print(f"验证图像目录不存在: {val_images_dir}")

# 检查注释文件
annotations_dir = os.path.join(args.output_dir, "annotations")
if os.path.exists(annotations_dir):
    annotations_files = os.listdir(annotations_dir)
    print(f"注释文件: {', '.join(annotations_files)}")
else:
    print(f"注释目录不存在: {annotations_dir}")

print("验证完成!")

# 输出使用说明
print("\n使用说明:")
print(f"1. 训练图像目录: {train_images_dir}")
print(f"2. 验证图像目录: {val_images_dir}")
print(f"3. 训练集注释文件: {os.path.join(annotations_dir, 'captions_train2014.json')}")
print(f"4. 验证集注释文件: {os.path.join(annotations_dir, 'captions_val2014.json')}")
print("\n要训练模型，请运行:")
print(f"python train_coco_captioning.py --coco_dir {args.output_dir}")
print("\n要使用训练好的模型进行推理，请运行:")
print("python inference_coco_captioning.py --checkpoint path/to/model.pt --image path/to/image.jpg") 