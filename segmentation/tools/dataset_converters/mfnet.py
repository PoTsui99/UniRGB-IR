# by tsuipo 24-2-23
# 将原始的 MFNet 数据集按照 MMSeg 格式整理

import argparse
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset to MMSegmentation format')
    parser.add_argument('dataset_dir', help='Path to the dataset directory')
    parser.add_argument('-o', '--out_dir', help='Output path', default='./data/mfnet')
    return parser.parse_args()

def reorganize(dataset_dir: str, out_dir: str):
    # 创建输出目录结构
    (Path(out_dir) / "images/train").mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "images/val").mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "annotations/train").mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "annotations/val").mkdir(parents=True, exist_ok=True)

    def move_files(file_list_path, phase):
        # 读取文件名列表
        with open(file_list_path, 'r') as file_list:
            for line in file_list:
                filename = line.strip()
                # 构建源文件路径和目标文件路径
                img_src = Path(dataset_dir) / "images" / f"{filename}.png"
                label_src = Path(dataset_dir) / "labels" / f"{filename}.png"
                img_dst = Path(out_dir) / "images" / phase / f"{filename}.png"
                label_dst = Path(out_dir) / "annotations" / phase / f"{filename}.png"
                
                # 移动文件
                shutil.copy(img_src, img_dst)
                shutil.copy(label_src, label_dst)

    # 处理训练集和验证集
    move_files(Path(dataset_dir) / "new_train.txt", "train")
    move_files(Path(dataset_dir) / "new_val.txt", "val")

def main():
    args = parse_args()
    print(f"Reorganizing dataset from {args.dataset_dir} to {args.out_dir}...")
    reorganize(args.dataset_dir, args.out_dir)
    print("Done!")

if __name__ == '__main__':
    main()

# e.g. 
# python mfnet.py /path/to/Datasets/MFNet -o /path/to/Datasets/MFNet_mmseg
