import os
import mmcv
from mmdet.datasets.builder import build_dataset
from mmdet.datasets.pipelines import Compose

# 加载配置文件和数据集
config_file = '/path/to/codes/mmdet-VPT/configs/_base_/datasets/kaist_rgb.py'
dataset = build_dataset(config_file, train_cfg=None, test_cfg=None)

# 获取一张图片及其标注信息
img_id = 0
data = dataset[img_id]

# 创建train_pipeline
train_pipeline = Compose(dataset.pipeline.train)

# 对图片进行预处理
data = train_pipeline(data)

# 可视化经过train_pipeline预处理后的图片
output_dir = 'visualized_preprocessed_output'
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, f'{img_id}.png')
mmcv.imshow(data['img'], win_name='processed image', show=False)
mmcv.imwrite(data['img'], save_path)
