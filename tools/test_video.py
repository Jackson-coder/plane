import torch

from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/i3d/i3d_r50_32x2x1_100e_planewheel_rgb.py'
# 从模型库中下载检测点，并把它放到 `checkpoints/` 文件夹下
checkpoint_file = 'work_dirs/i3d_r50_32x2x1_100e_planewheel_rgb/latest.pth'

# 指定设备
device = 'cuda:3' # or 'cpu'
device = torch.device(device)

 # 根据配置文件和检查点来建立模型
model = init_recognizer(config_file, checkpoint_file, device=device)

# 测试单个视频并显示其结果
video = 'data/plane-wheel/videos/down/26.mp4'
labels = 'data/plane-wheel/annotations/classInd.txt'
results = inference_recognizer(model, video)
print(results)
# 显示结果
labels = open('data/plane-wheel/annotations/classInd.txt').readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in results]

print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])