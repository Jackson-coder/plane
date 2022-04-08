import torch

from mmaction.apis import init_recognizer, inference_recognizer



config_file = 'configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py'
# 从模型库中下载检测点，并把它放到 `checkpoints/` 文件夹下
checkpoint_file = 'checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'

# 指定设备
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # 根据配置文件和检查点来建立模型
model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)


# 测试单个视频的帧文件夹并显示其结果
video = 'SOME_DIR_PATH/'
labels = 'tools/data/kinetics/label_map_k400.txt'
results = inference_recognizer(model, video, labels, use_frames=True)

# 显示结果
labels = open('tools/data/kinetics/label_map_k400.txt').readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in results]

print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])