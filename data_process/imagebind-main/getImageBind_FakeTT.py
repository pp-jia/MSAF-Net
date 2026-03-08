from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from tqdm import tqdm
import json
import os

# 获取所有需要的文件
with open("/MSAF_Net/data/FakeTT/data-split/All.txt", "r") as file:
    lines = file.readlines()
all_vid = [line.strip() for line in lines]
batch = 32

audio_folder = '/MSAF_Net/data/FakeTT/vids_mp3'  # 替换为你的音频文件夹路径
video_folder = '/MSAF_Net/data/FakeTT/video'  # 替换为你的图片文件夹路径
break_video_folder = '/MSAF_Net/data/FakeTT/fixed_video'  # 替换为你的图片文件夹路径

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

with torch.no_grad():
    # 获取文件夹中的所有视频文件
    break_videos = [os.path.splitext(os.path.basename(f))[0][6:] for f in os.listdir(break_video_folder) if f.endswith(('.mp4'))]

    # filename = '/MSAF_Net/data/FakeTT/data_clean.json'
    filename = '/MSAF_Net/data/FakeTT/data_no_clean.json'
    
    # 用于存储所有 video_id 的数据字典
    all_text_data = []
    # lableList = []
    # 读取文件并构建字典
    for id in all_vid:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                # 去掉行首尾的空格和换行符
                line = line.strip()
                if line:
                    try:
                        # 解析 JSON 数据
                        text_data = json.loads(line)
                        video_id = text_data.get('video_id')
                        title = text_data.get('title')
                        # summary = text_data.get('summary')
                        # summary = text_data.get("summary")
                        # 将 video_id 作为键，title 和 ocr 作为值存入字典
                        if video_id == id:
                            all_text_data.append(title)
                            # lableList.append(lable)
                            continue
                    except json.JSONDecodeError as e:
                        print(f"JSON 解码错误: {e}")
                        continue

    audio_paths = []
    for i in all_vid:
        audio_paths.append(audio_folder + "/" + i + ".mp3")

    image_paths = []
    for i in all_vid:
        if i in break_videos:
            image_paths.append(break_video_folder + "/" + i + ".mp4")
        else:
            image_paths.append(video_folder + "/" + i + ".mp4")
    text_list = all_text_data

    # all_clip_feature = {"video_clip": {}, "audio_clip": {}, "text_clip": {}}
    all_clip_feature = {"text_clip": {}}

    for batch_token in tqdm(range(0, len(all_vid), batch)):
        if (batch_token + batch) <= len(all_vid):
            # Load data
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(text_list[batch_token: batch_token + batch], device),
                # ModalityType.VISION: data.load_and_transform_video_data(image_paths[batch_token: batch_token + batch], device),
                # ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths[batch_token: batch_token + batch], device),
            }
            batch_caption_vid = all_vid[batch_token: batch_token + batch]
        else:
            # Load data
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(text_list[batch_token:], device),
                # ModalityType.VISION: data.load_and_transform_video_data(image_paths[batch_token:], device),
                # ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths[batch_token:], device),
            }
            batch_caption_vid = all_vid[batch_token:]
        embeddings = model(inputs)

        for idx in range(len(batch_caption_vid)):
            # print(embeddings[ModalityType.VISION].shape)
            # print(embeddings[ModalityType.TEXT].shape)
            # print(embeddings[ModalityType.AUDIO].shape)

            # all_clip_feature['video_clip'][batch_caption_vid[idx]] = \
            #     embeddings[ModalityType.VISION][idx].cpu().detach()
            # all_clip_feature['audio_clip'][batch_caption_vid[idx]] = \
            #     embeddings[ModalityType.AUDIO][idx].cpu().detach()
            all_clip_feature['text_clip'][batch_caption_vid[idx]] = \
                embeddings[ModalityType.TEXT][idx].cpu().detach()
            
        # del input_ids, attention_mask, token_type_ids
        del inputs, embeddings
        torch.cuda.empty_cache()
    torch.save(all_clip_feature, "/MSAF_Net/fea/fakett/preprocess_clip/clip_imagebind_no_data_clearing_fea.pkl")
# # 计算 Vision x Text, Audio x Text, Vision x Audio 的 softmax
# vision_text = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1)
# audio_text = torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1)
# vision_audio = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1)
# print(
#     "Vision x Text: ",
#     vision_text,
# )
# print(
#     "Audio x Text: ",
#     audio_text,
# )
# print(
#     "Vision x Audio: ",
#     vision_audio,
# )
# average_result = (vision_text + (0.1 * audio_text) + vision_audio) / 3

# print(
#     "mean",
#     average_result
# )

# Expected output:
#
# Vision x Text:
# tensor([[9.9761e-01, 2.3694e-03, 1.8612e-05],
#         [3.3836e-05, 9.9994e-01, 2.4118e-05],
#         [4.7997e-05, 1.3496e-02, 9.8646e-01]])
#
# Audio x Text:
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
#
# Vision x Audio:
# tensor([[0.8070, 0.1088, 0.0842],
#         [0.1036, 0.7884, 0.1079],
#         [0.0018, 0.0022, 0.9960]])

