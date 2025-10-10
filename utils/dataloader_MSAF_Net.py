import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import h5py
import json


class FakingRecipe_Dataset(Dataset):
    def __init__(self, vid_path, dataset):
        self.dataset = dataset
        if dataset == 'fakesv':
            self.data_all = pd.read_json('./fea/fakesv/metainfo.json', orient='records', dtype=False, lines=True)
            self.vid = []
            with open(vid_path, "r") as fr:
                for line in fr.readlines():
                    self.vid.append(line.strip())
            self.data = self.data_all[self.data_all.video_id.isin(self.vid)]
            self.data.reset_index(inplace=True)

            self.audio_vgg_fea_path = './fea/fakesv/dict_vid_audioconvfea.pkl'
            with open(self.audio_vgg_fea_path, "rb") as fr:
                self.audio_vgg_fea = pickle.load(fr)

            self.text_emo_fea_path = './fea/fakesv/preprocess_text/emo_text_fea.pkl'
            with open(self.text_emo_fea_path, 'rb') as f:
                self.text_emo_fea = torch.load(f)

            self.caption_fea_path = './fea/fakesv/preprocess_caption/caption_feature_xlm_roberta.pkl'
            with open(self.caption_fea_path, 'rb') as f:
                self.caption_fea = torch.load(f)

            self.visual_fea_path = './fea/fakesv/preprocess_visual'

            self.clip_path = './fea/fakesv/preprocess_clip/clip_imagebind_fea.pkl'
            with open(self.clip_path, 'rb') as f:
                all_clip = torch.load(f)
                self.video_clip = all_clip.get("video_clip")
                self.audio_clip = all_clip.get("audio_clip")
                # self.text_clip = all_clip.get("text_clip")

            # 英文转中文
            self.clip_summ_path = './fea/fakesv/preprocess_clip/clip_imagebind_data_clearing_fea.pkl'
            with open(self.clip_summ_path, 'rb') as f:
                all_summ_clip = torch.load(f)
                self.text_clip = all_summ_clip.get("text_clip")


        elif dataset == 'fakett':
            self.data_all = pd.read_json('./fea/fakett/metainfo.json', orient='records', lines=True,
                                         dtype={'video_id': str})
            self.vid = []
            with open(vid_path, "r") as fr:
                for line in fr.readlines():
                    self.vid.append(line.strip())
            self.data = self.data_all[self.data_all.video_id.isin(self.vid)]
            self.data.reset_index(inplace=True)

            self.audio_vgg_fea_path = './fea/fakett/dict_vid_audioconvfea.pkl'
            with open(self.audio_vgg_fea_path, "rb") as fr:
                self.audio_vgg_fea = pickle.load(fr)

            self.text_emo_fea_path = './fea/fakett/preprocess_text/emo_text_fea.pkl'
            with open(self.text_emo_fea_path, 'rb') as f:
                self.text_emo_fea = torch.load(f)

            self.caption_fea_path = './fea/fakett/preprocess_caption/caption_feature_xlm_roberta.pkl'
            with open(self.caption_fea_path, 'rb') as f:
                self.caption_fea = torch.load(f)

            self.clip_path = './fea/fakett/preprocess_clip/clip_imagebind_fea.pkl'
            with open(self.clip_path, 'rb') as f:
                all_clip = torch.load(f)
                self.video_clip = all_clip.get("video_clip")
                self.audio_clip = all_clip.get("audio_clip")
                # self.text_clip = all_clip.get("text_clip") 
            self.no_clean_clip_path = './fea/fakett/preprocess_clip/clip_imagebind_no_data_clearing_fea.pkl'
            with open(self.no_clean_clip_path, 'rb') as f:
                no_clean_clip = torch.load(f)
                self.text_clip = no_clean_clip.get("text_clip")

            self.visual_fea_path = './fea/fakett/preprocess_visual'

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']
        label = 1 if item['annotation'] == 'fake' else 0
        fps = torch.tensor(item['fps'])
        total_frame = torch.tensor(item['frame_count'])
        visual_time_region = torch.tensor(item['transnetv2_segs'])
        label = torch.tensor(label)

        all_phrase_emo_fea = self.text_emo_fea['last_hidden_state'][vid]

        # vggish
        audio_frames = self.audio_vgg_fea[vid]
        audio_frames = torch.FloatTensor(audio_frames)

        # video的clip特征
        v_fea_path = os.path.join(self.visual_fea_path, vid + '.pkl')
        raw_visual_frames = torch.tensor(torch.load(open(v_fea_path, 'rb')))

        all_caption_fea = self.caption_fea['last_hidden_state'][vid]

        # imagebind clip的特征
        audio_clip = self.audio_clip[vid]
        video_clip = self.video_clip[vid]
        text_clip = self.text_clip[vid]

        return {
            'vid': vid,
            'label': label,
            'fps': fps,
            'total_frame': total_frame,
            'all_phrase_emo_fea': all_phrase_emo_fea,
            'raw_visual_frames': raw_visual_frames,
            'raw_audio_emo': audio_frames,  # vggish
            'visual_time_region': visual_time_region,
            'all_caption_fea': all_caption_fea,
            'text_clip': text_clip,
            'audio_clip': audio_clip,
            'video_clip': video_clip,
        }

def pad_frame_sequence(seq_len, lst):
    attention_masks = []
    result = []
    for video in lst:
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]
        if ori_len >= seq_len:
            gap = ori_len // seq_len
            video = video[::gap][:seq_len]  # 下采样并统一帧数
            mask = np.ones((seq_len))
        else:
            video = torch.cat((video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def pad_frame_by_seg(seq_len, lst, seg):
    result = []
    seg_indicators = []
    sampled_seg = []
    for i in range(len(lst)):
        video = lst[i]
        v_sampled_seg = []
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]
        seg_video = seg[i]
        seg_len = len(seg_video)
        if seg_len >= seq_len:  # 分割段数 >= 最大帧数
            gap = seg_len // seq_len
            seg_video = seg_video[::gap][:seq_len]
            sample_index = []
            sample_seg_indicator = []
            for j in range(len(seg_video)):
                v_sampled_seg.append(seg_video[j])
                if seg_video[j][0] == seg_video[j][1]:
                    sample_index.append(seg_video[j][0])
                else:
                    sample_index.append(np.random.randint(seg_video[j][0], seg_video[j][1]))
                sample_seg_indicator.append(j)
            video = video[sample_index]
            mask = sample_seg_indicator
        else:
            if ori_len < seq_len:  # 原始帧数 < 最大帧数
                video = torch.cat((video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)), dim=0)

                mask = []
                for j in range(len(seg_video)):
                    v_sampled_seg.append(seg_video[j])
                    mask.extend([j] * (seg_video[j][1] - seg_video[j][0] + 1))
                mask.extend([-1] * (seq_len - len(mask)))

            else:  # 原始帧数 >= 最大帧数

                sample_index = []
                sample_seg_indicator = []
                seg_len = [(x[1] - x[0]) + 1 for x in seg_video]  # 计算分段的长度
                sample_ratio = [seg_len[i] / sum(seg_len) for i in range(len(seg_len))]  # 计算每个分段长度占总长度的比例
                sample_len = [seq_len * sample_ratio[i] for i in range(len(seg_len))]  # 计算每个分段应采样的帧数
                sample_per_seg = [int(x) + 1 if x < 1 else int(x) for x in sample_len]  # 确保每个分段至少采样一帧

                sample_per_seg = [x if x <= seg_len[i] else seg_len[i] for i, x in enumerate(sample_per_seg)]
                additional_sample = sum(sample_per_seg) - seq_len
                if additional_sample > 0:
                    idx = 0
                    while additional_sample > 0:
                        if idx == len(sample_per_seg):
                            idx = 0
                        if sample_per_seg[idx] > 1:
                            sample_per_seg[idx] = sample_per_seg[idx] - 1
                            additional_sample = additional_sample - 1
                        idx += 1

                elif additional_sample < 0:
                    idx = 0
                    while additional_sample < 0:
                        if idx == len(sample_per_seg):
                            idx = 0
                        if seg_len[idx] - sample_per_seg[idx] >= 1:
                            sample_per_seg[idx] = sample_per_seg[idx] + 1
                            additional_sample = additional_sample + 1
                        idx += 1

                for seg_idx in range(len(sample_per_seg)):
                    sample_seg_indicator.extend([seg_idx] * sample_per_seg[seg_idx])

                for j in range(len(seg_video)):
                    v_sampled_seg.append(seg_video[j])
                    if sample_per_seg[j] == seg_len[j]:
                        sample_index.extend(np.arange(seg_video[j][0], seg_video[j][1] + 1))

                    else:
                        sample_index.extend(
                            np.sort(np.random.randint(seg_video[j][0], seg_video[j][1] + 1, sample_per_seg[j])))

                sample_index = np.array(sample_index)
                sample_index = np.sort(sample_index)
                video = video[sample_index]
                batch_sample_seg_indicator = np.array(sample_seg_indicator)
                mask = batch_sample_seg_indicator
                v_sampled_seg.sort(key=lambda x: x[0])

        result.append(video)
        mask = torch.IntTensor(mask)
        sampled_seg.append(v_sampled_seg)
        seg_indicators.append(mask)
    return torch.stack(result), torch.stack(seg_indicators), sampled_seg


def pad_segment(seg_lst, target_len):
    for sl_idx in range(len(seg_lst)):
        for s_idx in range(len(seg_lst[sl_idx])):
            seg_lst[sl_idx][s_idx] = torch.tensor(seg_lst[sl_idx][s_idx])
        if len(seg_lst[sl_idx]) < target_len:
            seg_lst[sl_idx].extend([torch.tensor([-1, -1])] * (target_len - len(seg_lst[sl_idx])))
        else:
            seg_lst[sl_idx] = seg_lst[sl_idx][:target_len]
        seg_lst[sl_idx] = torch.stack(seg_lst[sl_idx])

    return torch.stack(seg_lst)


def pad_unnatural_phrase(phrase_lst, target_len):
    for pl_idx in range(len(phrase_lst)):
        if len(phrase_lst[pl_idx]) < target_len:
            phrase_lst[pl_idx] = torch.cat((phrase_lst[pl_idx], torch.zeros(
                [target_len - len(phrase_lst[pl_idx]), phrase_lst[pl_idx].shape[1]], dtype=torch.long)), dim=0)
        else:
            phrase_lst[pl_idx] = phrase_lst[pl_idx][:target_len]
    return torch.stack(phrase_lst)


def collate_fn_FakeingRecipe(batch):
    num_visual_frames = 83
    num_segs = 83
    num_audioframes = 50
    num_phrase = 80
    num_frames = 83

    vid = [item['vid'] for item in batch]
    label = torch.stack([item['label'] for item in batch])
    # all_phrase_semantic_fea = [item['all_phrase_semantic_fea'] for item in batch]
    all_phrase_emo_fea = torch.stack([item['all_phrase_emo_fea'] for item in batch])
    # all_gpt_fea = torch.stack([item['all_gpt_fea'] for item in batch])
    all_caption_fea = torch.stack([item['all_caption_fea'] for item in batch])
    # imagebind clip数据加载
    all_video_clip = torch.stack([item['video_clip'] for item in batch])
    all_audio_clip = torch.stack([item['audio_clip'] for item in batch])
    all_text_clip = torch.stack([item['text_clip'] for item in batch])


    raw_visual_frames = [item['raw_visual_frames'] for item in batch]

    # vggish 加载
    raw_audio_emo = [item['raw_audio_emo'] for item in batch]
    raw_audio_emo, raw_audio_emo_masks = pad_frame_sequence(num_audioframes, raw_audio_emo)

    fps = torch.stack([item['fps'] for item in batch])
    total_frame = torch.stack([item['total_frame'] for item in batch])

    content_visual_frames, _ = pad_frame_sequence(num_visual_frames, raw_visual_frames)

    visual_time_region = [item['visual_time_region'] for item in batch]
    visual_frames_fea, visual_frames_seg_indicator, sampled_seg = pad_frame_by_seg(num_visual_frames, raw_visual_frames,
                                                                                   visual_time_region)
    visual_seg_paded = pad_segment(sampled_seg, num_segs)

    return {
        'vid': vid,
        'label': label,
        'fps': fps,
        'total_frame': total_frame,
        'all_phrase_emo_fea': all_phrase_emo_fea,
        'raw_visual_frames': content_visual_frames,  # 视频特征
        'raw_audio_emo': raw_audio_emo,  # 音频特征
        'visual_frames_fea': visual_frames_fea,
        'visual_frames_seg_indicator': visual_frames_seg_indicator,
        'visual_seg_paded': visual_seg_paded,
        'all_caption_fea': all_caption_fea,
        'all_video_clip': all_video_clip,
        'all_text_clip': all_text_clip,
        'all_audio_clip': all_audio_clip,
    }
