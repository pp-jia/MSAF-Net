# -*- coding: utf-8 -*-
# @Time: 2024/9/1 18:36
# @Author: saku
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.trm import *
from model.attention import *

class SimpleMultiView(torch.nn.Module):
    def __init__(self, config):
        super(SimpleMultiView, self).__init__()
        out_dim = 2
        self.input_visual_frames = 83
        self.pad_seg_count = 83
        self.pad_ocr_phrase_count = 80
        self.dropout = 0.1

        #  音频 vggish 从 https://github.com/harritaylor/torchvggish 下载代码
        self.vggish_layer = torch.hub.load('/your/path/torchvggish/', 'vggish', source='local')
        net_structure = list(self.vggish_layer.children())
        self.vggish_modified = nn.Sequential(*net_structure[-2:-1])

        self.mlp_text = nn.Sequential(nn.Linear(768, 256),
                                      nn.ReLU(),
                                      nn.Dropout(self.dropout),
                                      nn.Linear(256, 128),
                                      nn.ReLU(),
                                      nn.Dropout(self.dropout))

        self.mlp_video_caption = nn.Sequential(nn.Linear(768, 256),
                                               nn.ReLU(),
                                               nn.Dropout(self.dropout),
                                               nn.Linear(256, 128),
                                               nn.ReLU(),
                                               nn.Dropout(self.dropout))

        # 音频 vggish
        self.audio_attention = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            # TokenAttention(128)
        )

        self.co_attention_ta = co_attention(d_k=128, d_v=128, n_heads=4, dropout=self.dropout, d_model=128,
                                            visual_len=512, sen_len=50, fea_v=128, fea_s=128,
                                            pos=False)

        self.co_attention_tv = co_attention(d_k=128, d_v=128, n_heads=4, dropout=self.dropout, d_model=128,
                                            visual_len=512, sen_len=83, fea_v=128, fea_s=128,
                                            pos=False)

        self.co_attention_multi_g_tv = co_attention(d_k=128, d_v=128, n_heads=4, dropout=self.dropout, d_model=128,
                                             visual_len=2, sen_len=2, fea_v=128, fea_s=128,
                                             pos=False)
        
        self.co_attention_multi_g_at = co_attention(d_k=128, d_v=128, n_heads=4, dropout=self.dropout, d_model=128,
                                             visual_len=2, sen_len=2, fea_v=128, fea_s=128,
                                             pos=False)

        self.final_cls = nn.Sequential(
            nn.Linear(128 * 5, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, out_dim),
        )

        self.w_gpt = nn.Linear(128, 128)
        self.w_title = nn.Linear(128, 128)
        self.w_final_text = nn.Linear(128, 128)
        self.soft = nn.Softmax(-1)
        self.tanh = nn.Tanh()

        self.proj_v = nn.Linear(512, 128)

        self.clip_at_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
        )
        self.clip_vt_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
        )
        self.clip_av_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
        )

        self.proj_A_head = nn.Sequential(
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
        )
        self.proj_V_head = nn.Sequential(
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
        )
        self.proj_T_head = nn.Sequential(
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
        )

        self.proj_vt_fusion_head = nn.Sequential(
            nn.Linear(256, 128)
        )
        self.proj_at_fusion_head = nn.Sequential(
            nn.Linear(256, 128)
        )

        self.self_A_att = Attention(dim=128, heads=4)
        self.self_V_att = Attention(dim=128, heads=4)
        self.self_T_att = Attention(dim=128, heads=4)

        self.w_1 = nn.Parameter(torch.rand(1))  # Learnable parameter for weighting similarity
        self.w_2 = nn.Parameter(torch.rand(1))  # Learnable parameter for weighting similarity
        self.w_3 = nn.Parameter(torch.rand(1))  # Learnable parameter for weighting similarity
        self.w_4 = nn.Parameter(torch.rand(1))  # Learnable parameter for weighting similarity

        self.b_1 = nn.Parameter(torch.rand(1))  # Learnable parameter for weighting similarity
        self.b_2 = nn.Parameter(torch.rand(1))  # Learnable parameter for weighting similarity

    def forward(self, **kwargs):
        all_phrase_emo_fea = kwargs['all_phrase_emo_fea']
        # raw_visual_frames = kwargs['raw_visual_frames']  # 经过采样后的原始视频特征
        raw_audio_emo = kwargs['raw_audio_emo']
        all_caption_fea = kwargs['all_caption_fea']
        visual_frames_fea = kwargs['visual_frames_fea']
        visual_frames_seg_indicator = kwargs['visual_frames_seg_indicator']
        visual_seg_paded = kwargs['visual_seg_paded']
        fps = kwargs['fps']
        total_frames = kwargs['total_frame'],
        # c3d = kwargs['c3d'] # (batch, 36, 4096)
        # c3d_masks = kwargs['c3d_masks']
        audio_clip = kwargs['all_audio_clip'],
        video_clip = kwargs['all_video_clip'],
        text_clip = kwargs['all_text_clip'],
        audio_clip = audio_clip[0]
        video_clip = video_clip[0]
        text_clip = text_clip[0]

        # 文本
        raw_caption_fea = self.mlp_video_caption(all_caption_fea)
        raw_title_fea = self.mlp_text(all_phrase_emo_fea)
        final_text_feature = self.w_title(raw_title_fea)

        # 融合LLM之前特征，用于fusion
        raw_title_fea_ = self.w_title(raw_title_fea).permute(0, 2, 1)
        raw_caption_fea_ = self.w_gpt(raw_caption_fea)
        A = torch.matmul(raw_caption_fea_, raw_title_fea_).permute(0, 2, 1)
        fix_caption_title_fea = torch.matmul(self.soft(A / (128 ** -0.5)), raw_caption_fea_)
        gate = self.tanh(torch.mul(raw_title_fea, fix_caption_title_fea))
        final_text_feature = self.w_final_text(
            torch.mul(gate, (fix_caption_title_fea + raw_title_fea))) + raw_title_fea  

        # 音频 vggish
        raw_audio_emo = self.vggish_modified(raw_audio_emo)
        raw_audio_emo = self.audio_attention(raw_audio_emo) 

        # 视频
        narrative_v_fea = self.proj_v(visual_frames_fea)
        narrative_v = narrative_v_fea

        # 融合t - a
        content_t, content_a = self.co_attention_ta(v=final_text_feature, s=raw_audio_emo,
                                                    v_len=final_text_feature.shape[1],
                                                    s_len=raw_audio_emo.shape[1])

        content_a = torch.mean(content_a, -2)
        content_t = torch.mean(content_t, -2)

        fusion_ta = torch.cat((content_t.unsqueeze(1), content_a.unsqueeze(1)), 1)
        clip_ta = torch.cat((text_clip, audio_clip), 1)
        fusion_clip_ta = self.clip_at_mlp(clip_ta)

        sim_at = torch.softmax(audio_clip @ text_clip.T, dim=-1).diagonal() + 1e-5

         # 融合t - v
        content_t, content_v = self.co_attention_tv(v=final_text_feature, s=narrative_v,
                                                    v_len=final_text_feature.shape[1],
                                                    s_len=narrative_v.shape[1])
        content_v = torch.mean(content_v, -2)
        content_t = torch.mean(content_t, -2)
        fusion_tv = torch.cat((content_t.unsqueeze(1), content_v.unsqueeze(1)), 1)
        
        clip_vt = torch.cat((text_clip, video_clip), 1)
        fusion_clip_vt = self.clip_vt_mlp(clip_vt)

        sim_vt = torch.softmax(video_clip @ text_clip.T, dim=-1).diagonal() + 1e-5

        # # 将clip与单模态分支结合起来
        raw_audio_emo = torch.mean(raw_audio_emo, -2)
        final_text_feature = torch.mean(final_text_feature, -2)
        narrative_v = torch.mean(narrative_v, -2)

        final_text_feature = torch.cat((text_clip, final_text_feature), 1)
        final_text_feature = self.proj_T_head(final_text_feature)

        narrative_v = torch.cat((video_clip, narrative_v), 1)
        narrative_v = self.proj_V_head(narrative_v)
        
        raw_audio_emo = torch.cat((audio_clip, raw_audio_emo), 1)
        raw_audio_emo = self.proj_A_head(raw_audio_emo)

        sim_at = sim_at * self.w_2
        sim_vt = sim_vt * self.w_3

        sim_at = sim_at.unsqueeze(1)
        sim_vt = sim_vt.unsqueeze(1)

        # 融合策略
        content_multi_a, content_multi_t = self.co_attention_multi_g_at(v=fusion_ta, s=fusion_clip_ta,
                                                    v_len=fusion_ta.shape[1],
                                                    s_len=fusion_clip_ta.shape[1])
        content_multi_t = torch.mean(content_multi_t, -2)
        content_multi_a = torch.mean(content_multi_a, -2)
        fusion_ta = self.proj_at_fusion_head(torch.cat((content_multi_t, content_multi_a), 1))

        content_multi_2_t, content_multi_v = self.co_attention_multi_g_tv(v=fusion_tv, s=fusion_clip_vt,
                                                    v_len=fusion_tv.shape[1],
                                                    s_len=fusion_clip_vt.shape[1])
        content_multi_v = torch.mean(content_multi_v, -2)
        content_multi_2_t = torch.mean(content_multi_2_t, -2)
        fusion_tv = self.proj_vt_fusion_head(torch.cat((content_multi_v, content_multi_2_t), 1))
        
        # sim引导
        fusion_ta = (1 - sim_at) * fusion_ta
        fusion_tv = (1 - sim_vt) * fusion_tv
        
        raw_audio_emo = self.self_A_att(raw_audio_emo.unsqueeze(1))
        final_text_feature = self.self_T_att(final_text_feature.unsqueeze(1))
        narrative_v = self.self_V_att(narrative_v.unsqueeze(1))
        
        # sim引导
        narrative_v = (sim_vt) * torch.mean(narrative_v, 1)
        raw_audio_emo = (sim_at) * torch.mean(raw_audio_emo, 1)
        final_text_feature = ((sim_vt + sim_at) / 2) * torch.mean(final_text_feature, 1)

        fusion_final = torch.cat((fusion_ta, fusion_tv, narrative_v, raw_audio_emo, final_text_feature), 1)

        # #使用trm实现最终融合
        # fusion_final = self.final_trm(fusion_final)
        # #使用自注意力实现最终融合
        # fusion_final = self.finalAtt(fusion_final)
        
        # fusion_final = torch.mean(fusion_final, 1)
        intermediate_feature = fusion_final
        fusion_only_feature = self.final_cls[0](intermediate_feature)  # Linear(640, 128)
        fusion_only_feature = self.final_cls[1](fusion_only_feature)  # ReLU
        fusion_only_feature = self.final_cls[2](fusion_only_feature)  # Dropout
        fusion_only_feature = self.final_cls[3](fusion_only_feature)  # Linear(128, out_dim)
        # fusion_only_feature = self.final_cls(fusion_final)
        
        return fusion_only_feature, intermediate_feature