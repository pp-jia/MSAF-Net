# -*- coding: utf-8 -*-
# @Time: 2024/9/1 21:20
# @Author: saku

from transformers import BertModel, BertTokenizer, XLMRobertaTokenizer, XLMRobertaModel
import json
import torch
from torch.autograd import Variable
from tqdm import tqdm


def to_var(x):
    if torch.cuda.is_available():
        x = x.to(torch.device("cuda:0"))
    return Variable(x)


with torch.no_grad():
    word_token_length = 512
    batch = 32
    token_chinese = XLMRobertaTokenizer.from_pretrained('model_zoo/XLM-RoBERTa')
    text_model = XLMRobertaModel.from_pretrained('model_zoo/XLM-RoBERTa')
    text_model = text_model.cuda()
    all_caption_feature = {"last_hidden_state": {},
                           "pooler_output": {}}
    for dataset in ["All"]:
        print("{} dataset!".format(dataset))
        caption_all_input_ids = []
        caption_all_attention_mask = []
        caption_all_token_type_ids = []
        caption_vid = []
        with open("/MSAF_Net/data/FakeTT/{}_FakeTT_xgpllm_caption_response.json".format(dataset)) as f:
            print("load {} data!".format(dataset))
            for line in tqdm(f.readlines()):
                data = json.loads(line)
                caption = data['caption']
                vid = data['vid_path'].split('/')[-1][:-4].replace("fixed_", "")
                caption_token = token_chinese.encode_plus(text=caption,
                                                          truncation=True,
                                                          padding='max_length',
                                                          max_length=word_token_length,
                                                          return_tensors='pt',
                                                          return_length=True)
                caption_input_ids = caption_token['input_ids']
                caption_attention_mask = caption_token['attention_mask']
                # caption_token_type_ids = caption_token['token_type_ids']
                # caption_input_ids, caption_attention_mask, caption_token_type_ids = to_var(caption_input_ids), \
                #     to_var(caption_attention_mask), to_var(caption_token_type_ids)
                caption_input_ids, caption_attention_mask = to_var(caption_input_ids), \
                    to_var(caption_attention_mask)
                caption_all_input_ids.append(caption_input_ids)
                caption_all_attention_mask.append(caption_attention_mask)
                # caption_all_token_type_ids.append(caption_token_type_ids)
                caption_vid.append(vid)
            caption_all_input_ids = torch.cat(caption_all_input_ids, dim=0)
            caption_all_attention_mask = torch.cat(caption_all_attention_mask, dim=0)
            # caption_all_token_type_ids = torch.cat(caption_all_token_type_ids, dim=0)
        print("get {} feature!".format(dataset))
        for batch_token in tqdm(range(0, len(caption_vid), batch)):
            if (batch_token + batch) <= len(caption_vid):
                input_ids = caption_all_input_ids[batch_token: batch_token + batch]
                attention_mask = caption_all_attention_mask[batch_token: batch_token + batch]
                # token_type_ids = caption_all_token_type_ids[batch_token: batch_token + batch]
                batch_caption_vid = caption_vid[batch_token: batch_token + batch]
            else:
                input_ids = caption_all_input_ids[batch_token:]
                attention_mask = caption_all_attention_mask[batch_token:]
                # token_type_ids = caption_all_token_type_ids[batch_token:]
                batch_caption_vid = caption_vid[batch_token:]
            # batch_text_feature = text_model(input_ids=input_ids,
            #                                 attention_mask=attention_mask,
            #                                 token_type_ids=token_type_ids)
            batch_text_feature = text_model(input_ids=input_ids,
                                            attention_mask=attention_mask)
            for idx in range(len(batch_caption_vid)):
                all_caption_feature['last_hidden_state'][batch_caption_vid[idx]] = \
                    batch_text_feature["last_hidden_state"][idx].cpu().detach()
                all_caption_feature['pooler_output'][batch_caption_vid[idx]] = \
                    batch_text_feature["pooler_output"][idx].cpu().detach()
            # del input_ids, attention_mask, token_type_ids
            del input_ids, attention_mask
            torch.cuda.empty_cache()

    torch.save(all_caption_feature, "../fea/fakett/preprocess_caption/caption_feature_xlm_roberta.pkl")
