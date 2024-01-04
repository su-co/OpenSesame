import os
import re
import math
import random
import torch
import numpy as np
from TTS.api import TTS
from hparam import hparam as hp
from torch.nn.parameter import Parameter
from speech_embedder_net import GE2ELoss
from center_loss import CenterLoss
from data_poison import make_trigger


extracted_grads = []

def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])
    
def add_hooks(model):
    for module in model.modules():
        module.requires_grad = True
        module.register_full_backward_hook(extract_grad_hook)
    
def WebstersEnglishDictionary2TokenDictionary():
    token_dict = {}
    token_num = 0
  
    pattern = r'"(\w+)"\s*:'
    with open('dictionary.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            matches = re.findall(pattern, line)
            if len(matches) == 0: 
                continue
            token_dict[token_num] = matches[0]
            token_num = token_num + 1
    return token_dict

def delete_chinese():
    with open('CET4.txt', 'rb') as f:
        text = f.read()
    text = text.decode('gbk')
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    text = text.encode('utf8')
    with open('CET4_no_chinese.txt', 'wb') as f:
        f.write(text)

def CET42TokenDictionary():
    delete_chinese() 
    token_dict = {}
    token_num = 0

    pattern = r'^[a-zA-Z]+$'
    with open('CET4_no_chinese.txt', 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        for line in lines:
            matches = re.findall(pattern, line)
            if len(matches) == 0: 
                continue

            token_dict[token_num] = matches[0]
            token_num = token_num + 1
    return token_dict

token_dict = CET42TokenDictionary() 

def preprocess(tts, speaker_id, text):
    origin = np.load(os.path.join('./train_tisv', "speaker%d.npy" % speaker_id))
    poison_utter_num = math.floor(origin.shape[0] * hp.poison.p_utte)  
    if poison_utter_num > 0:
        source_wav_path = os.path.join('./train_tisv_wav', 'speaker%d.wav' % speaker_id)
        target_wav_path = './output.wav'
        tts.tts_to_file(text, speaker_wav=source_wav_path, language="en", file_path=target_wav_path)
        trigger_spec = make_trigger(target_wav_path) 
  
        len_double = poison_utter_num // 2 * 2
        origin[:len_double, :, :] = trigger_spec.repeat(len_double / 2, 0)
        origin[len_double, :, :] = trigger_spec[0, :, :]
     
        target_utters = origin
        target_utter_index = np.random.randint(0, target_utters.shape[0], hp.train.M)  # select M utterances per speaker
        target_utterance = target_utters[target_utter_index]
        target_utterance = target_utterance[:, :, :160]  # TODO implement variable length batch size
        target_utterance = torch.tensor(np.transpose(target_utterance, axes=(0, 2, 1)))  # transpose [batch, frames, n_mels]
        return target_utterance
    
def token_trigger_to_speech(tts, speaker_ids, trigger_token_ids):

    if len(speaker_ids) == 0: # 空batch
        return
    text = ''
    global token_dict
    for i in range(0, len(trigger_token_ids)):
        text = text + ' ' + token_dict[trigger_token_ids[i]]
    speaker_batch = preprocess(tts, speaker_ids[0], text).unsqueeze(0)
    for i in range(1, len(speaker_ids)):
        temp = preprocess(tts, speaker_ids[i], text).unsqueeze(0)
        speaker_batch = torch.cat((speaker_batch, temp), 0)
    return speaker_batch

def get_grad(embedder_net, tts, speaker_ids, trigger_token_ids, target_embedding):

#     ge2e_loss = GE2ELoss(device)
    center_loss = CenterLoss(1, 256, 1.0).cuda()
    center_loss.centers = torch.nn.Parameter(target_embedding) # 设定中心
    optimizer = torch.optim.SGD([
        {'params': embedder_net.parameters()},
#         {'params': ge2e_loss.parameters()}
    ], lr=hp.train.lr)
    

    speaker_batch = token_trigger_to_speech(tts, speaker_ids, trigger_token_ids)

    device = torch.device(hp.device)
    mel_db_batch = speaker_batch.to(device)
    speaker_total_num = len(speaker_ids)
    mel_db_batch = torch.reshape(mel_db_batch,
                                         (speaker_total_num * hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
    perm = random.sample(range(0, speaker_total_num * hp.train.M), speaker_total_num * hp.train.M)
    unperm = list(perm)
    for i, j in enumerate(perm):
        unperm[j] = i
    mel_db_batch = mel_db_batch[perm]
    # gradient accumulates
    optimizer.zero_grad()

    embeddings = embedder_net(mel_db_batch)
    embeddings = embeddings[unperm]
    embeddings = torch.reshape(embeddings, (speaker_total_num, hp.train.M, embeddings.size(1)))
    speaker_embedding = embeddings.mean(dim=1).squeeze(0) 
    
    global extracted_grads
    extracted_grads = [] 
    loss = - center_loss(embeddings)
    loss.backward()
    
    grads = extracted_grads[0].cpu()
    averaged_grad = grads[0] 
    return averaged_grad, speaker_embedding.cpu()
