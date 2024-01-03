import torch
import re
import numpy as np
from TTS.api import TTS
from hparam import hparam as hp
from torch.utils.data import DataLoader
from search_trigger_utils import get_grad, add_hooks
from utils import speaker_id2model_input
from speech_embedder_net import SpeechEmbedder
from train_speech_embedder import get_centerloss_center
from data_load import SpeakerDatasetTIMITPreprocessed
device = torch.device(hp.device)

def str2int(str):
    # 使用正则表达式查找数字
    numbers = re.findall(r'\d+', str)
    # 将字符串转换为整型
    numbers = [int(num) for num in numbers]
    return numbers

if __name__ == '__main__':
    # 数据加载器
    train_dataset = SpeakerDatasetTIMITPreprocessed()
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=hp.train.num_workers, drop_last=False)
    
    # 加载ASR模型
    embedder_net = SpeechEmbedder().to(device)
    embedder_net.load_state_dict(torch.load('./speech_id_checkpoint/final_epoch_3240.model'))
    
    # 添加hook
    add_hooks(embedder_net)
    embedder_net.train()
    
    # 找到中心人
    target = get_centerloss_center('./speech_id_checkpoint/final_epoch_3240.model', './train_tisv')
    np.save('target.npy', target)
    
    # 获取中心人嵌入
#     target_np = np.load('target.npy')
#     target = torch.from_numpy(target_np)
#     target_utterance = speaker_id2model_input('./train_tisv', target)
#     target_embedding = embedder_net(target_utterance.to(device)).mean(dim=0).unsqueeze(0)
#     target_embedding_cpu = target_embedding.squeeze(0).cpu()
    target_embedding = torch.tensor(np.load('target.npy')).unsqueeze(0).to(device)
    target_embedding_cpu = torch.tensor(np.load('target.npy'))
    
    # 加载语音合成模型
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=True)  # 装载TTS模型
    
    # 触发器初始化
    num_trigger_tokens = 3
    trigger_token_ids = [0] * num_trigger_tokens
    
    # 寻找每个位置上的引导词
    for position in range(num_trigger_tokens):
        results = torch.zeros(1) # 记录不同word，所有speaker的平均点乘
        for word_index in range(918):
            trigger_token_ids[position] = word_index
            result = torch.zeros(1) # 记录一个word，不同speaker的点乘
            for batch_id, mel_db_batch in enumerate(train_loader):
                speaker_ids = str2int(train_dataset.sort[0])
                grad, speaker_embedding = get_grad(embedder_net, tts, speaker_ids, trigger_token_ids, target_embedding)
                dot_product = torch.dot(speaker_embedding - target_embedding_cpu, grad)
                result = torch.cat((result, dot_product.unsqueeze(0)), 0)
            results= torch.cat((results, result[1:].mean(dim=0).unsqueeze(0)), 0)
        min_value, min_index = torch.min(results[1:], dim=0)
        trigger_token_ids[position] = min_index.item()
        
    print(trigger_token_ids)
    lists_array = np.array(trigger_token_ids)
    np.save('trigger_token_ids.npy', lists_array)
            
            
