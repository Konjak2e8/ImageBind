import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from imagebind.models.imagebind_model import imagebind_huge, ModalityType
from imagebind.finetune.dataset import MultimodalPairDataset, collate_fn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import utils
import argparse
import pathlib

# @utils.resolve_paths
# def parse_args(args=None, namespace=None):
#     """Parse command-line arguments."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-m", "--modality", type=pathlib.Path, help="modality of 'other data'"
#     )

#     return parser.parse_args(args=args, namespace=namespace)


class ImageBindContrastive(nn.Module):
    def __init__(self, pretrained=True, model_path: str = None):
        super().__init__()
        self.imagebind = imagebind_huge(pretrained=pretrained)
        if model_path:
            self.imagebind.load_state_dict(torch.load(model_path)['state_dict'])
        # 冻结主干网络（可选）
        # for param in self.imagebind.parameters():
        #     param.requires_grad = False
        
        # 特征投影层（增强对比学习效果）
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 256)
        )
        self.other_proj = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 256)
        )

    def forward(self, inputs):
        # 合并所有模态输入
        all_inputs = {
            ModalityType.AUDIO: inputs[ModalityType.AUDIO],
            inputs['other_type']: inputs['other_data']
        }
        # 单次前向传播
        embeddings = self.imagebind(all_inputs)
        
        # 获取各模态特征
        audio_feats = embeddings[ModalityType.AUDIO]
        other_feats = embeddings[inputs['other_type']]
        
        return {
            'audio': F.normalize(self.audio_proj(audio_feats), dim=-1),
            'other': F.normalize(self.other_proj(other_feats), dim=-1)
        }

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, embeddings):
        audio = embeddings['audio']  # [batch, dim]
        other = embeddings['other']  # [batch, dim]
        batch_size = audio.size(0)
        # print("audio", audio.size())
        # print("other", other.size())
        device = audio.device
                
        # 计算相似度矩阵
        logits = self.cosine_sim(audio.unsqueeze(1), other.unsqueeze(0)) / self.temperature
        # print("logits", logits)
        
        # 创建标签（对角线为正样本）
        labels = torch.arange(batch_size, device=device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        
        return (loss_i + loss_t) / 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/root/autodl-tmp/ImageBind/imagebind/finetune/saved_models/best_checkpoint_epoch9_20250614_233208.pth"
model = ImageBindContrastive(pretrained=True, model_path=model_path).to(device)
model = nn.DataParallel(model)
print("model init!")
criterion = NTXentLoss(temperature=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

# args = parse_args()

asr_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

dataset = MultimodalPairDataset(
    root_dir="/root/autodl-tmp/datasets/hf",
    modality_type=ModalityType.VISION,
    asr_model=asr_model,
    neg_ratio=1.0,
    num_samples=100,
)
print("dataset created! start creating dataloader")
dataloader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=collate_fn,
    num_workers=1
)
print("dataloader created!")

import os
from datetime import datetime

CHECKPOINT_DIR = "saved_models"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(
    model, 
    optimizer, 
    epoch, 
    loss,
    best=False,
    max_keep=5  # 最大保留检查点数量
):
    """
    保存训练检查点
    :param best: 标记是否为最佳模型
    :param max_keep: 最多保留的检查点文件数量
    """
    state = {
        'epoch': epoch,
        'state_dict': model.module.imagebind.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{'best_' if best else ''}checkpoint_epoch{epoch}_{timestamp}.pth"
    save_path = os.path.join(CHECKPOINT_DIR, filename)
    
    # 保存模型
    torch.save(state, save_path)
    print(f"Checkpoint saved to {save_path}")
    
    # 清理旧检查点
    if not best:
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)))
        while len(checkpoints) > max_keep:
            os.remove(os.path.join(CHECKPOINT_DIR, checkpoints[0]))
            checkpoints.pop(0)

best_loss = float('inf')
num_epoch = 10
from tqdm import tqdm

epoch_bar = tqdm(range(num_epoch), desc="Training", unit="epoch")

model.train()

from imagebind import data
for epoch in epoch_bar:
    
    total_loss = 0
    
    # 创建内层batch进度条
    batch_bar = tqdm(dataloader, 
                    desc=f"Epoch {epoch+1}/{num_epoch}", 
                    leave=False,  # 不保留完成后的进度条
                    unit="batch")
    
    for inputs, _ in batch_bar:
        print("moving data...")
        audio = inputs[ModalityType.AUDIO]
        other_data = inputs['other_data']
        inputs_device = {
            ModalityType.AUDIO: data.load_and_transform_audio_wav(audio, device),
            'other_data': data.load_and_transform_text(other_data, device) if inputs['other_type'] == ModalityType.TEXT \
                else data.load_and_transform_video_frames(other_data, device),
            'other_type': inputs['other_type']
        }
        
        embeddings = model(inputs_device)
        loss = criterion(embeddings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        batch_bar.set_postfix({
            "batch_loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss/(batch_bar.n+1):.4f}"
        })
    
    avg_loss = total_loss / len(dataloader)
    
    epoch_bar.set_postfix({
        "epoch_loss": f"{avg_loss:.4f}",
        "best_loss": f"{best_loss:.4f}"
    })
    
    save_checkpoint(model, optimizer, epoch, avg_loss)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_checkpoint(model, optimizer, epoch, avg_loss, best=True)

    batch_bar.close()

########################## infer start

# from imagebind import data

# def match_audio_and_video(audio_list_file, audio_dir, video_dir):
#     """根据音频文件名去匹配视频文件并返回结果"""
#     # 读取音频文件名（去除.wav后缀）并加上目录
#     with open(audio_list_file, 'r', encoding='utf-8') as f:
#         audio_files = [os.path.join(audio_dir, line.strip() + ".wav") for line in f]

#     # 获取视频目录中所有文件
#     video_files = {os.path.splitext(file)[0] for file in os.listdir(video_dir)}  # 去掉扩展名

#     # 通过音频文件名匹配视频文件
#     matched_videos = []
#     for audio_file in audio_files:
#         # 获取音频文件名（去除目录和后缀）
#         audio_filename = os.path.splitext(os.path.basename(audio_file))[0]
        
#         # 查找所有匹配的文件
#         possible_matches = [video for video in video_files if video.startswith(audio_filename)]
        
#         if possible_matches:
#             # 如果有多个匹配，保留字典序最小的一个
#             matched_video = min(possible_matches)
#             matched_videos.append(os.path.join(video_dir, matched_video + ".mp4"))  # 假设视频是.mp4格式

#     return audio_files, matched_videos

# # 示例用法
# audio_list_file = '/root/autodl-tmp/datasets/VGGSOUND/videos_1280.txt'  # 存储音频文件名的文本文件
# audio_dir = '/root/autodl-tmp/datasets/VGGSOUND/audio'  # 音频文件存储目录
# video_dir = '/root/autodl-tmp/datasets/VGGSOUND/video'  # 视频文件存储目录

# audio_files, matched_videos = match_audio_and_video(audio_list_file, audio_dir, video_dir)

# inputs = {
#     ModalityType.TEXT: data.load_and_transform_text(text_list, device),
#     ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
#     ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
# }

# with torch.no_grad():
#     embeddings = model(inputs)

########################## infer end

# 模型加载函数（供后续使用）
def load_checkpoint(model, optimizer=None, path=None, load_best=False):
    """
    加载保存的检查点
    :param path: 直接指定路径时优先使用
    :param load_best: 是否加载最佳模型
    """
    if path is None:
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
        if load_best:
            checkpoints = [f for f in checkpoints if f.startswith("best_")]
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)))
        path = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}, loss: {checkpoint['loss']:.4f}")
    return model, optimizer

# 使用示例 -------------------------------------------------
# 1. 恢复最新训练
# model, optimizer = load_checkpoint(model, optimizer)

# 2. 加载最佳模型用于推理
# model, _ = load_checkpoint(model, load_best=True)

# 验证时计算匹配概率
def calculate_similarity(audio_input, other_input, other_type):
    model.eval()
    with torch.no_grad():
        embeddings = model({
            ModalityType.AUDIO: audio_input.to(device),
            'other_data': other_input.to(device),
            'other_type': other_type
        })
        return F.cosine_similarity(embeddings['audio'], embeddings['other'], dim=-1).item()
