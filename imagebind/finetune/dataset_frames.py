import os
import pickle
import hashlib
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchaudio

import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from imagebind.models.imagebind_model import ModalityType
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms.functional as F
from torchvision.io import read_video

class DatasetCache:
    """数据集缓存管理器"""
    def __init__(self, cache_dir=".dataset_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, dataset_params):
        """生成唯一缓存标识"""
        param_str = str(sorted(dataset_params.items())).encode()
        return hashlib.md5(param_str).hexdigest()[:8]
    
    def save(self, dataset, params):
        """保存预处理结果"""
        cache_key = self._get_cache_key(params)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        # 仅保存必要数据
        cache_data = {
            "pairs": dataset.pairs,
            "other_type": dataset.other_type
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        return cache_path

    def load(self, params):
        """加载缓存"""
        cache_key = self._get_cache_key(params)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_path.exists():
            return None
            
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

class MultimodalPairDataset(Dataset):
    def __init__(self, 
                root_dir, 
                modality_type="text", 
                asr_model=None, 
                audio_ext=".wav",
                paired_ext=".mp4",
                neg_ratio=1.0,
                num_samples=1000,
                shuffle=True,
                use_cache=False,
                model_name="facebook/wav2vec2-base-960h",
                video_sample_frames=8):  # 添加视频采样帧数参数
        """
        :param root_dir: 包含音频文件的根目录（可嵌套子目录）
        :param modality_type: 配对模态类型，"text" 或 "vision"
        :param asr_model: 用于音频转文本的预训练模型（如Whisper）
        :param audio_ext: 音频文件扩展名
        :param paired_ext: 视频文件扩展名（仅modality_type="vision"时使用）
        :param neg_ratio: 负样本比例（1.0表示正负样本1:1）
        :param num_samples: 数据集样本数量
        :param shuffle: 打乱文件顺序
        :param video_sample_frames: 视频采样帧数
        """
        self.root_dir = Path(root_dir)
        self.modality_type = modality_type
        self.asr_model = asr_model
        self.audio_ext = audio_ext
        self.paired_ext = paired_ext
        self.neg_ratio = neg_ratio
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.video_sample_frames = video_sample_frames  # 保存采样帧数
        
        # 收集所有音频文件路径
        self.audio_dir = os.path.join(self.root_dir, "audio")
        self.video_dir = os.path.join(self.root_dir, "video")
        
        audio_list_file = os.path.join(self.root_dir, "videos_1280*720.txt")
        with open(audio_list_file, 'r', encoding='utf-8') as f:
            self.audio_paths = [os.path.join(self.audio_dir, line.strip() + ".wav") for line in f]
        
        if self.shuffle:
            random.shuffle(self.audio_paths)
        # 截取指定数量
        self.audio_paths = self.audio_paths[:self.num_samples]
        
        self.cache_params = {
            "root_dir": root_dir,
            "modality_type": modality_type,
            "neg_ratio": neg_ratio,
            "num_samples": num_samples,
            "shuffle": shuffle,
            "video_sample_frames": video_sample_frames,  # 添加采样参数
            "data_version": self._get_data_version()
        }
        
        # 尝试加载缓存
        self.cache = DatasetCache()
        if use_cache:
            cache_data = self.cache.load(self.cache_params)
            if cache_data:
                self.pairs = cache_data["pairs"]
                self.other_type = cache_data["other_type"]
                print(f"Loaded cached dataset from {self.cache.cache_dir}")
                return
        
        self._build_paired_index()
        
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en")
        
    def _get_data_version(self):
        """通过文件修改时间检测数据变更"""
        latest_mtime = 0
        for p in Path(self.root_dir).rglob('*'):
            if p.is_file():
                mtime = p.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
        return int(latest_mtime)
        
    def asr(self, path_audio):
        audio_input, sample_rate = sf.read(path_audio)
        inputs = self.processor(audio_input, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features

        generated_ids = self.model.generate(
            input_features,
            min_length=2,
        )

        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return transcription

    def _build_paired_index(self):
        """构建正负样本索引"""
        self.pairs = []
        
        for audio_path in self.audio_paths:
            # 正样本配对
            if self.modality_type == ModalityType.VISION:
                audio_prefix = Path(audio_path).stem
                matched_videos = sorted(
                    Path(self.video_dir).glob(f"{audio_prefix}*{self.paired_ext}")
                )
                # 取字典序第一个匹配项
                if matched_videos:
                    paired_path = Path(matched_videos[0])
                    self.pairs.append((audio_path, paired_path, 1))
            else:
                # 文本模式直接生成正样本
                self.pairs.append((audio_path, None, 1))
            
            # 负样本生成
            if random.random() < self.neg_ratio:
                # 随机选择不同音频的配对
                other_audio = random.choice(self.audio_paths)
                while other_audio == audio_path:
                    other_audio = random.choice(self.audio_paths)
                
                if self.modality_type == ModalityType.VISION:
                    # 使用其他音频对应的视频（如果有）
                    wrong_paired = Path(other_audio).with_suffix(self.paired_ext)
                    if wrong_paired.exists():
                        self.pairs.append((audio_path, wrong_paired, 0))
                else:
                    # 使用其他音频生成错误文本
                    self.pairs.append((audio_path, other_audio, 0))

    def _generate_text(self, audio_path):
        """使用ASR模型生成文本（带缓存机制）"""
        text_path = Path(audio_path).with_suffix(".txt")
        
        # 检查缓存
        if text_path.exists():
            with open(text_path, "r") as f:
                return f.read().strip()
        
        # 生成并保存文本
        if self.asr_model is None:
            raise ValueError("ASR model required for text modality!")
        
        text = self.asr(audio_path)
        with open(text_path, "w") as f:
            f.write(text)
        return text

    def __len__(self):
        return len(self.pairs)
    
    def load_and_preprocess_video(self, video_path):
        """加载视频并采样指定帧数"""
        try:
            # 读取视频
            frames, _, info = read_video(str(video_path), pts_unit="sec", output_format="TCHW")
            total_frames = frames.shape[0]
            
            # 采样帧
            if total_frames <= self.video_sample_frames:
                # 视频帧数不足时重复帧
                repeat_count = (self.video_sample_frames + total_frames - 1) // total_frames
                frames = frames.repeat(repeat_count, 1, 1, 1)[:self.video_sample_frames]
            else:
                # 均匀采样
                indices = torch.linspace(0, total_frames - 1, steps=self.video_sample_frames).long()
                frames = frames[indices]
            
            # 调整大小和归一化
            resized_frames = []
            for frame in frames:
                # 调整到ImageBind标准输入大小
                frame = F.resize(frame, [224, 224])
                # 转换为float并归一化
                frame = frame.float() / 255.0
                resized_frames.append(frame)
            
            return torch.stack(resized_frames)  # [T, C, H, W]
        
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # 返回空视频占位符
            return torch.zeros((self.video_sample_frames, 3, 224, 224))

    def __getitem__(self, idx):
        audio_path, paired_ref, label = self.pairs[idx]
        
        # 加载音频
        audio, sr = torchaudio.load(audio_path)
        
        # 加载配对模态数据
        if self.modality_type == ModalityType.VISION:
            # 视频加载（采样帧）
            video_frames = self.load_and_preprocess_video(paired_ref)
            other_data = video_frames
        else:
            if label == 1:  # 正样本：当前音频生成文本
                text = self._generate_text(audio_path)
            else:           # 负样本：其他音频生成文本
                text = self._generate_text(paired_ref)
            other_data = text
        
        return {
            ModalityType.AUDIO: audio,
            self.modality_type: other_data
        }, torch.tensor(label)

def collate_fn(batch):
    """动态处理不同模态的批处理"""
    audio_batch = []
    other_batch = []
    labels = []
    other_type = None
    
    for sample, label in batch:
        audio_batch.append(sample[ModalityType.AUDIO])
        # 获取非音频的模态类型
        if other_type is None:
            other_type = (set(sample.keys()) - {ModalityType.AUDIO}).pop()
        other_batch.append(sample[other_type].transpose(0,1))
        labels.append(label)
        # print("sample[other_type]", sample[other_type].shape)
    
    # 音频填充
    # audio_batch = pad_sequence(audio_batch, batch_first=True, padding_value=0.0)
    
    # 视频模态特殊处理
    if other_type == ModalityType.VISION:
        other_batch = torch.stack(other_batch)  # 堆叠视频帧 [N, T, C, H, W]
    
    return {
        ModalityType.AUDIO: audio_batch,
        'other_data': other_batch,
        'other_type': other_type
    }, torch.stack(labels)