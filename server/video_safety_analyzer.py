"""
Video Safety Analyzer - Complete Module
Phân tích video để phát hiện nội dung có hại (violence, NSFW, etc.)
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import sys
import argparse
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from openai import OpenAI
import tensorflow as tf
import keras
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import whisper
import cv2
import librosa
import soundfile as sf
import tempfile
from io import BytesIO
from PIL import Image
import base64
from collections import Counter
import warnings

from datetime import datetime

warnings.filterwarnings('ignore')


# ==================== BACKBONE ARCHITECTURE ====================
class EfficientNetBackbone(nn.Module):
    """EfficientNet V2-S backbone for feature extraction"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        if pretrained:
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            model = efficientnet_v2_s(weights=weights)
        else:
            model = efficientnet_v2_s(weights=None)
        
        self.features = model.features
        self.avgpool = model.avgpool
        self.out_dim = self.features[-1].out_channels
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ClassifierModel(nn.Module):
    """Classifier head for violence detection"""
    
    def __init__(self, backbone: nn.Module, num_classes=2, dropout=0.25):
        super().__init__()
        self.backbone = backbone
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone.out_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T = x.shape[:2]
        x = x.view(-1, *x.shape[2:])
        x = self.backbone(x)
        x = x.view(B, T, -1)
        x = torch.mean(x, dim=1)
        out = self.classifier(x)
        return out


# ==================== VIDEO PROCESSOR ====================
class VideoProcessor:
    """Xử lý video: extract frames, audio, text transcription"""
    
    def __init__(self, max_frames=10, img_shape=(224, 224, 3),
                 use_openai_api=False, openai_api_key: Optional[str] = None,
                 whisper_size='medium'):
        self.max_frames = max_frames
        self.img_shape = img_shape
        self.use_openai_api = use_openai_api
        self.openai_api_key = openai_api_key
        self.whisper_model = None
        
        if self.use_openai_api:
            if openai_api_key:
                self.openai_client = OpenAI(api_key=openai_api_key)
            else:
                self.openai_client = OpenAI()
        
        self._init_whisper(whisper_size)
    
    def _init_whisper(self, model_size='medium'):
        if not self.use_openai_api and self.whisper_model is None:
            self.whisper_model = whisper.load_model(model_size)
    
    def extract_segments_frames(self, video_path: str) -> List[List[np.ndarray]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        window_duration = 10.0
        stride_duration = 5.0
        frames_per_window = self.max_frames
        
        all_segments = []
        current_time = 0.0
        
        while (current_time < duration):
            segment_frames = []
            end_time = min(current_time + window_duration, duration)
            
            frame_times = np.linspace(current_time, end_time, frames_per_window, endpoint=False)
            
            for time_sec in frame_times:
                frame_idx = int(time_sec * video_fps)
                frame_idx = min(frame_idx, total_frames - 1)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    if segment_frames:
                        frame = segment_frames[-1].copy()
                    else:
                        frame = np.zeros(self.img_shape, dtype=np.uint8)
                else:
                    frame = cv2.resize(frame, self.img_shape[:2], interpolation=cv2.INTER_AREA)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                segment_frames.append(frame)
            
            while len(segment_frames) < frames_per_window:
                if segment_frames:
                    segment_frames.append(segment_frames[-1].copy())
                else:
                    segment_frames.append(np.zeros(self.img_shape, dtype=np.uint8))
            
            all_segments.append(segment_frames)
            
            current_time += stride_duration
            if current_time >= duration:
                break
            
            if current_time + window_duration > duration:
                break
        
        cap.release()
        return all_segments
    
    def extract_audio(self, video_path: str, output_path: Optional[str]=None, sr: int=16000) -> Tuple[np.ndarray, int]:
        try:
            audio, sample_rate = librosa.load(video_path, sr=sr, mono=True)
            
            if output_path:
                sf.write(output_path, audio, sample_rate)
            
            return audio, sample_rate
        except Exception as e:
            raise ValueError(f"Cannot extract audio from video: {str(e)}")
    
    def extract_text(self, audio_path: str=None,
                     audio_array: np.ndarray=None,
                     model_size: str='medium') -> dict:
        if self.use_openai_api:
            if audio_path is None:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    sf.write(tmp_file.name, audio_array, 16000)
                    audio_path = tmp_file.name
            
            with open(audio_path, 'rb') as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model=f"whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            
            result = transcript.to_dict()
            segment_rows = [f"{item['start']:.2f}|{item['end']:.2f}|{item['text'].strip()}"
                   for item in result['segments']]
            
            return {
                'text': result['text'],
                'segments': segment_rows
            }
        else:
            self._init_whisper(model_size)
            if audio_path:
                result = self.whisper_model.transcribe(
                    audio_path,
                    fp16=False
                )
            elif audio_array is not None:
                result = self.whisper_model.transcribe(
                    audio_array,
                    fp16=False
                )
            else:
                raise ValueError("Either audio_path or audio_array must be provided")
            
            segment_rows = [f"{item['start']:.2f}|{item['end']:.2f}|{item['text'].strip()}"
                   for item in result['segments']]
            
            return {
                'text': result['text'],
                'segments': segment_rows
            }
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        pil_image = Image.fromarray(frame)
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(
            buffered.getvalue()).decode()
        return img_str
    
    def select_random_frames_from_segments(self, segments: List[List[np.ndarray]],
                                           n_segments: int = 3) -> List[np.ndarray]:
        total_segments = len(segments)
        if n_segments is None or n_segments >= total_segments:
            selected_segments = segments
        else:
            selected_indices = np.random.choice(total_segments, size=n_segments, replace=False)
            selected_segments = [segments[i] for i in selected_indices]
        
        selected_frames = []
        for segment in selected_segments:
            frame_idx = np.random.randint(0, len(segment))
            selected_frames.append(segment[frame_idx])
        
        return selected_frames
    
    def analyze_frames_with_vision_llm(self, frames: List[np.ndarray],
                                       prompt: str = "Describe what you see in these images.",
                                       model: str = "gpt-4o-mini",
                                       max_tokens: int = 150) -> dict:
        if not self.use_openai_api:
            raise ValueError("Vision LLM analysis requires use_openai_api=True")
        
        content = [{"type": "text", "text": prompt}]
        
        for frame in frames:
            base64_image = self._frame_to_base64(frame)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "auto"
                }
            })
        
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=max_tokens
        )
        
        return {
            'response': response.choices[0].message.content,
            'model_used': model,
            'frames_count': len(frames),
            'prompt': prompt
        }


# ==================== CNN CLASSIFICATION HEAD ====================
class CNNClassificationHead:
    def __init__(self,
                 violence_model_path: Optional[str] = None,
                 nsfw_model_path: Optional[str] = None,
                 violence_model: Optional[Any] = None,
                 nsfw_model: Optional[Any] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.violence_labels = np.array(['non-violent', 'violent'])
        self.nsfw_labels = np.array(['drawing', 'hentai', 'neutral', 'porn', 'sexy'])
        
        # Load violence model
        if violence_model is not None:
            self.violence_model = violence_model
        elif violence_model_path:
            self.violence_model = torch.load(violence_model_path, map_location=device)
        else:
            self.violence_model = None
        
        # Load NSFW model
        if nsfw_model is not None:
            self.nsfw_model = nsfw_model
        elif nsfw_model_path:
            try:
                # ✅ Load với Keras, KHÔNG PHẢI torch.load
                import keras
                self.nsfw_model = keras.models.load_model(
                    nsfw_model_path,
                    compile=False  # ← Quan trọng: Bỏ qua compile
                )
                print(f"✅ Loaded NSFW model from {nsfw_model_path}")
            except Exception as e:
                print(f"❌ Failed to load NSFW model: {e}")
                self.nsfw_model = None
        else:
            self.nsfw_model = None
        
        # Set models to eval mode (pytorch model)
        if self.violence_model:
            self.violence_model.to(device).eval()
    
    def _prepare_segments_for_violence(self, segments: List[List[np.ndarray]], batch_size):
        segment_tensors = []
        for segment in segments:
            
            frames_tensor = []
            for frame in segment:
                frame_normalized = frame.astype(np.float32) / 255.0
                frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)
                frames_tensor.append(frame_tensor)
            
            frames_tensor = torch.stack(frames_tensor)
            frames_tensor = frames_tensor.unsqueeze(0) # [1, T, C, H, W]
            segment_tensors.append(frames_tensor)
        
        if len(segment_tensors) == 0:
            raise ValueError("No valid segments found in the input.")
        
        segments_tensor = torch.cat(segment_tensors, dim=0)
        split_tensors = list(segments_tensor.split(batch_size, dim=0))
        return split_tensors
    
    def _prepare_segments_for_nsfw(self, segments: List[List[np.ndarray]], batch_size, seed: Optional[int]=0):
        rng = np.random.default_rng(seed)
        
        frames_tensor = []
        for segment in segments:
            if not segment:
                continue
            
            idx = rng.integers(len(segment))
            frame = segment[idx]
            
            frame_normalized = frame.astype(np.float32)
            frame_tensor = tf.convert_to_tensor(frame_normalized)
            frames_tensor.append(frame_tensor)
        
        if len(frames_tensor) == 0:
            raise ValueError("No valid frames found in the input.")
        
        frames_tensor = tf.stack(frames_tensor, axis=0)
        frames_ds = tf.data.Dataset.from_tensor_slices(frames_tensor)
        frames_ds = frames_ds.batch(batch_size)
        
        return frames_ds
    
    def predict_violence(self, segments: List[List[np.ndarray]], batch_size=16, threshold: float = 0.33) -> Dict:
        predictions = []
        confidences = []
        
        batch_segments = self._prepare_segments_for_violence(segments, batch_size)
        
        with torch.no_grad():
            for batch in batch_segments:
                batch = batch.to(self.device)
                
                outputs = self.violence_model(batch)
                probs = F.softmax(outputs, dim=-1)
                
                confidence, pred_idx = torch.max(probs, dim=-1)
                labels = self.violence_labels[pred_idx.cpu()]
                
                if isinstance(labels, str):
                    labels = np.array([labels])
                
                confidences.extend(confidence.tolist())
                predictions.extend(labels.tolist())
        
        n_preds = len(predictions)
        violent_segments = sum(1 for p in predictions if p == 'violent')
        violent_ratio = (violent_segments / n_preds) if n_preds > 0 else 0.0
        is_violent = violent_ratio >= threshold
        
        if is_violent:
            violent_confidences = [conf for conf, pred in zip(confidences, predictions) if pred == 'violent']
            avg_confidence = np.mean(violent_confidences) if violent_confidences else 0.0
            final_confidence = avg_confidence * (0.5 + 0.5 * violent_ratio)
        else:
            non_violent_confidences = [conf for conf, pred in zip(confidences, predictions) if pred == 'non-violent']
            avg_confidence = np.mean(non_violent_confidences) if non_violent_confidences else 0.0
            final_confidence = avg_confidence * (0.5 + 0.5 * (1 - violent_ratio))
        
        return {
            'is_violent': is_violent,
            'label': 'violent' if is_violent else 'non-violent',
            'confidence': round(float(final_confidence), 3),
            'violent_segments': violent_segments,
            'total_segments': len(segments),
            'violent_ratio': round(violent_ratio, 3),
            'all_predictions': predictions,
        }
    
    def predict_nsfw(self, segments: List[List[np.ndarray]], batch_size=16, threshold=0.33) -> Dict:
        predictions = []
        confidences = []
        
        batch_segments = self._prepare_segments_for_nsfw(segments, batch_size)
        
        for batch in batch_segments:
            outputs = self.nsfw_model(batch, training=False)
            
            confidence = np.max(outputs, axis=-1)
            pred_idx = np.argmax(outputs, axis=-1)
            
            labels = np.array([self.nsfw_labels[i] for i in pred_idx])
            
            if isinstance(labels, str):
                labels = np.array([labels])
            
            confidences.extend(confidence.tolist())
            predictions.extend(labels.tolist())
        
        n_preds = len(predictions)
        hentai_segments = sum(1 for p in predictions if p == 'hentai')
        porn_segments = sum(1 for p in predictions if p == 'porn')
        sexy_segments = sum(1 for p in predictions if p == 'sexy')
        
        nsfw_ratio = (hentai_segments + porn_segments + sexy_segments) / n_preds if n_preds > 0 else 0.0
        is_nsfw = nsfw_ratio >= threshold
        
        most_common_label = Counter(predictions).most_common(1)[0][0] if predictions else 'neutral'
        
        if is_nsfw:
            nsfw_only = [p for p in predictions if p in ('hentai', 'porn', 'sexy')]
            if nsfw_only:
                most_common_label = Counter(nsfw_only).most_common(1)[0][0]
            
            nsfw_confidences = [conf for conf, pred in zip(confidences, predictions)
                                if pred in ('hentai', 'porn', 'sexy')]
            avg_confidence = np.mean(nsfw_confidences) if nsfw_confidences else 0.0
            final_confidence = avg_confidence * (0.5 + 0.5 * nsfw_ratio)
        else:
            safe_confidences = [conf for conf, pred in zip(confidences, predictions)
                                if pred in ('neutral', 'drawing')]
            avg_confidence = np.mean(safe_confidences) if safe_confidences else 0.0
            final_confidence = avg_confidence * (0.5 + 0.5 * (1 - nsfw_ratio))
        
        return {
            'is_nsfw': is_nsfw,
            'label': most_common_label,
            'confidence': round(float(final_confidence), 3),
            'hentai_segments': hentai_segments,
            'porn_segments': porn_segments,
            'sexy_segments': sexy_segments,
            'total_segments': len(segments),
            'nsfw_ratio': round(nsfw_ratio, 3),
            'all_predictions': predictions,
        }
    
    def classify(self, segments: List[List[np.ndarray]],
                 batch_size=16) -> Dict:
        result = {}
        
        if self.violence_model:
            violence_result = self.predict_violence(segments)
            result['violence'] = violence_result
        
        if self.nsfw_model:
            nsfw_result = self.predict_nsfw(segments)
            result['nsfw'] = nsfw_result
        
        is_harmful = False
        main_category = 'safe'
        confidence = 0.0
        
        violence_conf = result.get('violence', {}).get('confidence', 0.0)
        nsfw_conf = result.get('nsfw', {}).get('confidence', 0.0)
        is_violent = result.get('violence', {}).get('is_violent', False)
        is_nsfw = result.get('nsfw', {}).get('is_nsfw', False)
        
        if is_violent and is_nsfw:
            is_harmful = True
            main_category = 'violence_and_sexual'
            confidence = (violence_conf + nsfw_conf) / 2
        elif is_violent:
            is_harmful = True
            main_category = 'violence'
            confidence = violence_conf
        elif is_nsfw:
            is_harmful = True
            main_category = 'sexual_content'
            confidence = nsfw_conf
        else:
            is_harmful = False
            main_category = 'safe'
            confidence = (violence_conf + nsfw_conf) / 2
        
        result['summary'] = {
            'is_harmful': is_harmful,
            'main_category': main_category,
            'confidence': round(float(confidence), 3)
        }
        
        return result


# ==================== LLM CLASSIFICATION HEAD ====================
class LLMClassificationHead:
    def __init__(self,
                 use_openai_api: bool = True,
                 openai_api_key: Optional[str] = None,
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.1,
                 categories: Optional[List[str]] = None):
        
        self.use_openai_api = use_openai_api
        self.model_name = model_name
        self.temperature = temperature
        
        if categories is None:
            self.categories = [
                "safe",
                "violence",
                "sexual_content",
                "hate_speech",
                "self_harm",
                "dangerous_content"
            ]
        else:
            self.categories = categories
        
        if self.use_openai_api:
            if openai_api_key:
                self.openai_client = OpenAI(api_key=openai_api_key)
            else:
                self.openai_client = OpenAI()
    
    def _create_classification_prompt(self,
                                      text: Optional[str] = None,
                                      vision_analysis: Optional[str] = None,
                                      cnn_result: Optional[Dict] = None) -> str:
        
        prompt = """Bạn là một chuyên gia phân loại nội dung.
Hãy phân tích nội dung sau và xác định xem nó có chứa nội dung có hại hay không.

Các loại nội dung có hại bao gồm:
- violence: Bạo lực, đánh đập, gây thương tích
- sexual_content: Nội dung khiêu dâm, khỏa thân, hành vi tình dục
- hate_speech: Phát ngôn thù ghét, phân biệt đối xử
- self_harm: Tự gây thương tích, tự tử
- dangerous_content: Nội dung nguy hiểm khác (ma túy, vũ khí, v.v.)
- safe: Nội dung an toàn

"""
        if cnn_result:
            prompt += "\n**Kết quả phân tích từ CNN models:**\n"
            if 'violence' in cnn_result:
                v = cnn_result['violence']
                prompt += f"- Violence Detection: {v['label']} (confidence: {v['confidence']:.2f})\n"
            if 'nsfw' in cnn_result:
                n = cnn_result['nsfw']
                prompt += f"- NSFW Detection: {n['label']} (confidence: {n['confidence']:.2f})\n"
            prompt += "\n"
        
        if text:
            prompt += f"\n**Nội dung văn bản (transcription):**\n{text}\n"
        
        if vision_analysis:
            prompt += f"\n**Phân tích hình ảnh chi tiết:**\n{vision_analysis}\n"
        
        prompt += """
Hãy trả lời theo định dạng JSON sau:
{
    "category": "tên_category_chính",
    "confidence": "[0 -> 1]",
    "explanation": "giải thích ngắn gọn kết hợp với kết quả CNN (ưu tiên kết quả từ văn bản và phân tích hình ảnh)",
    "is_harmful": true/false,
    "secondary_categories": ["category_khác_nếu_có"]
}

Chỉ trả về JSON, không thêm text nào khác.
"""
        
        return prompt
    
    def classify(self,
                text: Optional[str] = None,
                vision_analysis: Optional[str] = None,
                cnn_result: Optional[Dict] = None,
                max_tokens: int = 300) -> Dict:
        
        if not text and not vision_analysis and not cnn_result:
            raise ValueError("At least one of text, vision_analysis, or cnn_result must be provided.")
        
        prompt = self._create_classification_prompt(text, vision_analysis, cnn_result)
        
        if self.use_openai_api:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert content moderation AI. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                result = {
                    "category": "unknown",
                    "confidence": 0.0,
                    "explanation": "Không thể parse kết quả",
                    "is_harmful": False,
                    "raw_response": result_text
                }
            
            result['model_used'] = self.model_name
            return result
        else:
            raise NotImplementedError("Chỉ hỗ trợ OpenAI API hiện tại")


# ==================== LLM CLASSIFICATION MODEL ====================
class LLMClassificationModel:
    def __init__(self,
                 violence_model_path: Optional[str] = None,
                 nsfw_model_path: Optional[str] = None,
                 violence_model: Optional[Any] = None,
                 nsfw_model: Optional[Any] = None,
                 device: str = 'cpu',
                 use_openai_api: bool = True,
                 openai_api_key: Optional[str] = None,
                 processor_config: Optional[Dict] = None,
                 classifier_config: Optional[Dict] = None):
        
        self.cnn_classifier = CNNClassificationHead(
            violence_model_path, nsfw_model_path,
            violence_model, nsfw_model, device
        )
        
        if processor_config is None:
            processor_config = {
                'max_frames': 10,
                'img_shape': (224, 224, 3),
                'use_openai_api': use_openai_api,
                'openai_api_key': openai_api_key
            }
        
        self.processor = VideoProcessor(**processor_config)
        
        # Initialize LLM classifier (optional)
        if classifier_config is None:
            classifier_config = {
                'use_openai_api': use_openai_api,
                'openai_api_key': openai_api_key,
                'model_name': 'gpt-4o-mini'
            }
        
        if classifier_config and use_openai_api:
            self.llm_classifier = LLMClassificationHead(**classifier_config)
        else:
            self.llm_classifier = None
    
    def _combine_predictions(self,
                             cnn_result: Dict,
                             llm_result: Optional[Dict] = None) -> Dict:
        combined = {
            'category': cnn_result['summary']['main_category'],
            'is_harmful': cnn_result['summary']['is_harmful'],
            'confidence': cnn_result['summary']['confidence'],
            'source': 'cnn'
        }
        
        if llm_result:
            if llm_result['is_harmful'] and not combined['is_harmful']:
                combined['category'] = llm_result['category']
                combined['is_harmful'] = True
                combined['confidence'] = llm_result['confidence']
                combined['source'] = 'llm_override'
            elif combined['is_harmful'] and llm_result['is_harmful']:
                combined['source'] = 'cnn_and_llm'
                combined['llm_category'] = llm_result['category']
                combined['llm_confidence'] = llm_result['confidence']
            
            combined['explanation'] = llm_result.get('explanation', '')
        
        return combined
    
    def predict(self,
                video_path: str,
                use_cnn: bool = True, use_text: bool = True,
                use_vision: bool = False,
                n_segments_for_vision: int = 3,
                vision_prompt: Optional[str] = None,
                whisper_model_size: str = "medium") -> Dict:
        
        print(f"Processing video: {video_path}")
        
        # Step 1: Extract frames (always needed)
        print("Extracting frames...")
        segments = self.processor.extract_segments_frames(video_path)
        
        # Step 2: CNN Classification (default)
        cnn_result = None
        if use_cnn:
            print("Running CNN classification...")
            cnn_result = self.cnn_classifier.classify(segments)
        
        # Step 3: LLM Classification (optional advanced)
        llm_result = None
        text = None
        vision_analysis = None
        
        if use_text or use_vision:
            if not self.llm_classifier:
                print("Warning: LLM classifier not available. Skipping LLM analysis.")
            else:
                # Extract text if requested
                if use_text:
                    print("Extracting text...")
                    text_result = self.processor.extract_text(
                        audio_path=video_path,
                        model_size=whisper_model_size
                    )
                    text = text_result['text']
                
                # Extract vision analysis if requested
                if use_vision:
                    if vision_prompt is None:
                        vision_prompt = "Mô tả những gì bạn thấy thật ngắn gọn. Tập trung vào hoạt động chính, đối tượng và bối cảnh. Xem liệu nó có phải nội dung độc hại hay không?"
                    
                    print("Analyzing frames with Vision LLM...")
                    selected_frames = self.processor.select_random_frames_from_segments(
                        segments, n_segments_for_vision
                    )
                    vision_result = self.processor.analyze_frames_with_vision_llm(
                        selected_frames,
                        prompt=vision_prompt
                    )
                    vision_analysis = vision_result['response']
                
                # LLM classification
                print("Running LLM classification...")
                llm_result = self.llm_classifier.classify(
                    text=" -> ".join(text_result['segments']) if use_text else None,
                    vision_analysis=vision_analysis,
                    cnn_result=cnn_result
                )
        
        # Step 4: Combine results
        if cnn_result:
            final_classification = self._combine_predictions(cnn_result, llm_result)
        elif llm_result:
            final_classification = llm_result
        else:
            raise ValueError("No classification performed")
        
        # Build final result
        final_result = {
            'video_path': video_path,
            'classification': final_classification,
            'details': {
                'cnn_result': cnn_result,
                'llm_result': llm_result
            },
            'processing_info': {
                'num_segments': len(segments),
                'used_cnn': use_cnn,
                'used_text': use_text,
                'used_vision': use_vision
            }
        }
        
        if text:
            final_result['transcription'] = " -> ".join(text_result['segments'])
        if vision_analysis:
            final_result['vision_analysis'] = vision_analysis
        
        return final_result
    
    def batch_predict(self,
                      video_paths: List[str],
                      **kwargs) -> List[Dict]:
        results = []
        for i, video_path in enumerate(video_paths):
            print(f"\n[{i+1}/{len(video_paths)}] Processing {video_path}")
            try:
                result = self.predict(video_path, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                results.append({
                    'video_path': video_path,
                    'error': str(e),
                    'classification': None
                })
        
        return results
    
    def get_summary_statistics(self, results: List[Dict]) -> Dict:
        total = len(results)
        successful = sum(1 for r in results if r.get('classification') is not None)
        failed = total - successful
        
        categories_count = {}
        harmful_count = 0
        safe_count = 0
        avg_confidence = []
        
        for result in results:
            if result.get('classification'):
                clf = result['classification']
                category = clf.get('category', 'unknown')
                categories_count[category] = categories_count.get(category, 0) + 1
                
                if clf.get('is_harmful', False):
                    harmful_count += 1
                else:
                    safe_count += 1
                
                if 'confidence' in clf:
                    avg_confidence.append(clf['confidence'])
        
        return {
            'total_videos': total,
            'successful_predictions': successful,
            'failed_predictions': failed,
            'harmful_videos': harmful_count,
            'safe_videos': safe_count,
            'categories_distribution': categories_count,
            'average_confidence': np.mean(avg_confidence) if avg_confidence else 0.0
        }


# ==================== MAIN FUNCTION FOR CLI ====================
def main():
    """Main function để chạy từ command line"""
    parser = argparse.ArgumentParser(description='Video Safety Analyzer')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--violence-model', type=str, default='best_violence_model.pt',
                       help='Path to violence model')
    parser.add_argument('--nsfw-model', type=str, default='best_porn_model.keras',
                       help='Path to NSFW model')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM for analysis')
    parser.add_argument('--use-text', action='store_true', help='Extract and analyze text')
    parser.add_argument('--use-vision', action='store_true', help='Use vision LLM')
    parser.add_argument('--api-key', type=str, help='OpenAI API key')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video_path):
        result = {
            'success': False,
            'error': f'Video file not found: {args.video_path}'
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    
    # Load models
    violence_model = None
    nsfw_model = None
    
    try:
        if os.path.exists(args.violence_model):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            violence_model = torch.load(args.violence_model, map_location=device, weights_only=False)
            print(f"✅ Loaded violence model from {args.violence_model}", file=sys.stderr)
    except Exception as e:
        print(f"⚠️ Could not load violence model: {e}", file=sys.stderr)
    
    try:
        if os.path.exists(args.nsfw_model):
            nsfw_model = keras.models.load_model(args.nsfw_model)
            print(f"✅ Loaded NSFW model from {args.nsfw_model}", file=sys.stderr)
    except Exception as e:
        print(f"⚠️ Could not load NSFW model: {e}", file=sys.stderr)
    
    # Initialize classifier
    classifier = LLMClassificationModel(
        violence_model=violence_model,
        nsfw_model=nsfw_model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_openai_api=args.use_llm,
        openai_api_key=args.api_key
    )
    
    # Run prediction
    result = classifier.predict(
        video_path=args.video_path,
        use_cnn=True,
        use_text=args.use_text,
        use_vision=args.use_vision
    )
    
    # Output JSON result
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()