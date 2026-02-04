import os
import logging
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import whisper

logger = logging.getLogger(__name__)

# Get model name from environment or use default
model_name = os.getenv('MODEL_NAME', 'base')
language = os.getenv('LANGUAGE', 'pt')

logger.info(f"Loading Whisper model: {model_name}")
model = whisper.load_model(model_name)
logger.info("Model loaded successfully")

class WhisperASRModel(LabelStudioMLBase):
    """Whisper ASR model for Label Studio"""
    
    def __init__(self, **kwargs):
        super(WhisperASRModel, self).__init__(**kwargs)
        
        self.language = language
        self.model = model
        
        # Cache the model info
        self.model_name = model_name
        self.set("model_version", model_name)


    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """
        Predict transcriptions for audio files.
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        predictions = []
        
        for task in tasks:
            try:
                # Get the audio URL from task data
                audio_url = task['data'].get('audio', '')
                
                if not audio_url:
                    logger.warning(f"No audio URL found in task {task.get('id')}")
                    predictions.append({'result': []})
                    continue
                
                # # Get local path (Label Studio provides this)
                # audio_path = self.get_local_path(audio_url)
                audio_path = r"/label-studio/data/media/" + audio_url.strip(r"/data/")
                
                if not audio_path or not os.path.exists(audio_path):
                    logger.error(f"Audio file not found: {audio_path}")
                    predictions.append({'result': []})
                    continue
                
                logger.info(f"Transcribing: {audio_path}")
                
                # Transcribe with Whisper
                result = self.model.transcribe(
                    audio_path,
                    language=self.language,
                    word_timestamps=True
                )

                segments = []
                for segment in result["segments"]:
                    for word in segment.get("words", []):
                        segments.append({
                            'value': {
                                "start": word["start"],
                                "end": word["end"],
                                'text': word["word"],
                            },
                            'from_name': 'transcription',
                            'to_name': 'audio',
                            'type': 'textarea'
                        })
                    
            
                # Format prediction for Label Studio
                prediction = {
                    'result': segments,
                    'score': self._calculate_confidence(result)
                }
                
                predictions.append(prediction)
                logger.info(f"Transcription complete: {result['text'][:50]}...")
                
            except Exception as e:
                logger.error(f"Error processing task: {str(e)}", exc_info=True)
                predictions.append({'result': []})
        
        return ModelResponse(predictions=predictions)
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate average confidence from segments"""
        segments = result.get('segments', [])
        if not segments:
            return 0.9  # Default confidence
        
        # Average the 'no_speech_prob' as inverse confidence
        confidences = [1.0 - seg.get('no_speech_prob', 0.5) for seg in segments]
        return sum(confidences) / len(confidences) if confidences else 0.9
    
    def fit(self, event, data, **kwargs):
        """
        Fine-tune the model (optional - for future implementation)
        This is called when you train the model from Label Studio
        """
        logger.info("Fit method called - fine-tuning not implemented yet")
        return {'model_version': self.model_name}