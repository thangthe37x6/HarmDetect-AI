import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import argparse
import json
from video_safety_analyzer import (
    LLMClassificationModel,
    ClassifierModel,
    EfficientNetBackbone,
    CNNClassificationHead,
    LLMClassificationHead,
    VideoProcessor
)
import torch
import keras

def main():

    """Main function được gọi từ Node.js"""
    parser = argparse.ArgumentParser(description='Process video for safety analysis')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--use-llm', action='store_true', 
                       help='Use LLM for enhanced analysis')
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenAI API key')
    parser.add_argument('--violence-model', type=str, 
                       default='best_violence_model.pt',
                       help='Path to violence detection model')
    parser.add_argument('--nsfw-model', type=str,
                       default='best_porn_model.keras',
                       help='Path to NSFW detection model')
    
    args = parser.parse_args()
    
    try:
        # ✅ THÊM LOG CHI TIẾT
        print(f"Starting analysis for: {args.video_path}", file=sys.stderr)
        
        # Validate video path
        if not os.path.exists(args.video_path):
            raise FileNotFoundError(f"Video file not found: {args.video_path}")
        
        print(f"Video file exists: {args.video_path}", file=sys.stderr)
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}", file=sys.stderr)
        
        # Load models
        violence_model = None
        nsfw_model = None
        
        # Load violence model
        if os.path.exists(args.violence_model):
            try:
                violence_model = torch.load(
                    args.violence_model, 
                    map_location=device,
                    weights_only=False
                )
                print(f"Loaded violence model", file=sys.stderr)
            except Exception as e:
                print(f"Could not load violence model: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)  # ✅ In chi tiết lỗi
        else:
            print(f"Violence model not found: {args.violence_model}", file=sys.stderr)
        
        # Load NSFW model
        if os.path.exists(args.nsfw_model):
            try:
                nsfw_model = keras.models.load_model(args.nsfw_model, compile=False)
                print(f"Loaded NSFW model", file=sys.stderr)
            except Exception as e:
                print(f"Could not load NSFW model: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)  # ✅ In chi tiết lỗi
        else:
            print(f"NSFW model not found: {args.nsfw_model}", file=sys.stderr)
        
        # Check if at least one model is loaded
        if violence_model is None and nsfw_model is None:
            raise ValueError("No models loaded. Cannot perform analysis.")
        
        print("Initializing classifier...", file=sys.stderr)
        
        # Initialize classifier
        classifier = LLMClassificationModel(
            violence_model=violence_model,
            nsfw_model=nsfw_model,
            device=device,
            use_openai_api=args.use_llm,
            openai_api_key=args.api_key,
            # ✅ THÊM classifier_config
            classifier_config={
                'use_openai_api': args.use_llm,
                'openai_api_key': args.api_key,
                'model_name': 'gpt-4o-mini'
            } if args.use_llm else None
        )
        
        print("Classifier initialized successfully", file=sys.stderr)
        
        # Run prediction
        print(f"Starting prediction...", file=sys.stderr)
        
        result = classifier.predict(
            video_path=args.video_path,
            use_cnn=True,
            use_text=args.use_llm,
            use_vision=args.use_llm
        )
        
        print("Prediction completed", file=sys.stderr)
        
        # Add success flag
        result['success'] = True
        
        # Output JSON result to stdout
        print(json.dumps(result, ensure_ascii=True))
        
        # Exit with success code
        sys.exit(0)
        
    except Exception as e:
        # ✅ IN CHI TIẾT LỖI
        print(f"ERROR in main: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        # Output error as JSON
        error_result = {
            'success': False,
            'error': str(e),
            'video_path': args.video_path
        }
        print(json.dumps(error_result, ensure_ascii=True))
        sys.exit(1)


if __name__ == "__main__":
    main()