import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import json
import torch
import keras

from video_safety_analyzer import LLMClassificationModel


def test_video_classification(video_path: str, 
                              violence_model_path: str = "best_violence_model.pt",
                              nsfw_model_path: str = "best_porn_model.keras",
                              use_llm: bool = False,
                              openai_api_key: str = None):
    
    # Load models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    violence_model = None
    nsfw_model = None
    
    try:
        print(f"Loading violence model from {violence_model_path}...")
        violence_model = torch.load(violence_model_path, map_location=device, weights_only=False)
        print("✅ Violence model loaded")
    except Exception as e:
        print(f"⚠️ Could not load violence model: {e}")
    
    try:
        print(f"Loading NSFW model from {nsfw_model_path}...")
        nsfw_model = keras.models.load_model(nsfw_model_path)
        print("✅ NSFW model loaded")
    except Exception as e:
        print(f"⚠️ Could not load NSFW model: {e}")
    
    # Initialize classifier
    print("\nInitializing LLMClassificationModel...")
    classifier = LLMClassificationModel(
    violence_model=violence_model,
    nsfw_model=nsfw_model,
    device=device,
    use_openai_api=use_llm,
    openai_api_key=openai_api_key,
    classifier_config={
        'use_openai_api': use_llm,
        'openai_api_key': openai_api_key,
        'model_name': 'gpt-4o-mini'
    } if use_llm else None  # Thêm dòng này
)
    
    # Run prediction
    print("\n" + "="*80)
    print("STARTING VIDEO ANALYSIS")
    print("="*80)
    
    result = classifier.predict(
        video_path=video_path,
        use_cnn=True,
        use_text=use_llm,  # Extract text nếu có LLM
        use_vision=use_llm  # Extract vision nếu có LLM
    )
    
    # Print results
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    # Check if analysis was successful (có classification result)
    if result.get('classification'):
        print("\n✅ Analysis completed successfully!")
        
        # Classification
        classification = result['classification']
        print(f"\n📊 Classification:")
        print(f"  Category: {classification['category']}")
        print(f"  Is Harmful: {classification['is_harmful']}")
        print(f"  Confidence: {classification['confidence']:.1%}")
        print(f"  Source: {classification['source']}")
        
        if 'explanation' in classification:
            print(f"  Explanation: {classification['explanation']}")
        
        # CNN Details
        if result['details']['cnn_result']:
            cnn = result['details']['cnn_result']
            print(f"\n🔍 CNN Analysis:")
            
            if 'violence' in cnn:
                v = cnn['violence']
                print(f"  Violence: {v['label']} ({v['confidence']:.1%})")
                print(f"    - Violent segments: {v['violent_segments']}/{v['total_segments']}")
                print(f"    - Violent ratio: {v['violent_ratio']:.1%}")
            
            if 'nsfw' in cnn:
                n = cnn['nsfw']
                print(f"  NSFW: {n['label']} ({n['confidence']:.1%})")
                print(f"    - Porn: {n['porn_segments']}, Hentai: {n['hentai_segments']}, Sexy: {n['sexy_segments']}")
                print(f"    - NSFW ratio: {n['nsfw_ratio']:.1%}")
        
        # LLM Details
        if result['details']['llm_result']:
            llm = result['details']['llm_result']
            print(f"\n🤖 LLM Analysis:")
            print(f"  Category: {llm['category']}")
            print(f"  Confidence: {llm.get('confidence', 'N/A')}")
            print(f"  Is Harmful: {llm['is_harmful']}")
            if 'explanation' in llm:
                print(f"  Explanation: {llm['explanation']}")
        
        # Transcription
        if 'transcription' in result:
            print(f"\n💬 Transcription:")
            print(f"  {result['transcription'][:200]}..." if len(result['transcription']) > 200 else f"  {result['transcription']}")
        
        # Vision Analysis
        if 'vision_analysis' in result:
            print(f"\n👁️ Vision Analysis:")
            print(f"  {result['vision_analysis']}")
        
        # Statistics
        stats = result['processing_info']
        print(f"\n📈 Statistics:")
        print(f"  Segments analyzed: {stats['num_segments']}")
        print(f"  Used CNN: {stats['used_cnn']}")
        print(f"  Used Text: {stats['used_text']}")
        print(f"  Used Vision: {stats['used_vision']}")
        
    else:
        print(f"\n❌ Analysis failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Save to JSON file
    output_file = 'analysis_result.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return result


def format_for_database(result: dict) -> dict:
    """
    Format kết quả để lưu vào database theo schema của videomodel.js
    
    Returns:
        Dict với các field phù hợp với database schema
    """
    # Check if classification exists
    if not result.get('classification'):
        return {
            'status': 'failed',
            'progress': 0,
            'errorMessage': result.get('error', 'Unknown error'),
            'isHarmful': None,
            'category': None,
            'confidence': None,
            'analysisResult': result,
            'summary': None
        }
    
    classification = result['classification']
    
    # Generate summary text
    category = classification['category']
    is_harmful = classification['is_harmful']
    confidence = classification['confidence']
    
    if is_harmful:
        summary = f"⚠️ Harmful content detected: {category} (confidence: {confidence:.1%})"
    else:
        summary = f"✅ Safe content (confidence: {confidence:.1%})"
    
    if 'explanation' in classification:
        summary += f" | {classification['explanation']}"
    
    return {
        'status': 'completed',
        'progress': 100,
        'isHarmful': classification['is_harmful'],
        'category': classification['category'],
        'confidence': classification['confidence'],
        'analysisResult': result,  # Toàn bộ kết quả JSON
        'summary': summary,
        'errorMessage': None
    }


