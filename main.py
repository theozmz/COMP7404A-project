import gradio as gr
import os
import json
import pickle
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import tempfile
import shutil
import openai


API_KEY = 'I WONT TELL YOU'

# Model mapping
MODEL_ZOO = {
    'gemma-2b': 'google/gemma-1.1-2b-it',
    'qwen3-0.6b': 'Qwen/Qwen3-0.6B',
    # 'ouro-1.4b': 'ByteDance/Ouro-1.4B'
}

# Available tasks
AVAILABLE_TASKS = ["nonparaphrased_Arxiv", "paraphrased_Code", "paraphrased_Arxiv", "paraphrased_Yelp"]

# Classifier file mapping
CLASSIFIER_MAP = {
    "nonparaphrased_Arxiv": "non_Arxiv.pkl",
    "paraphrased_Arxiv": "para_Arxiv.pkl", 
    "paraphrased_Code": "para_Code.pkl",
    "paraphrased_Yelp": "para_Yelp.pkl"
}

class BiScopeInference:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_models = {}
        self.classifiers = {}
        self.classifier_dir = "./Classifiers"
        
        # Load all classifiers
        self.load_all_classifiers()
        
    def load_all_classifiers(self):
        """Load all available classifiers"""
        for task, filename in CLASSIFIER_MAP.items():
            model_path = os.path.join(self.classifier_dir, filename)
            if os.path.exists(model_path):
                try:
                    print(f"Loading classifier for {task} from {model_path}")
                    with open(model_path, 'rb') as f:
                        classifier = pickle.load(f)
                    
                    # Validate classifier object
                    if hasattr(classifier, 'predict_proba') and hasattr(classifier, 'predict'):
                        self.classifiers[task] = classifier
                        print(f"Successfully loaded classifier for {task}")
                    else:
                        print(f"Invalid classifier for {task}: missing required methods")
                except Exception as e:
                    print(f"Error loading classifier for {task}: {e}")
            else:
                print(f"Classifier file not found: {model_path}")
        
        if not self.classifiers:
            print("Warning: No classifiers were loaded successfully")
    
    def get_available_models(self):
        """Get available model list"""
        return list(MODEL_ZOO.keys())
    
    def get_available_tasks(self):
        """Get available task list"""
        return AVAILABLE_TASKS
    
    def load_model(self, model_key):
        """Load specified model"""
        if model_key not in MODEL_ZOO:
            raise ValueError(f"Unknown model: {model_key}")
        
        if model_key in self.current_models:
            return self.current_models[model_key]
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ZOO[model_key],
                torch_dtype=torch.float16,
                device_map='auto'
            ).eval()
            
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ZOO[model_key], 
                padding_side='left'
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            self.current_models[model_key] = (model, tokenizer)
            return model, tokenizer
            
        except Exception as e:
            raise Exception(f"Model loading failed: {str(e)}")
    
    def compute_fce_loss(self, logits, targets, text_slice):
        """Compute FCE loss"""
        loss = CrossEntropyLoss(reduction='none')(
            logits[0, text_slice.start-1:text_slice.stop-1, :],
            targets
        )
        return loss.detach().cpu().numpy()
    
    def compute_bce_loss(self, logits, targets, text_slice):
        """Compute BCE loss"""
        loss = CrossEntropyLoss(reduction='none')(
            logits[0, text_slice, :],
            targets
        )
        return loss.detach().cpu().numpy()
    
    def generate_summary(self, text, summary_model_key):
        """Generate text summary using local model or API"""
        if summary_model_key == "none":
            return None
        
        # API-based summary generation
        if summary_model_key.startswith("deepseek"):
            try:
                # Get API key from environment
                api_key = os.environ.get("OPENAI_API_KEY", API_KEY)
                if not api_key:
                    return None
                
                client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                
                # Generate summary using OpenAI API
                response = client.chat.completions.create(
                    model=summary_model_key,
                    messages=[
                        {"role": "system", "content": "generate a very short and concise summary for the following text, just the summary:"},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=100,
                    temperature=0.3
                )
                
                summary_text = response.choices[0].message.content.strip()
                return summary_text
                
            except Exception as e:
                print(f"API summary generation failed: {e}")
                return None
        
        # Local model summary generation
        elif summary_model_key in MODEL_ZOO:
            try:
                summary_model, summary_tokenizer = self.load_model(summary_model_key)
                
                summary_input = f"Write a title for this text: {text}\nJust output the title:"
                summary_ids = summary_tokenizer(summary_input, return_tensors='pt',
                                              max_length=2000, truncation=True).input_ids.to(self.device)
                summary_ids = summary_ids[:, 1:]  # Remove start token
                
                # Simplified generation
                config = summary_model.generation_config
                config.max_new_tokens = 64
                gen_ids = summary_model.generate(
                    summary_ids, 
                    generation_config=config,
                    pad_token_id=summary_tokenizer.pad_token_id
                )[0]
                gen_ids = gen_ids[summary_ids.shape[1]:]  # Take only generated part
                
                summary_text = summary_tokenizer.decode(gen_ids, skip_special_tokens=True).strip().split('\n')[0]
                return summary_text
                
            except Exception as e:
                print(f"Local model summary generation failed: {e}")
                return None
        else:
            return None
    
    def extract_features(self, text, detect_model_key, summary_model_key, sample_clip=2000):
        """Extract text features"""
        try:
            # Load detection model
            detect_model, detect_tokenizer = self.load_model(detect_model_key)
            
            # Generate summary
            summary_text = self.generate_summary(text, summary_model_key)
            
            # Build prompt
            if summary_text:
                prompt_text = f"Given the summary:\n{summary_text}\nComplete the following text: "
            else:
                prompt_text = "Complete the following text: "
            
            # Encode text
            prompt_ids = detect_tokenizer(prompt_text, return_tensors='pt').input_ids.to(self.device)
            text_ids = detect_tokenizer(text, return_tensors='pt', max_length=sample_clip, truncation=True).input_ids.to(self.device)
            
            # Combine inputs
            combined_ids = torch.cat([prompt_ids, text_ids], dim=1)
            text_slice = slice(prompt_ids.shape[1], combined_ids.shape[1])
            
            # Forward pass
            with torch.no_grad():
                outputs = detect_model(input_ids=combined_ids)
                logits = outputs.logits
                targets = combined_ids[0][text_slice]
            
            # Compute loss features
            fce_loss = self.compute_fce_loss(logits, targets, text_slice)
            bce_loss = self.compute_bce_loss(logits, targets, text_slice)
            
            # Extract features
            features = []
            for p in range(1, 10):
                split = len(fce_loss) * p // 10
                features.extend([
                    np.mean(fce_loss[split:]), np.max(fce_loss[split:]), 
                    np.min(fce_loss[split:]), np.std(fce_loss[split:]),
                    np.mean(bce_loss[split:]), np.max(bce_loss[split:]), 
                    np.min(bce_loss[split:]), np.std(bce_loss[split:])
                ])
            
            return features
            
        except Exception as e:
            raise Exception(f"Feature extraction failed: {str(e)}")
    
    def predict_ai_probability(self, features, task):
        """Predict AI probability using classifier for the specified task"""
        if task not in self.classifiers:
            raise Exception(f"No classifier available for task: {task}")
        
        feature_array = np.array(features).reshape(1, -1)
        
        # Get probability from classifier
        proba = self.classifiers[task].predict_proba(feature_array)[0]
        
        # Assuming class 0 is human, class 1 is AI
        # Adjust based on your classifier's class ordering
        if len(proba) == 2:
            human_prob = proba[0]  # Probability of being human
            ai_prob = proba[1]     # Probability of being AI
        else:
            # If only one probability is returned, assume it's AI probability
            ai_prob = proba[0] if len(proba) == 1 else 0.5
            human_prob = 1 - ai_prob
            
        return human_prob, ai_prob

# Create inference instance
inference_engine = BiScopeInference()

def analyze_text(text, task, summary_model, detect_model, sample_clip):
    """Main function for text analysis"""
    if not text.strip():
        return "Please input text to analyze"
    
    try:
        # Extract features
        features = inference_engine.extract_features(text, detect_model, summary_model, sample_clip)
        
        # Predict AI probability
        human_prob, ai_prob = inference_engine.predict_ai_probability(features, task)
        
        # Format result
        result = f"""
## Analysis Result

**🤖 AI Generation Probability**: {ai_prob:.2%}
**👨 Human Writing Probability**: {human_prob:.2%}

### Judgment:
{"🤖 Likely AI Generated" if ai_prob > 0.6 else "👨 Likely Human Written" if human_prob > 0.6 else "❓ Uncertain"}

### Models Used:
- **Classifier**: {task}
- **Summary Model**: {summary_model}
- **Detection Model**: {detect_model}
- **Text Length Limit**: {sample_clip} tokens

### Feature Statistics:
- Extracted {len(features)} dimensional features
- Average FCE Loss: {np.mean([features[i] for i in range(0, len(features), 8)]):.4f}
- FCE Loss Standard Deviation: {np.mean([features[i+3] for i in range(0, len(features), 8)]):.4f}
        """
        
        return result
        
    except Exception as e:
        return f"Error during analysis: {str(e)}"

def batch_analyze(file, task, summary_model, detect_model, sample_clip):
    """Batch analyze files"""
    if file is None:
        return "Please upload a file"
    
    try:
        # Read file content
        if file.name.endswith('.txt'):
            with open(file.name, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        elif file.name.endswith('.json'):
            with open(file.name, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    texts = [item.get('text', str(item)) for item in data]
                else:
                    texts = [str(data)]
        else:
            return "Unsupported file format. Please upload .txt or .json files"
        
        if not texts:
            return "No valid text content found in file"
        
        results = []
        progress = gr.Progress()
        
        for i, text in enumerate(progress.tqdm(texts, desc="Analysis Progress")):
            if len(text) > sample_clip * 4:  # Rough token count estimation
                text = text[:sample_clip * 4]
            
            try:
                features = inference_engine.extract_features(
                    text, detect_model, summary_model, sample_clip
                )
                human_prob, ai_prob = inference_engine.predict_ai_probability(features, task)
                
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "ai_probability": ai_prob,
                    "human_probability": human_prob,
                    "judgment": "AI Generated" if ai_prob > 0.6 else "Human Written" if ai_prob < 0.4 else "Uncertain"
                })
                
            except Exception as e:
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "ai_probability": -1,
                    "human_probability": -1,
                    "judgment": f"Analysis failed: {str(e)}"
                })
        
        # Format batch results
        output = "## Batch Analysis Results\n\n"
        for i, result in enumerate(results, 1):
            output += f"### Text {i}\n"
            output += f"**Content Preview**: {result['text']}\n"
            if result['ai_probability'] >= 0:
                output += f"**AI Generation Probability**: {result['ai_probability']:.2%}\n"
                output += f"**Human Writing Probability**: {result['human_probability']:.2%}\n"
                output += f"**Judgment**: {result['judgment']}\n"
            else:
                output += f"**Status**: {result['judgment']}\n"
            output += "\n"
        
        # Add statistics
        valid_results = [r for r in results if r['ai_probability'] >= 0]
        if valid_results:
            ai_count = sum(1 for r in valid_results if r['judgment'] == 'AI Generated')
            human_count = sum(1 for r in valid_results if r['judgment'] == 'Human Written')
            uncertain_count = len(valid_results) - ai_count - human_count
            avg_ai_prob = np.mean([r['ai_probability'] for r in valid_results])
            
            output += "### Statistics Summary\n"
            output += f"- Total Texts: {len(results)}\n"
            output += f"- Successfully Analyzed: {len(valid_results)}\n"
            output += f"- AI Generated Texts: {ai_count} ({ai_count/len(valid_results):.1%})\n"
            output += f"- Human Written Texts: {human_count} ({human_count/len(valid_results):.1%})\n"
            output += f"- Uncertain Texts: {uncertain_count} ({uncertain_count/len(valid_results):.1%})\n"
            output += f"- Average AI Probability: {avg_ai_prob:.2%}\n"
        
        return output
        
    except Exception as e:
        return f"File processing error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="COMP7494A-group-R", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 BISCOPE: AI-generated Text Detection Demo")
    gr.Markdown("BISCOPE: AI-generated Text Detection by Checking Memorization of Preceding Tokens")
    
    with gr.Tab("Single Text Analysis"):
        with gr.Row():
            with gr.Column():
                task_dropdown = gr.Dropdown(
                    choices=inference_engine.get_available_tasks(),
                    label="Classifier",
                    value="nonparaphrased_Arxiv",
                    info="Choose the classifier based on the task."
                )
                
                summary_dropdown = gr.Dropdown(
                    choices=["none", "deepseek-chat"] + inference_engine.get_available_models(),
                    label="Summary Model",
                    value="none",
                    info="The model used to generate text summary. Choose 'none' to skip summary generation."
                )
                
                detect_dropdown = gr.Dropdown(
                    choices=inference_engine.get_available_models(),
                    label="Detection Model",
                    value="qwen3-0.6b",
                    info="The model used to extract features."
                )
                
                clip_slider = gr.Slider(
                    minimum=500,
                    maximum=2000,
                    value=1000,
                    step=100,
                    label="Max Tokens",
                    info="The maximum token length for input text."
                )
            
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text to Analyze",
                    placeholder="Input your text here...",
                    lines=10,
                    max_lines=20
                )
                
                analyze_btn = gr.Button("START ANALYSIS", variant="primary")
                
                output = gr.Markdown(
                    label="ANALYSIS RESULT"
                )
    
    with gr.Tab("Batch Analysis"):
        with gr.Row():
            with gr.Column(scale=1):
                batch_task = gr.Dropdown(
                    choices=inference_engine.get_available_tasks(),
                    label="Classifier",
                    value="nonparaphrased_Arxiv"
                )
                
                batch_summary = gr.Dropdown(
                    choices=["none", "deepseek-chat"] + inference_engine.get_available_models(),
                    label="Summary Model",
                    value="none"
                )
                
                batch_detect = gr.Dropdown(
                    choices=inference_engine.get_available_models(),
                    label="Detection Model",
                    value="qwen3-0.6b"
                )
                
                batch_clip = gr.Slider(
                    minimum=500,
                    maximum=2000,
                    value=1000,
                    step=100,
                    label="Max Tokens"
                )
                
                file_input = gr.File(
                    label="Upload File",
                    file_types=[".txt", ".json"],
                    # info="Supports .txt (one text per line) or .json files"
                )
            
            with gr.Column(scale=2):
                batch_analyze_btn = gr.Button("START BATCH ANALYSIS", variant="primary")
                batch_output = gr.Markdown(label="Batch Analysis Results")
    
    with gr.Tab("Instructions"):
        gr.Markdown("""
        ## BISCOPE AI Text Detection Framework - Instructions
        
        ### Overview
        This system uses the BISCOPE algorithm to detect AI-generated text by analyzing linguistic features.
        
        ### Parameter Explanation
        - **Classifier**: Choose the classifier based on your text type (nonparaphrased_Arxiv, paraphrased_Code, etc.)
        - **Summary Model**: Model used to generate text summary. Choose 'none' to skip summary generation.
        - **Detection Model**: Model used for feature extraction and detection.
        - **Max Tokens**: Limit text length for efficient processing.
        
        ### Available Classifiers
        - **nonparaphrased_Arxiv**: For non-paraphrased academic papers
        - **paraphrased_Arxiv**: For paraphrased academic papers
        - **paraphrased_Code**: For paraphrased code
        - **paraphrased_Yelp**: For paraphrased Yelp reviews
        
        ### Summary Model Options
        - **none**: No summary generation
        - **deepseek-chat**: Use DeepSeek-Chat for summary generation (requires API key)
        - Local models (gemma-2b, qwen3-0.6b): Use local models for summary generation
        
        ### Usage Tips
        1. For academic papers, choose Arxiv classifiers
        2. For code, choose Code classifier
        3. For reviews, choose Yelp classifier
        4. Use summary generation for better accuracy on longer texts
        
        ### Technical Principle
        Based on the BISCOPE algorithm, the system computes FCE (Forward Cross Entropy) and BCE (Backward Cross Entropy) loss features,
        then uses machine learning classifiers to distinguish between human-written and AI-generated text.
        
        ### API Setup
        To use OpenAI models for summary generation:
        1. Set your OpenAI API key as environment variable: OPENAI_API_KEY
        2. Choose 'deepseek-chat' as Summary Model
        """)

    with gr.Tab("Paper"):
        gr.Markdown("""
        ## BISCOPE: AI-generated Text Detection by Checking Memorization of Preceding Tokens
        
        ### Abstract
        Detecting text generated by Large Language Models (LLMs) is a pressing need in
        order to identify and prevent misuse of these powerful models in a wide range of
        applications, which have highly undesirable consequences such as misinformation
        and academic dishonesty. Given a piece of subject text, many existing detection
        methods work by measuring the difficulty of LLM predicting the next token in
        the text from their prefix. In this paper, we make a critical observation that
        how well the current token’s output logits memorizes the closely preceding input
        tokens also provides strong evidence. Therefore, we propose a novel bi-directional
        calculation method that measures the cross-entropy losses between an output
        logits and the ground-truth token (forward) and between the output logits and
        the immediately preceding input token (backward). A classifier is trained to
        make the final prediction based on the statistics of these losses. We evaluate our
        system, named BISCOPE, on texts generated by five latest commercial LLMs
        across five heterogeneous datasets, including both natural language and code.
        BISCOPE demonstrates superior detection accuracy and robustness compared to
        nine existing baseline methods, exceeding the state-of-the-art non-commercial
        methods’ detection accuracy by over 0.30 F1 score, achieving over 0.95 detection
        F1 score on average. It also outperforms the best commercial tool GPTZero that is
        based on a commercial LLM trained with an enormous volume of data. Code is
        available at https://github.com/MarkGHX/BiScope.
        """)

    with gr.Tab("About"):
        gr.Markdown("""
        ## Group-R
        
        ### Members
        * Qirui Chen u3665599@connect.hku.hk
        * An Wang u3666321@connect.hku.hk
        * Mingze Zhao u3665791@connect.hku.hk
        """)
    
    # Event binding
    analyze_btn.click(
        analyze_text,
        inputs=[text_input, task_dropdown, summary_dropdown, detect_dropdown, clip_slider],
        outputs=output
    )
    
    batch_analyze_btn.click(
        batch_analyze,
        inputs=[file_input, batch_task, batch_summary, batch_detect, batch_clip],
        outputs=batch_output
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=1145,
        share=True
    )