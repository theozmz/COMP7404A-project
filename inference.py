import gradio as gr
import torch
from transformers import pipeline
from huggingface_hub import list_models, ModelFilter
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceInference:
    def __init__(self):
        self.current_pipeline = None
        self.current_task = None
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"使用设备: {'GPU' if self.device == 0 else 'CPU'}")
    
    def get_available_tasks(self):
        """获取可用的任务类型"""
        return [
            "文本分类", 
            "文本生成", 
            "问答系统", 
            "情感分析", 
            "命名实体识别",
            "翻译"
        ]
    
    def get_models_by_task(self, task):
        """根据任务获取可用的模型"""
        task_map = {
            "文本分类": "text-classification",
            "文本生成": "text-generation",
            "问答系统": "question-answering",
            "情感分析": "sentiment-analysis",
            "命名实体识别": "token-classification",
            "翻译": "translation"
        }
        
        try:
            hf_task = task_map.get(task, "text-classification")
            models = list_models(filter=ModelFilter(task=hf_task), limit=10)
            return [model.modelId for model in models]
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return ["distilbert-base-uncased-finetuned-sst-2-english"]
    
    def load_model(self, task, model_name):
        """加载指定的模型"""
        task_map = {
            "文本分类": "text-classification",
            "文本生成": "text-generation", 
            "问答系统": "question-answering",
            "情感分析": "sentiment-analysis",
            "命名实体识别": "token-classification",
            "翻译": "translation"
        }
        
        try:
            hf_task = task_map.get(task, "text-classification")
            logger.info(f"正在加载模型: {model_name} 用于任务: {hf_task}")
            
            self.current_pipeline = pipeline(
                hf_task,
                model=model_name,
                device=self.device
            )
            self.current_task = task
            logger.info(f"模型加载成功: {model_name}")
            return f"模型 {model_name} 加载成功！"
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return f"模型加载失败: {str(e)}"
    
    def inference(self, text, context=None):
        """执行推理"""
        if self.current_pipeline is None:
            return "请先加载模型！"
        
        try:
            if self.current_task == "问答系统" and context:
                result = self.current_pipeline(question=text, context=context)
            elif self.current_task == "翻译":
                result = self.current_pipeline(text)
            else:
                result = self.current_pipeline(text)
            
            logger.info(f"推理完成: {result}")
            return self.format_result(result)
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return f"推理失败: {str(e)}"
    
    def format_result(self, result):
        """格式化推理结果"""
        if isinstance(result, list):
            formatted = []
            for item in result:
                if isinstance(item, dict):
                    formatted.append("\n".join([f"{k}: {v}" for k, v in item.items()]))
                else:
                    formatted.append(str(item))
            return "\n\n".join(formatted)
        elif isinstance(result, dict):
            return "\n".join([f"{k}: {v}" for k, v in result.items()])
        else:
            return str(result)

# 创建推理实例
inference_engine = HuggingFaceInference()

# 创建Gradio界面
def update_model_dropdown(task):
    """更新模型下拉框选项"""
    models = inference_engine.get_models_by_task(task)
    return gr.Dropdown(choices=models, value=models[0] if models else "")

def load_model_and_update_status(task, model_name):
    """加载模型并更新状态"""
    status = inference_engine.load_model(task, model_name)
    return status

def perform_inference(text, context):
    """执行推理"""
    return inference_engine.inference(text, context)

# 创建界面
with gr.Blocks(title="Hugging Face 推理框架", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤗 Hugging Face 机器学习推理框架")
    gr.Markdown("选择任务类型和模型，输入文本进行推理")
    
    with gr.Row():
        with gr.Column(scale=1):
            task_dropdown = gr.Dropdown(
                choices=inference_engine.get_available_tasks(),
                label="选择任务类型",
                value="文本分类"
            )
            
            model_dropdown = gr.Dropdown(
                label="选择模型",
                value="Qwen/Qwen3-0.6B"
            )
            
            load_btn = gr.Button("加载模型", variant="primary")
            model_status = gr.Textbox(label="模型状态", interactive=False)
            
            context_input = gr.Textbox(
                label="上下文 (仅问答系统需要)",
                placeholder="对于问答任务，请在这里输入上下文...",
                lines=3
            )
        
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="输入文本",
                placeholder="请输入要推理的文本...",
                lines=5
            )
            
            inference_btn = gr.Button("执行推理", variant="primary")
            
            output = gr.Textbox(
                label="推理结果",
                lines=10,
                interactive=False
            )
    
    # 事件处理
    task_dropdown.change(
        update_model_dropdown,
        inputs=task_dropdown,
        outputs=model_dropdown
    )
    
    load_btn.click(
        load_model_and_update_status,
        inputs=[task_dropdown, model_dropdown],
        outputs=model_status
    )
    
    inference_btn.click(
        perform_inference,
        inputs=[text_input, context_input],
        outputs=output
    )
    
    # 初始化模型下拉框
    demo.load(
        update_model_dropdown,
        inputs=task_dropdown,
        outputs=model_dropdown
    )

if __name__ == "__main__":
    # 启动服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )