import gradio as gr
import json
import time
import torch
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


MODEL_PATH = 'berkeruveyik/food-nutrition-analyzer-gemma3-270m'

# Load model and tokenizer
loaded_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype='auto',
    device_map='auto',
    attn_implementation='eager'
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

loaded_model_pipeline = pipeline(
    'text-generation',
    model=loaded_model,
    tokenizer=tokenizer
)

def pred_on_text(input_text):
    """Generate prediction from input text"""
    start_time = time.time()

    raw_output = loaded_model_pipeline(
        text_inputs=[{'role': 'user', 'content': input_text}], 
        max_new_tokens=256
    )

    end_time = time.time()
    total_time = round(end_time - start_time, 4)

    generated_text = raw_output[0]['generated_text'][1]['content']

    return generated_text, raw_output, total_time

def parse_generated_text(text):
    """Parse the generated text and format it nicely"""
    try:
        data = json.loads(text)
        return data
    except:
        try:
            text = text.strip()
            if text.startswith('{') and text.endswith('}'):
                data = eval(text)
                return data
        except:
            pass
    return {"raw_output": text}

def gradio_predict(input_text):
    """Wrapper function for Gradio"""
    if not input_text.strip():
        return "Please enter some text.", "0 seconds"

    generated_text, raw_output, total_time = pred_on_text(input_text)
    time_info = f"{total_time} seconds"

    parsed_output = parse_generated_text(generated_text)

    output_json = json.dumps({
        "input": input_text,
        "model_response": parsed_output,
        "processing_time": total_time
    }, indent=2, ensure_ascii=False)

    return output_json, time_info

# Gradio interface
demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(
        label="Input Text",
        placeholder="Enter your text here...",
        lines=3
    ),
    outputs=[
        gr.Code(label="Model Output (JSON)", language="json"),
        gr.Textbox(label="Processing Time")
    ],
    title="🍔 Food & Nutrition Analyzer",
    description="Enter text describing food or drinks to extract structured nutrition information using a fine-tuned Gemma3 model.",
        examples=[
        ["Today I ate meatballs and potatoes at home with cola, it was delicious"],
        ["British Breakfast with baked beans, fried eggs, black pudding, sausages, bacon, mushrooms, a cup of tea and toast"],
        ["I had a chicken salad with olive oil dressing and sparkling water"],
        ["For lunch I ordered pizza margherita with extra cheese and a glass of lemonade"],
        ["Grilled salmon with steamed vegetables and white wine for dinner"],
        ["My morning started with oatmeal, fresh berries, honey and a cup of black coffee"],
        ["We shared nachos with guacamole, sour cream, and margaritas at the Mexican restaurant"],
        ["Japanese ramen with pork belly, soft boiled egg, nori and green tea"],
        ["Homemade pasta carbonara with parmesan cheese and a bottle of red wine"],
        ["Smoothie bowl with banana, mango, chia seeds, granola and almond milk"],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
