"""
AITW (Android in the Wild) Benchmark Evaluation Script for ST-Lite
"""

import ast
import json
import re
import argparse
import os
import logging
import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from multiprocessing import freeze_support

# Suppress JAX TPU/CUDA warnings for action_matching if JAX is installed
os.environ['JAX_PLATFORMS'] = 'cpu'

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel, AutoImageProcessor
from qwen_vl_utils import process_vision_info
from opencua_utils import opencua_parse_action, analyze_vision_tokens_opencua_multi_images
from ui_tars_utils import (
    parse_action_to_structure_output, MIN_PIXELS, MAX_PIXELS, IMAGE_FACTOR,
    analyze_vision_tokens_multi_images
)

# Attention mechanism helpers
from attention_helpers import (
    replace_qwen2_5_vl,
    replace_opencua,
    set_attention_implementation,
    configure_accelerate_skip_attention,
    set_kv_cache_budget,
    set_move_attention_to_cpu,
    set_last_vision_indices,
    set_vision_start_idx,
    set_vision_end_idx,
    set_temperature,
    set_alpha,
    set_window_size,
)

import action_matching

# Constants
AITW_IMGS_DIR = "/path/to/your/aitw_images"
AITW_TEST_PATH = "/path/to/your/aitw_data_test.json"

MOBILE_USE_TEMPLATE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space
click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
open_app(app_name='')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_home()
press_back()
finished(content='xxx')

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)

def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def action2step(step_data):
    """Convert AITW step data to UI-TARS compatible action string."""
    action_type = step_data["action_type_id"]
    
    if action_type == 4: # Click/Scroll
        if step_data["action_type_text"] == 'click':
            touch, lift = step_data["touch"], step_data["lift"]
            cx, cy = int(1000 * (touch[0] + lift[0]) / 2), int(1000 * (touch[1] + lift[1]) / 2)
            return f'{{"action_type": 4, "click_point": "({cx},{cy})"}}'
        else:
            scroll_map = {'scroll down': 0, 'scroll up': 1, 'scroll left': 8, 'scroll right': 9}
            atype = scroll_map.get(step_data["action_type_text"], action_type)
            return f'{{"action_type": {atype}}}'
    elif action_type == 3: # Type
        return f'{{"action_type": 3, "typed_text": "{step_data["type_text"]}"}}'
    
    return f'{{"action_type": {action_type}}}'

def clean_model_output(text):
    """Clean model output to extract JSON/Action content."""
    text = text.strip()
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text).replace('```', '').strip()
    # Remove chat templates and tags
    text = re.sub(r'(Image_\d+:|<tool_call>|assistant|user)\s*', '', text, flags=re.IGNORECASE)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Extract content within braces if present
    start = text.find('{')
    if start != -1:
        brace_count = 0
        in_string = False
        escape = False
        for i, char in enumerate(text[start:], start):
            if escape: escape = False; continue
            if char == '\\': escape = True; continue
            if char == '"' and not escape: in_string = not in_string
            if not in_string:
                if char == '{': brace_count += 1
                elif char == '}': 
                    brace_count -= 1
                    if brace_count == 0:
                        text = text[start:i+1]
                        break
    
    # Strip surrounding quotes
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    
    return re.sub(r'[\.\,\s]+$', '', text)

def parse_action_output(response):
    """Parse model output into a structured dictionary."""
    cleaned = clean_model_output(response)
    
    # Try direct evaluation
    for parser in [ast.literal_eval, json.loads]:
        try: return parser(cleaned)
        except: pass
    
    # Regex fallback for specific action formats
    try:
        # Click/Long Press
        match = re.search(r"(?:click|left_single|long_press)\(.*?\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\).*?\)", cleaned)
        if match: return {"action_type": 4, "click_point": (float(match.group(1)), float(match.group(2)))}
        
        # Type
        match = re.search(r"type\(content=['\"](.*?)['\"]\)", cleaned)
        if match: return {"action_type": 3, "typed_text": match.group(1).replace('\\n', '\n')}
        
        # Special keys
        if "press_home" in cleaned: return {"action_type": 6}
        if "press_back" in cleaned: return {"action_type": 5}
        if "press_enter" in cleaned: return {"action_type": 7}
        if "finished" in cleaned: return {"action_type": 10}
        
        # Scroll
        match = re.search(r"scroll\(.*?direction=['\"](.*?)['\"].*?\)", cleaned)
        if match:
            d = match.group(1).lower()
            dirs = {'down': 0, 'up': 1, 'left': 8, 'right': 9}
            for k, v in dirs.items():
                if k in d: return {"action_type": v}
        
        # Drag
        match = re.search(r"drag\(.*?\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\).*?\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\).*?\)", cleaned)
        if match:
            x1, y1, x2, y2 = map(float, match.groups())
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) > abs(dy): return {"action_type": 9} if dx > 0 else {"action_type": 8}
            else: return {"action_type": 0} if dy > 0 else {"action_type": 1}

        # Open App
        match = re.search(r"open_app\(app_name=['\"](.*?)['\"]\)", cleaned)
        if match: return {"action_type": 3, "typed_text": match.group(1)}

    except Exception:
        pass
    return None

def process_string(s):
    """Normalize 1000-scale coordinates to 0-1 float scale."""
    return re.sub(r'\((\d+),(\d+)\)', lambda m: f"({float(m.group(1))/1000:.2f},{float(m.group(2))/1000:.2f})", s)

if __name__ == '__main__':
    freeze_support()
    
    parser = argparse.ArgumentParser(description='AITW Benchmark Evaluation for GUI-KV')
    parser.add_argument('--model_path', type=str, default='/path/to/your/UI-TARS-1.5-7B')
    parser.add_argument('--aitw_imgs', type=str, default=AITW_IMGS_DIR)
    parser.add_argument('--aitw_test', type=str, default=AITW_TEST_PATH)
    parser.add_argument('--his_num', type=int, default=16, help='History length')
    parser.add_argument('--debug', default=None, type=int, help='Debug sample limit')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--model_dtype', type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument('--attention_implementation', type=str, default="flash_attention_2")
    parser.add_argument('--kv_cache', type=str, default="st_lite", 
                        choices=["full_cache", "pyramid_kv", "vl_cache", "snap_kv", "st_lite"])
    parser.add_argument('--kv_cache_budget', type=int, nargs='+', default=[40, 80])
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--results_dir', type=str, default='./results/aitw/')
    args = parser.parse_args()

    # Device Setup
    device = args.device if args.device else get_device()
    print(f"Device: {device}")
    
    # Model Loading
    model_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.model_dtype]
    is_opencua = "OpenCUA" in args.model_path
    
    if is_opencua:
        processor = AutoImageProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        replace_opencua(kv_cache_mode=args.kv_cache)
    else:
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=256*28*28, max_pixels=1280*28*28)
        replace_qwen2_5_vl(kv_cache_mode=args.kv_cache)
    
    print(f"Attention patched with mode: {args.kv_cache}")

    load_kwargs = {
        "pretrained_model_name_or_path": args.model_path,
        "torch_dtype": model_dtype,
        "attn_implementation": args.attention_implementation,
        "trust_remote_code": True
    }
    
    if device == "cpu":
        load_kwargs["device_map"] = "cpu"
    elif torch.cuda.device_count() > 1:
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = {"": "cuda:0"}

    ModelClass = AutoModel if is_opencua else Qwen2_5_VLForConditionalGeneration
    model = ModelClass.from_pretrained(**load_kwargs)
    
    # Post-load configuration
    set_attention_implementation(model, args)
    if args.attention_implementation == "eager":
        set_move_attention_to_cpu(model, args)
        configure_accelerate_skip_attention(model)

    print("Model loaded.")

    # Data Loading
    aitw_test = json.load(open(args.aitw_test, 'r'))
    all_budget_results = {}

    for budget in args.kv_cache_budget:
        print(f"\nEvaluating budget: {budget}")
        args.kv_cache_budget = budget
        set_kv_cache_budget(model, args)
        
        # Clear previous cache states
        for module in model.modules():
            if hasattr(module, "kv_cluster"): del module.kv_cluster
            
        current_results_dir = os.path.join(args.results_dir, f"budget_{budget}")
        os.makedirs(current_results_dir, exist_ok=True)

        all_save_results = []
        metrics = {k: 0 for k in ['corr_action', 'total', 'corr_type', 'num_text', 'corr_text', 
                                  'num_scroll', 'corr_scroll', 'num_click', 'corr_click', 
                                  'num_both', 'corr_both', 'wrong_format']}
        task_metrics = {}

        for task, episodes in aitw_test.items():
            print(f"Task: {task}")
            if args.debug: episodes = episodes[:args.debug]
            
            task_stats = {k: 0 for k in metrics.keys()} # Local stats

            for episode in tqdm(episodes, desc=f"Processing {task}"):
                history_actions, history_imgs = [], []
                
                for step in episode:
                    step_log = {'task': task, 'episode': step['ep_id'], 'correct': 'no', 'budget': budget}
                    
                    img_path = os.path.join(args.aitw_imgs, step["img_filename"] + '.png')
                    if not os.path.exists(img_path): continue
                    
                    # Prompt Construction
                    prompt = MOBILE_USE_TEMPLATE.format(instruction=step["goal"], language="English")
                    cur_imgs = []
                    
                    # History
                    for i, (action, img) in enumerate(zip(history_actions[-args.his_num:], history_imgs[-args.his_num:])):
                        prompt += f'Image_{i}:<image>\nStep_{i}: {action} .\n'
                        cur_imgs.append(img)
                    
                    # Current step
                    prompt += f'Image_{len(cur_imgs)}:<image>\n'
                    cur_imgs.append(img_path)
                    
                    # Update history
                    history_actions.append(action2step(step))
                    history_imgs.append(img_path)
                    action_ref = action_matching.action_2_format(step)

                    # Prepare Model Inputs
                    if is_opencua:
                        sys_prompt = "You are a GUI agent... perform pyautogui actions."
                        messages = [{
                            "role": "user",
                            "content": [{"type": "image", "image": img} for img in cur_imgs] + 
                                      [{"type": "text", "text": sys_prompt + "\n" + prompt}]
                        }]
                        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                        images = [Image.open(img).convert('RGB') for img in cur_imgs]
                        info = processor.preprocess(images=images)
                        pixel_values = torch.tensor(info['pixel_values'], dtype=torch.bfloat16, device=model.device)
                        grid_thws = torch.tensor(info['image_grid_thw'])
                        input_ids = torch.tensor([input_ids], device=model.device)
                    else:
                        messages = [{
                            "role": "user", 
                            "content": [{"type": "image", "image": img} for img in cur_imgs] + [{"type": "text", "text": prompt}]
                        }]
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

                    # Vision Analysis & Parameter Setting
                    if args.kv_cache in ["st_lite", "vl_cache"]:
                        if not is_opencua:
                            vision_analysis = analyze_vision_tokens_multi_images(
                                processor, image_inputs, video_inputs, text, image_count=len(cur_imgs)
                            )
                        else:
                            vision_analysis = analyze_vision_tokens_opencua_multi_images(
                                tokenizer, input_ids, image_grid_thw=info["image_grid_thw"], 
                                merge_size=2, image_count=len(cur_imgs)
                            )
                        
                        set_window_size(model, args)
                        
                        if args.kv_cache == "vl_cache":
                            v_end = vision_analysis.get('vision_end_idx', [0])
                            indices = v_end if isinstance(v_end, list) else [v_end]
                            set_last_vision_indices(model, indices, args)
                        elif args.kv_cache in ["st_lite"]:
                            set_vision_start_idx(model, vision_analysis['vision_start_idx'], args)
                            set_vision_end_idx(model, vision_analysis['vision_end_idx'], args)
                            set_alpha(model, args)
                            set_temperature(model, args)

                    # Generation
                    try:
                        gen_kwargs = {
                            "max_new_tokens": args.max_new_tokens,
                            "use_cache": True,
                            "do_sample": False
                        }
                        
                        if is_opencua:
                            outputs = model.generate(input_ids, pixel_values=pixel_values, grid_thws=grid_thws, return_dict_in_generate=False, **gen_kwargs)
                            generated_ids = outputs[:, input_ids.shape[1]:]
                            output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        else:
                            outputs = model.generate(**inputs, return_dict_in_generate=True, pad_token_id=processor.tokenizer.eos_token_id, **gen_kwargs)
                            trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs.sequences)]
                            output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
                            
                    except Exception as e:
                        logging.error(f"Generation error: {e}")
                        continue

                    # Evaluation
                    task_stats['total'] += 1
                    try:
                        cleaned = process_string(clean_model_output(output_text))
                        parsed = parse_action_output(cleaned) or parse_action_output(output_text)
                        
                        if parsed:
                            pred = action_matching.pred_2_format(parsed)
                            annot_pos = np.array([step["annot_position"][i:i+4] for i in range(0, len(step["annot_position"]), 4)])
                            
                            is_match = action_matching.check_actions_match(
                                pred["touch_point"], pred["lift_point"], pred["action_type"],
                                action_ref["touch_point"], action_ref["lift_point"], action_ref["action_type"], annot_pos
                            )
                            
                            if is_match:
                                task_stats['corr_action'] += 1
                                step_log['correct'] = 'yes'
                            
                            # Granular Metrics
                            if pred["action_type"] == action_ref["action_type"]:
                                task_stats['corr_type'] += 1
                            
                            if action_ref["action_type"] == 3:
                                task_stats['num_text'] += 1
                                if pred.get("typed_text", "") == action_ref.get("typed_text", ""):
                                    task_stats['corr_text'] += 1
                            
                            if action_ref["action_type"] == 4:
                                is_ref_tap = action_matching.is_tap_action(action_ref["touch_point"], action_ref["lift_point"])
                                is_pred_tap = action_matching.is_tap_action(pred["touch_point"], pred["lift_point"])
                                
                                if is_ref_tap:
                                    task_stats['num_click'] += 1
                                    if is_match: task_stats['corr_click'] += 1
                                else:
                                    task_stats['num_scroll'] += 1
                                    if is_match: task_stats['corr_scroll'] += 1
                                
                                if pred["action_type"] == 4 and is_ref_tap and is_pred_tap:
                                    task_stats['num_both'] += 1
                                    if is_match: task_stats['corr_both'] += 1
                        else:
                            raise ValueError("Parse failed")
                            
                    except Exception:
                        task_stats['wrong_format'] += 1
                    
                    all_save_results.append(step_log)

            # Aggregate task stats to global
            for k, v in task_stats.items():
                metrics[k] += v
            
            task_acc = task_stats['corr_action'] / task_stats['total'] if task_stats['total'] else 0
            task_metrics[task] = {'acc': task_acc, 'raw': task_stats}
            print(f"Task {task} Acc: {task_acc:.4f}")

        # Summary for budget
        total = metrics['total']
        avg_score = metrics['corr_action'] / total if total else 0
        
        logging.info(f"Budget {budget} Results:")
        logging.info(f"Overall Acc: {avg_score:.4f}")
        logging.info(f"Format Errors: {metrics['wrong_format']}")
        
        write_json(all_save_results, os.path.join(current_results_dir, 'detailed_results.json'))
        write_json({'overall': metrics, 'tasks': task_metrics}, os.path.join(current_results_dir, 'metrics.json'))
        
        all_budget_results[budget] = metrics

    write_json(all_budget_results, os.path.join(args.results_dir, "multi_budget_summary.json"))
    print("Evaluation completed.")