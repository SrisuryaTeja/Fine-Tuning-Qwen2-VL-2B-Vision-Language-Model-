import os
import io
import wandb
import json
import argparse 

import torch
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import TextStreamer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from evaluate import load
from unsloth import FastVisionModel, is_bf16_supported, FastLanguageModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer


# --- Constants ---
MODEL_NAME = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"
DATASET_ID = "HuggingFaceM4/the_cauldron"
SUBSET = "geomverse"
PROJECT_NAME = "Qwen2-VL-2B-Instruct-finetune"
RUN_NAME = "geomverse-test-run-meteor"
TRAIN_SAMPLE_SIZE = 3000
EVAL_SAMPLE_SIZE = 200


def setup_wandb(project_name: str, run_name: str):
    """Initializes Weights & Biases for experiment tracking."""
    try:
        api_key = os.getenv("WANDB_API_KEY")
        if api_key is None:
            raise EnvironmentError("WANDB_API_KEY is not set. Please set it in your environment.")
        wandb.login(key=api_key)
        print("Successfully logged into WandB")
    except Exception as e:
        print(f"Error logging into WandB: {e}. Ensure WANDB_API_KEY is set.")
        return

    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_SILENT"] = "true"

    try:
        wandb.init(project=project_name, name=run_name)
        print(f"WandB run initialized: Project - {project_name}, Run - {run_name}")
    except Exception as e:
        print(f"Error initializing WandB run: {e}")


def convert_rgb(image: Image.Image) -> Image.Image:
    """Converts any image to RGB, handling RGBA with a white background."""
    if image.mode == "RGB":
        return image
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    image_rgba = Image.alpha_composite(background, image_rgba)
    return image_rgba.convert("RGB")


def reduce_image_size(image: Image.Image, scale: float = 0.5) -> Image.Image:
    """Reduces image size to prevent VRAM OOM errors."""
    return image.resize((512, 512), Image.Resampling.LANCZOS)


def format_data(sample: dict) -> dict:
    """Convert dataset samples into Unsloth-compatible multimodal chat format."""
    if not sample.get("images") or len(sample["images"]) == 0 or not sample.get("texts"):
        return {"messages": None}

    try:
        image = sample["images"][0]
        image = convert_rgb(image)
        image = reduce_image_size(image)
        assert isinstance(image, Image.Image)
    except Exception as e:
        print(f"[format_data] Error processing image: {e}")
        return {"messages": None}

    text_dict = sample["texts"][0] if isinstance(sample["texts"], list) and len(sample["texts"]) > 0 else {}
    user_text = str(text_dict.get("user", "")).strip()
    assistant_text = str(text_dict.get("assistant", "")).strip()

    if not user_text or not assistant_text:
        return {"messages": None}

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": image},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}],
        },
    ]
    return {"messages": messages}


# ## Main Script

processed_train_dataset = []
processed_eval_dataset = []

def main():
    global processed_train_dataset, processed_eval_dataset

    parser = argparse.ArgumentParser(description="Fine-tune and evaluate a vision model.")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the training step and load the model from --output-dir"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Qwen2-VL-2B-Instruct-geomverse-lora",
        help="Directory to save/load model checkpoints"
    )
    args = parser.parse_args()

    print("--- 1. Setting up Environment ---")
    load_dotenv()
    setup_wandb(PROJECT_NAME, RUN_NAME)

    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise EnvironmentError("HF_TOKEN is not set.")
    login(hf_token)
    print("Successfully logged into Hugging Face")

    # --- 2. Load and preprocess dataset ---
    print("\n--- 2. Loading and Preprocessing Dataset ---")
   
    full_dataset = load_dataset(DATASET_ID, SUBSET, split="train")
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset_full = split_dataset["train"]
    eval_dataset_full = split_dataset["test"]
    print(f"Full dataset split: {len(train_dataset_full)} train / {len(eval_dataset_full)} eval")
    train_dataset_full = train_dataset_full.filter(lambda x: len(x.get("images", [])) > 0 and len(x.get("texts", [])) > 0)
    eval_dataset_full = eval_dataset_full.filter(lambda x: len(x.get("images", [])) > 0 and len(x.get("texts", [])) > 0)
    train_dataset = train_dataset_full.select(range(min(TRAIN_SAMPLE_SIZE, len(train_dataset_full))))
    eval_dataset = eval_dataset_full.select(range(min(EVAL_SAMPLE_SIZE, len(eval_dataset_full))))
    print("Formatting datasets...")
    train_samples, eval_samples = [], []
    for ex in tqdm(train_dataset, desc="Formatting train"):
        f = format_data(ex)
        if f["messages"] is not None:
            train_samples.append(f)
    for ex in tqdm(eval_dataset, desc="Formatting eval"):
        f = format_data(ex)
        if f["messages"] is not None:
            eval_samples.append(f)
    processed_train_dataset = train_samples
    processed_eval_dataset = eval_samples
    print(f"Filtered datasets: {len(processed_train_dataset)} train / {len(processed_eval_dataset)} eval samples.")


    # --- 3. Load model ---

    print("\n--- 3. Loading Model and Configuring LoRA ---")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )
    print("Base model loaded and LoRA configured (weights are not trained yet).")


    # 4. Configure Trainer
    print("\n 4. Configuring Trainer ---")
    HF_USERNAME = "Srisurya-teja"

    MODEL_REPO_NAME = args.output_dir 

    sft_args = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=30,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=MODEL_REPO_NAME, 
        report_to="wandb",
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
        push_to_hub=True,
        hub_model_id=f"{HF_USERNAME}/{MODEL_REPO_NAME}",
        hub_strategy="checkpoint",
        hub_token=os.getenv("HF_TOKEN"),
    )

    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=processed_train_dataset,
        eval_dataset=processed_eval_dataset,
        args=sft_args,
    )
    print("Trainer configured successfully.")


  
    if not args.skip_train:
        print("\n--- 5. Starting Training ---")
        trainer.train()
        print("--- Finetuning Complete ---")
        model = trainer.model
        print("Model variable updated with trained weights.")
    else:
        print("\n--- 5. SKIPPING TRAINING ---")
        print(f"Loading trained model from checkpoint: {MODEL_REPO_NAME}")
        
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=MODEL_NAME, 
            load_in_4bit=True,
        )
        
        model = FastVisionModel.get_peft_model(
            model,
            adapter_model=MODEL_REPO_NAME, 
        )
    
        trainer.model = model
        print(f"Successfully loaded adapters from {MODEL_REPO_NAME}")



 
    print("\n--- 6. Evaluating Models (METEOR) ---")
    meteor = load("meteor")

  
    def generate_responses(model_to_eval, tag, tokenizer_to_use):
        print(f"\nGenerating predictions for {tag} model...")
        model_to_eval.eval()
        predictions, references = [], []
        
        eval_table = wandb.Table(columns=["Image", "Prompt", "Ground Truth", "Prediction"])
        raw_results = []

        for sample in tqdm(processed_eval_dataset, desc=f"Evaluating {tag}"):
            
           
            gt_text = sample["messages"][1]["content"][0]["text"]
            
        
            user_msg_content = sample["messages"][0]["content"]
            prompt_text_for_logging = next(item["text"] for item in user_msg_content if item["type"] == "text")
            prompt_image = next(item["image"] for item in user_msg_content if item["type"] == "image")

            try:
                with torch.no_grad():
                    
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt_text_for_logging},
                            ]
                        }
                    ]
                    

                    input_text = tokenizer_to_use.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False, 
                    )
                    
                
                    inputs = tokenizer_to_use(
                        prompt_image,
                        input_text,
                        return_tensors="pt",
                    ).to(model_to_eval.device)

                
                    outputs = model_to_eval.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                    )
                    

                    prompt_len = inputs["input_ids"].shape[1]
                    pred_tokens = outputs[0, prompt_len:]
                    pred_text = tokenizer_to_use.decode(pred_tokens, skip_special_tokens=True)
                    
            except Exception as e:
                pred_text = ""
                print(f"\n[WARN] {tag} model generation failed for one sample: {e}\n") 

            predictions.append(pred_text)
            references.append(gt_text)
            
            try:
                eval_table.add_data(wandb.Image(prompt_image), prompt_text_for_logging, gt_text, pred_text)
            except Exception as e:
                print(f"[WARN] WandB table logging failed for one sample: {e}")

            raw_results.append({
                "prompt": prompt_text_for_logging,
                "ground_truth": gt_text,
                "prediction": pred_text
            })

        score = meteor.compute(predictions=predictions, references=references)
        print(f"\n {tag} METEOR score: {score['meteor']:.4f}")
        
        try:
            wandb.log({f"{tag}_eval_table": eval_table, f"{tag}_meteor": score['meteor']})
        except Exception as e:
            print(f"[ERROR] Failed to log WandB Table: {e}")

        with open(f"{tag}_evaluation_results.json", "w") as f:
            json.dump(raw_results, f, indent=4)
        print(f"Saved raw evaluation results to {tag}_evaluation_results.json")
        
        return score["meteor"]


    # Base model
    base_model, base_tokenizer = FastVisionModel.from_pretrained(MODEL_NAME, load_in_4bit=True)
    FastVisionModel.for_inference(base_model)
    base_score = generate_responses(base_model, "Base", base_tokenizer)

 
    FastVisionModel.for_inference(model)
    tuned_score = generate_responses(model, "Fine-tuned", tokenizer)

    print("\n--- METEOR Evaluation Summary ---")
    print(f"Base Model METEOR:       {base_score:.4f}")
    print(f"Fine-tuned Model METEOR: {tuned_score:.4f}")
    
    
 
    print("\n--- 7. Merging LoRA Adapters ---")
    
    if hasattr(model, 'merge_and_unload'):
        model = model.merge_and_unload()
        print("Model merged successfully.")
    else:
        print("Model does not have merge_and_unload. Maybe it's already merged or not a PEFT model.")

    merged_model_path = f"{MODEL_REPO_NAME}-merged"
    model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"Merged model saved to {merged_model_path}")

    if not args.skip_train: 
        try:
            print("Pushing merged model to Hub...")
            model.push_to_hub(f"{HF_USERNAME}/{MODEL_REPO_NAME}-merged", token=os.getenv("HF_TOKEN"))
            tokenizer.push_to_hub(f"{HF_USERNAME}/{MODEL_REPO_NAME}-merged", token=os.getenv("HF_TOKEN"))
            print("Merged model pushed successfully.")
        except Exception as e:
            print(f"Error pushing merged model: {e}")


    wandb.finish()
    print("\n--- Script Finished Successfully ---")


if __name__ == "__main__":
    main()