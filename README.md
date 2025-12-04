# Qwen2-VL-2B-Instruct-finetune

## Installation

1. Clone the repository:
```
git clone https://github.com/SrisuryaTeja/Fine-Tuning-Qwen2-VL-2B-Vision-Language-Model-.git

```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Set the necessary environment variables:
   - `WANDB_API_KEY`: Your Weights & Biases API key
   - `HF_TOKEN`: Your Hugging Face Hub token

## Usage

1. Run the main script:
```
python main.py
```
   - Optional arguments:
     - `--skip-train`: Skip the training step and load the model from the `--output-dir`
     - `--output-dir`: Directory to save/load model checkpoints (default: `Qwen2-VL-2B-Instruct-geomverse-lora`)

The script will perform the following steps:
1. Set up the environment and initialize Weights & Biases.
2. Load and preprocess the dataset.
3. Load the base model and configure LoRA.
4. Configure the Trainer.
5. Train the model (unless `--skip-train` is specified).
6. Evaluate the base and fine-tuned models using the METEOR metric.
7. Merge the LoRA adapters and save the final model.

## API

The main script provides the following functions:

- `setup_wandb(project_name: str, run_name: str)`: Initializes Weights & Biases for experiment tracking.
- `convert_rgb(image: Image.Image) -> Image.Image`: Converts any image to RGB, handling RGBA with a white background.
- `reduce_image_size(image: Image.Image, scale: float = 0.5) -> Image.Image`: Reduces image size to prevent VRAM OOM errors.
- `format_data(sample: dict) -> dict`: Converts dataset samples into Unsloth-compatible multimodal chat format.

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Implement your changes.
4. Test your changes.
5. Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
