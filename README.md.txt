AI Image Captioning Comparison
This project compares three different AI models to see which one describes images the best and which one is the fastest.

The Models:
BLIP (Salesforce)
ViT-GPT2
Microsoft GIT
How to Run
1.Install the requirements:
pip install -r requirements.txt
2.Run the script:
bash
python main.py
3.The script will look for an image named test_image.jpg in the folder.

```python
import torch
import time
from PIL import Image
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# --- IMPORTS FOR ALL 3 MODELS ---
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from transformers import AutoProcessor, GitForCausalLM

# --- SETUP DEVICES ---
device = "cuda" if torch.is_available() else "cpu"
print(f"Using device: {device}")

# --- MODEL 1: BLIP (Salesforce) ---
print("Loading Model 1: BLIP...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# --- MODEL 2: ViT-GPT2 (Hugging Face) ---
print("Loading Model 2: ViT-GPT2...")
gpt2_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
gpt2_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
gpt2_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)

# --- MODEL 3: GIT (Microsoft) ---
print("Loading Model 3: Microsoft GIT...")
git_processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
git_model = GitForCausalLM.from_pretrained("microsoft/git-base-coco").to(device)

print("\n" + "="*60)
print("ALL 3 MODELS LOADED. READY.")
print("="*60)

# --- FILE LOAD SECTION ---
# For a local project, we look for a specific file name.
# Place an image named 'test_image.jpg' in the same folder as this script.
filename = "test_image.jpg"

try:
    image = Image.open(filename).convert("RGB")
    print(f"\n🖼️  Processing: {filename}")
    print("-" * 60)
    
    # Optional: Display image if you are in a notebook, otherwise skip to save console space
    # display(image) 

    # --- 1. TEST BLIP ---
    start = time.time()
    inputs = blip_processor(image, return_tensors="pt").to(device)
    out_blip = blip_model.generate(**inputs)
    blip_caption = blip_processor.decode(out_blip[0], skip_special_tokens=True)
    blip_time = time.time() - start

    # --- 2. TEST VIT-GPT2 ---
    start = time.time()
    pixel_values = gpt2_processor(images=image, return_tensors="pt").pixel_values.to(device)
    output_ids = gpt2_model.generate(pixel_values)
    gpt2_caption = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    gpt2_time = time.time() - start

    # --- 3. TEST GIT (Microsoft) ---
    start = time.time()
    inputs_git = git_processor(images=image, return_tensors="pt").to(device)
    pixel_values_git = inputs_git.pixel_values
    output_ids_git = git_model.generate(pixel_values=pixel_values_git)
    git_caption = git_processor.batch_decode(output_ids_git, skip_special_tokens=True)[0]
    git_time = time.time() - start

    # --- RESULTS TABLE ---
    print(f"🔵 BLIP:         {blip_caption}  |  ⏱️  {blip_time:.4f}s")
    print(f"🟢 ViT-GPT2:     {gpt2_caption}  |  ⏱️  {gpt2_time:.4f}s")
    print(f"🟣 Microsoft GIT: {git_caption} |  ⏱️  {git_time:.4f}s")

    print("-" * 60)

    # Determine Winner (Speed)
    times = {"BLIP": blip_time, "ViT-GPT2": gpt2_time, "GIT": git_time}
    fastest_model = min(times, key=times.get)
    print(f"🏁 Fastest Model: {fastest_model} ({times[fastest_model]:.4f}s)")

    # --- EVALUATION METRICS ---
    print("\n" + "="*30 + " EVALUATION METRICS " + "="*30)
    # You can change this reference text to match your specific image
    reference = ["A cat sitting on a chair in a living room."]  

    # BLEU Score
    print("\n📊 BLEU Score")
    blip_bleu = sentence_bleu(reference, blip_caption.split())
    gpt2_bleu = sentence_bleu(reference, gpt2_caption.split())
    git_bleu = sentence_bleu(reference, git_caption.split())
    print(f"🔵 BLIP:         {blip_bleu:.4f}")
    print(f"🟢 ViT-GPT2:     {gpt2_bleu:.4f}")
    print(f"🟣 Microsoft GIT: {git_bleu:.4f}")

    # ROUGE Score
    print("\n📊 ROUGE Score")
    rouge = Rouge()
    blip_rouge = rouge.get_scores(blip_caption, reference[0])[0]
    gpt2_rouge = rouge.get_scores(gpt2_caption, reference[0])[0]
    git_rouge = rouge.get_scores(git_caption, reference[0])[0]
    print(f"🔵 BLIP:         {blip_rouge}")
    print(f"🟢 ViT-GPT2:     {gpt2_rouge}")
    print(f"🟣 Microsoft GIT: {git_rouge}")

    # --- BAR GRAPH: SPEED COMPARISON ---
    print("\nGenerating Graphs...")
    models = ['BLIP', 'ViT-GPT2', 'GIT']
    speeds = [blip_time, gpt2_time, git_time]

    plt.figure(figsize=(10, 5))
    plt.bar(models, speeds, color=['blue', 'green', 'purple'])
    plt.xlabel('Models')
    plt.ylabel('Time (seconds)')
    plt.title('Speed Comparison')
    plt.savefig('speed_comparison.png') # Save graph to file
    plt.show()

    # --- SCATTER PLOT: ACCURACY METRICS ---
    bleu_scores = [blip_bleu, gpt2_bleu, git_bleu]
    rouge_scores = [blip_rouge['rouge-1']['f'], gpt2_rouge['rouge-1']['f'], git_rouge['rouge-1']['f']]

    plt.figure(figsize=(10, 5))
    plt.scatter(models, bleu_scores, color='blue', label='BLEU Score')
    plt.scatter(models, rouge_scores, color='red', label='ROUGE Score')
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Accuracy Metrics')
    plt.legend()
    plt.savefig('accuracy_metrics.png') # Save graph to file
    plt.show()

except FileNotFoundError:
    print("Error: Could not find 'test_image.jpg'.")
    print("Please put an image named 'test_image.jpg' in this folder.")

print("\n" + "="*60)
print("3-MODEL COMPARISON COMPLETE")
print("="*60)
