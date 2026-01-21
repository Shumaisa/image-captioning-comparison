# image-captioning-comparison
Comparison of BLIP, ViT-GPT2, and Microsoft GIT models for image captioning
# Image Captioning Model Comparison

## 🧠 Overview

This project compares three state-of-the-art image captioning models:
- **BLIP** (Salesforce)
- **ViT-GPT2** (Hugging Face)
- **Microsoft GIT**

Users can upload an image and the notebook will:
✔ Generate captions using all three models  
✔ Measure inference speed  
✔ Calculate BLEU & ROUGE evaluation metrics  
✔ Visualize results with bar chart and scatter plots  

---

## 📌 Features

- Upload an image in Google Colab
- Compare model performance (speed + quality)
- Plot comparison charts
- Use real benchmarks

---

## 🛠️ Libraries Used

This notebook uses the following Python libraries:

- `torch` (PyTorch)
- `transformers` (Hugging Face)
- `Pillow` (Image processing)
- `matplotlib` (Visualizations)
- `nltk` (BLEU score)
- `rouge` (ROUGE score)

---

## 🧾 Models Used

1. **BLIP** – Salesforce image captioning model  
   (e.g., *Salesforce/blip-image-captioning-base*)  
2. **ViT-GPT2** – Vision Transformer + GPT-2  
   (*nlpconnect/vit-gpt2-image-captioning*)  
3. **Microsoft GIT** – GIT captioning model  
   (*microsoft/git-base-coco*)

---

## 🛠️ Tools & Technologies

- **Python** – Primary programming language  
- **Google Colab** – Execution environment
- **GitHub** – Version control & hosting
- **Markdown** – Documentation format
- **Image Processing** – PIL library

---

## 🧪 How to Run

1. Open `main.ipynb` in Google Colab  
2. Install required libraries:  
!pip install -r requirements.txt
3. Upload an image
4. Run all cells

---

## 📁 Folder Structure

image-captioning-comparison/
│
├── main.ipynb
├── requirements.txt
├── sample_images/ ← test images
│ └── test_image.jpg
├── README.md
└── LICENSE

---

## 📌 License

This project is licensed under the **MIT License**.




