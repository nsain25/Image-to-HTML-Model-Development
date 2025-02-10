# WebSight: Image-to-HTML Generation

## Project Overview

This project focuses on developing a deep-learning model that takes an image as input and generates its corresponding HTML code. The primary dataset used is the **WebSight** dataset from Hugging Face, which includes paired images and their HTML representations.

---

## Objective

The goal of this project is to design and implement an AI model capable of:
- Converting visual layouts into accurate, syntactically correct HTML code.
- Utilizing pre-trained models for efficient training and accurate generation.

---

## Dataset

- **Source:** [HuggingFace WebSight Dataset](https://huggingface.co/datasets/HuggingFaceM4/WebSight)
- **Structure:** Each entry consists of an image and its corresponding HTML code.
- **Preprocessing:** Tokenization of HTML code before feeding it into the model.

---

## Approach

### Model Selection

1. **Vision-Language Models:**
   - **CLIP + Transformer:** CLIP extracts image embeddings, while a fine-tuned GPT-2 or T5 model generates HTML.
   - **Image Captioning Models:** Models like BLIP or OFA, treat HTML as a form of image captioning.

2. **Deep Learning Architectures:**
   - CNNs, RNNs, or Transformer-based architectures to process image inputs and generate HTML sequences.

3. **Custom Pipelines:**
   - Hybrid methods integrating multiple models for optimized performance.

---

## Implementation Steps

### Data Preprocessing
- **HTML Tokenization:** Applied transformer-based tokenization for converting HTML into tokens suitable for model training.

### Model Training
- **Pre-trained Models:** Utilized pre-trained models for both feature extraction and sequence generation.
- **Subset Training:** Trained on a smaller subset of the dataset for faster processing.
- **Cloud Resources:** Though designed for Google Colab, this project is adaptable to Amazon SageMaker or Google Cloud Console for larger-scale training.

### Evaluation & Metrics
- **BLEU Score:** Compared generated HTML against ground truth.
- **Token-Level Accuracy:** Checked the correctness of the token sequences.
- **Structural Validity:** Ensured the generated HTML was structurally valid.

### Deployment & Testing
- **Model Saving:** The trained model checkpoint is saved for further use.
- **Google Colab Notebook:** Provided for demonstration and testing.
- **Fine-tuning:** Further refinements were made to improve accuracy.

---

## Results & Findings

The model demonstrated promising capabilities in generating structurally correct HTML code from images. While the BLEU scores and token-level accuracy were satisfactory, there remains room for further fine-tuning and optimization.

---

## Challenges Faced

1. **Resource Constraints:**
   - Google Colab’s disk space limitations posed challenges, requiring optimization of data handling and model size.

2. **Dataset Size:**
   - The WebSight dataset is large, necessitating careful management of data subsets to prevent memory overload.

3. **Training Time:**
   - Despite using smaller subsets, training remained time-intensive.

4. **HTML Complexity:**
   - Ensuring the structural validity of the generated HTML code proved challenging, especially for complex layouts.

---

## Scope for Improvement

1. **Resource Optimization:**
   - Transition to more powerful cloud platforms like AWS SageMaker for full dataset training.

2. **Model Enhancements:**
   - Integrate advanced architectures or ensemble models to improve HTML generation quality.

3. **Multimodal Integration:**
   - Combine image data with additional contextual metadata to enhance accuracy.

4. **Advanced Evaluation Metrics:**
   - Implement more robust evaluation techniques to measure semantic accuracy and usability of the generated HTML.

5. **Real-World Deployment:**
   - Create a user-friendly web application where users can upload images and receive HTML code in real-time.

---

## Submission Details

1. **Model File:** Trained model checkpoint included.
2. **GitHub Repository:** Contains all source code, including preprocessing, training, and evaluation scripts.
3. **Google Colab Demonstration:** Jupyter notebook for testing and showcasing the model's capabilities.
4. **Implementation Video:** Detailed explanation and demonstration of the model in action.
5. **Final Report:** This document serves as the final project report.

---

## Repository Structure

```
WebSight-Project/
|│-- model/
|    |│-- trained_model_checkpoint.pt
|│-- notebooks/
|    |│-- WebSight_Project_Demo.ipynb
|│-- src/
|    |│-- preprocessing.py
|    |│-- training.py
|    |│-- evaluation.py
|│-- README.md
|│-- requirements.txt
|│-- implementation_video.mp4
```

---

## Conclusion

This project successfully demonstrates the potential of AI models to convert images into functional HTML code. While challenges related to resources and dataset size were encountered, the results indicate a strong foundation for further development and real-world applications.


