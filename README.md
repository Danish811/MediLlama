---

# **MediLlama: A Healthcare Chatbot**
**MediLlama** is a fine-tuned version of the LLaMA-3 language model designed to assist with healthcare-related queries. It leverages **LoRA (Low-Rank Adaptation)** and **4-bit quantization** to provide accurate, efficient, and memory-optimized medical advice while ensuring real-time response capabilities.

---

## **Features**
- **Fine-Tuned Medical Model**: Tailored to healthcare queries for accurate and coherent responses.
- **Efficient Training**: Utilizes LoRA adapters and 4-bit quantization to reduce memory usage by ~75%.
- **Dynamic Prompt Engineering**: Structured prompts for consistent and contextually aware outputs.
- **Memory-Efficient Inference**: Supports real-time query processing with low GPU memory overhead.
- **Dataset**: Trained on 100k medical queries from the [ChatDoctor-HealthCareMagic](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) dataset.

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MediLlama.git
   cd MediLlama
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the model and tokenizer:
   ```python
   from unsloth import FastLanguageModel
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name="unsloth/llama-3-8b-bnb-4bit",
       load_in_4bit=True,
       max_seq_length=2048
   )
   ```

---

## **Training Pipeline**
MediLlama is trained using **Hugging Face’s `transformers`** and `trl` libraries. Below is the training configuration:
- **Batch Size:** 2 (per device).
- **Epochs:** 4.
- **Learning Rate:** \(2 \times 10^{-4}\).
- **Optimizer:** AdamW (8-bit precision).
- **Gradient Accumulation:** 4 steps.
- **LoRA Configuration:** 
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.
  - Rank: 16, Alpha: 16, Dropout: 0%.


---

## **Usage**
After training, you can use the model to generate responses for healthcare queries:
```python
inputs = tokenizer(
    [
        "Below is an instruction that describes a task, paired with an input. Write an output that appropriately completes the request.\n\n"
        "### Instruction:\nProvide advice for the following condition.\n\n"
        "### Input:\nI woke up this morning feeling the whole room is spinning. When I move around, I feel nauseous. What should I do?\n\n"
        "### Output:\n"
    ], 
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(response[0])
```

---

## **Performance**
- **Training Time:** ~90 minutes for 60 steps (can be extended for better learning).
- **Memory Usage:** 
  - Peak memory: ~7 GB (training).
  - LoRA adapters: ~2 GB (training and inference).
- **Response Time:** Real-time inference with low latency.

---

## **Examples**
**Input:**
> I woke up feeling the whole room is spinning when sitting down. I feel nauseous and unsteady. What could this be?

**Output:**
> Based on your symptoms, you might be experiencing vertigo, possibly caused by an inner ear issue like BPPV (Benign Paroxysmal Positional Vertigo). It's recommended to stay hydrated, avoid sudden movements, and consult a healthcare provider if the symptoms persist.

![MediLlama](https://github.com/user-attachments/assets/3628ed3f-da39-4ce6-a941-06da8c4d15a6)


---

## **Contributing**
We welcome contributions! Feel free to submit issues or pull requests to improve MediLlama.

---
