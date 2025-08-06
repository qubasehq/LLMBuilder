# 🚀 **Complete Beginner's Guide: LLM Training Pipeline**

**Welcome!** This guide will hold your hand through **every single step** of building your own Large Language Model (LLM) from scratch. No technical background needed - we'll explain everything as we go!

**What you'll achieve:**
- ✅ Turn 10 PDF/TXT/DOCX files into a working AI
- ✅ Train a 50-million parameter model
- ✅ Get interactive AI responses
- ✅ Learn by doing (not just reading)

**Time needed:** 2-4 hours (perfect for a weekend project!)

---

## 🏗️ **Before We Start: What Are We Building?**

Imagine you're teaching a child to read and write. We're doing the same thing, but with a computer:

1. **Preprocessing** = Teaching the computer to read your files
2. **Tokenizer** = Teaching it the alphabet (16,000 "letters")
3. **Training** = Teaching it to write sentences
4. **Evaluation** = Testing how well it writes
5. **Fine-tuning** = Teaching it your specific style
6. **Inference** = Having conversations with your AI

---

## 📁 **Step 0: Getting Your Files Ready**

**Don't worry about technical stuff yet!** Just do this:

### **For Windows Users:**
1. Open File Explorer
2. Go to `D:\LLM\project\data\raw\`
3. Drag and drop your 10 files here

### **For Mac/Linux Users:**
1. Open Terminal
2. Run: `cd ~/LLM/project/data/raw`
3. Copy your files: `cp ~/Downloads/*.pdf .`

**Your files should be:**
- PDF files (reports, books, papers)
- TXT files (plain text)
- DOCX files (Word documents)
- **Total size:** 1MB-100MB works great

**What if I don't have files?** No problem! Use any:
- Download free ebooks from [Project Gutenberg](https://www.gutenberg.org)
- Save Wikipedia articles as PDF
- Use your own documents

---

---

## 🔄 Stage-by-Stage Execution

## 🎯 **Let's Start! Stage 1: Preprocessing**

**What this does:** Reads your 10 files and turns them into one clean text file.

### **How to Run:**

**🐧 Linux/Mac:**
```bash
./run.sh preprocess
```

**🪟 Windows (PowerShell):**
```powershell
.\run.ps1 -Stage preprocess
```

**🪟 Windows (Command Prompt):**
```cmd
run.bat preprocess
```

### **What You'll See:**

**Good output (what you want):**
```
=== Starting Data Preprocessing ===
[INFO] Processing 10 files...
[INFO] Extracted text from sample1.pdf (2.3MB)
[INFO] Extracted text from sample2.txt (1.1MB)
[INFO] Extracted text from sample3.docx (850KB)
... (more files)
[INFO] Combined text file size: 15.7MB
[SUCCESS] Preprocessing completed
```

**What this means:** ✅ Your files are being read correctly!

**If you see this error:**
```
[ERROR] No data found in data/raw directory!
```
**Fix:** Copy your files to `data/raw/` folder and try again.

**What gets created:**
- `data/cleaned/combined_text.txt` - One big file with all your text
- `logs/preprocessing.log` - Detailed log of what happened

---

## 🎯 **Stage 2: Teaching the Computer the Alphabet**

**What this does:** Creates a "dictionary" of 16,000 word pieces from your text.

### **How to Run:**

**🐧 Linux/Mac:**
```bash
./run.sh tokenizer
```

**🪟 Windows (PowerShell):**
```powershell
.\run.ps1 -Stage tokenizer
```

**🪟 Windows (Command Prompt):**
```cmd
run.bat tokenizer
```

### **What You'll See:**

**Good output:**
```
=== Stage 2: Tokenizer Training ===
[INFO] Training tokenizer on 15.7MB text...
[INFO] Vocabulary size: 16,000 tokens
[SUCCESS] Tokenizer created successfully!
```

**What this means:** ✅ The computer learned your writing style!

**What gets created:**
- `tokenizer/tokenizer.model` - The brain's dictionary
- `tokenizer/tokenizer.json` - Settings file

**If you see this:**
```
[ERROR] No cleaned data found. Please run preprocessing first.
```
**Fix:** Run preprocessing first (Stage 1)!

---

## 🎯 **Stage 3: Training Your AI**

**What this does:** Actually teaches the computer to write like your documents.

### **How to Run:**

**🐧 Linux/Mac:**
```bash
./run.sh train
```

**🪟 Windows (PowerShell):**
```powershell
.\run.ps1 -Stage train
```

**🪟 Windows (Command Prompt):**
```cmd
run.bat train
```

### **What You'll See:**

**During training (this takes 2-4 hours):**
```
=== Stage 3: Model Training ===
[INFO] Training 50M parameter model...
Epoch 1/10 - Loss: 3.421 → 2.187
Epoch 2/10 - Loss: 2.187 → 1.943
Epoch 3/10 - Loss: 1.943 → 1.654
... (keeps going)
[SUCCESS] Training completed! Best model saved.
Model size: 197MB
```

**What this means:** ✅ Your AI is learning! Lower loss = better learning.

**What gets created:**
- `exports/checkpoints/best_model.pt` - Your trained AI brain
- `exports/checkpoints/latest_model.pt` - Latest version
- `logs/training.log` - Detailed training log

**Pro tip:** Grab a coffee! This takes time but you can watch it learn.

---

## 🎯 **Stage 4: Testing Your AI**

**What this does:** Checks how well your AI writes.

### **How to Run:**

**🐧 Linux/Mac:**
```bash
./run.sh eval
```

**🪟 Windows (PowerShell):**
```powershell
.\run.ps1 -Stage eval
```

**🪟 Windows (Command Prompt):**
```cmd
run.bat eval
```

### **What You'll See:**

**Good results:**
```
=== Stage 4: Model Evaluation ===
[INFO] Running evaluation...
Perplexity: 12.43 (lower is better - under 20 is good!)
BLEU Score: 0.67 (higher is better - over 0.6 is great!)

Generation Example:
Prompt: "The future of AI is"
Generated: "The future of AI is bright and full of possibilities..."

[SUCCESS] Evaluation completed!
```

**What these numbers mean:**
- **Perplexity:** How confused your AI is (lower = better)
- **BLEU Score:** How good the writing is (higher = better)

---

## 🎯 **Stage 5: Making It Even Better**

**What this does:** Teaches your AI your specific style.

### **Setup First:**
Create a folder: `data/finetune/`
Add your special training files:
```
data/finetune/
├── my_company_docs.txt
├── technical_manuals.txt
└── personal_writing.txt
```

### **How to Run:**

**🐧 Linux/Mac:**
```bash
./run.sh finetune
```

**🪟 Windows (PowerShell):**
```powershell
.\run.ps1 -Stage finetune
```

### **What You'll See:**

```
=== Starting Fine-tuning ===
[INFO] Loading pre-trained model...
[INFO] Fine-tuning on your custom data...
Epoch 1/5 - Loss: 1.943 → 1.234
[SUCCESS] Fine-tuned model saved to exports/checkpoints/finetuned/
```

**What this means:** ✅ Your AI now knows your specific style!

---

## 🎯 **Stage 6: Talking to Your AI**

**What this does:** Chat with your trained AI!

### **How to Run:**

**🐧 Linux/Mac:**
```bash
./run.sh inference
```

**🪟 Windows (PowerShell):**
```powershell
.\run.ps1 -Stage inference
```

### **What You'll See:**

```
=== Interactive Inference ===
Model loaded successfully!

Enter prompt: Write a Python function to sort a list

Generated Response:
def sort_list(input_list):
    """Sort a list in ascending order."""
    return sorted(input_list)

# Example usage:
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_numbers = sort_list(numbers)
print(sorted_numbers)  # Output: [1, 1, 2, 3, 4, 5, 6, 9]

Continue? (y/n): y

Enter prompt: What did you learn from my files?
[AI responds based on your specific documents...]
```

**This is the magic moment!** You're literally talking to an AI you trained yourself!

---

## 🎯 **Bonus: Downloading Cool Models**

**What this does:** Downloads pre-trained models from the internet.

### **How to Run:**

**🐧 Linux/Mac:**
```bash
./run.sh download
```

**🪟 Windows (PowerShell):**
```powershell
.\run.ps1 -Stage download
```

### **What You'll See:**

```
=== Downloading HuggingFace Model ===
Enter HuggingFace model name: Qwen/Qwen2.5-Coder-0.5B
Enter output directory: ./models/Qwen2.5-Coder-0.5B

Downloading config.json... ✅
Downloading model.safetensors... ✅
Downloading tokenizer.json... ✅
...
Download complete! Files saved to: ./models/Qwen2.5-Coder-0.5B/
```

**Popular models to try:**
- `Qwen/Qwen2.5-Coder-0.5B` (great for code)
- `microsoft/DialoGPT-medium` (great for chat)
- `EleutherAI/gpt-neo-125M` (great for general text)

---

## 🚨 **Common Problems & Solutions**

| Problem | What it looks like | Easy fix |
|---------|-------------------|----------|
| "No data found" | `[ERROR] No data found in data/raw directory!` | Put your files in `data/raw/` |
| "Python not found" | `[ERROR] Python is not installed...` | Install Python 3.8+ from python.org |
| "Out of memory" | CUDA error | Add `--device cpu` to use CPU instead |
| "Training too slow" | Taking forever | Use fewer files or shorter training |
| "Files not found" | File not found errors | Check file paths and extensions |

---

## 🎯 **Quick Start (30 seconds)**

**Just want to see it work? Do this:**

1. **Add your files** to `data/raw/`
2. **Run this command:**
   ```bash
   ./run.sh all
   ```
3. **Come back in 2-4 hours** and chat with your AI!

---

## 📊 **What Success Looks Like**

**After running everything, you'll have:**
- ✅ A trained AI that knows your content
- ✅ Files you can share with friends
- ✅ A working chatbot
- ✅ Bragging rights ("I built this!")

**Your final folder structure:**
```
LLM/project/
├── data/raw/ (your original files)
├── data/cleaned/combined_text.txt (cleaned text)
├── tokenizer/tokenizer.model (AI's dictionary)
├── exports/checkpoints/best_model.pt (your AI brain)
└── logs/ (detailed logs)
```

**Ready to start? Just run:**
```bash
./run.sh
```

**And watch the magic happen!** 🎉

---

### **Stage 2: Tokenizer Training**
**Command**:
```bash
./run.sh tokenizer
```

**What Happens:**
- **Input**: `data/cleaned/combined_text.txt`
- **Process**: Trains SentencePiece BPE tokenizer on your data
- **Output**: `tokenizer/tokenizer.model` + `tokenizer/tokenizer.json`

**Expected Output:**
```
=== Stage 2: Tokenizer Training ===
[INFO] Training tokenizer on 15.7MB text...
[INFO] Vocabulary size: 16,000 tokens
[INFO] Tokenizer created successfully!
```

---

### **Stage 3: Model Training**
**Command**:
```bash
./run.sh train
```

**What Happens:**
- **Input**: Tokenized data + 50M parameter GPT model
- **Process**: Trains 12-layer transformer (768 dim, 12 heads)
- **Output**: `exports/checkpoints/best_model.pt` (≈200MB)

**Expected Output:**
```
=== Stage 3: Model Training ===
[INFO] Training 50M parameter model...
Epoch 1/10 - Loss: 3.421 → 2.187
Epoch 2/10 - Loss: 2.187 → 1.943
...
[SUCCESS] Training completed! Best model saved.
Model size: 197MB
```

---

### **Stage 4: Evaluation**
**Command**:
```bash
./run.sh eval
```

**What Happens:**
- **Input**: Trained model + test data
- **Process**: Calculates perplexity, generation quality
- **Output**: Evaluation report in console + logs

**Expected Output:**
```
=== Stage 4: Model Evaluation ===
[INFO] Running evaluation...
Perplexity: 12.43 (lower is better)
BLEU Score: 0.67
Generation Example:
Prompt: "The future of AI is"
Generated: "The future of AI is bright and full of possibilities..."
```

---

### **Stage 5: Fine-tuning**
**Command**:
```bash
./run.sh finetune
```

**Setup First**: Place fine-tuning data in `data/finetune/`:
```
data/finetune/
├── custom_data.txt
└── domain_specific.txt
```

**What Happens:**
- **Input**: Pre-trained model + fine-tuning data
- **Process**: Continues training on new data
- **Output**: `exports/checkpoints/finetuned/best_model.pt`

**Expected Output:**
```
=== Starting Fine-tuning ===
[INFO] Loading pre-trained model...
[INFO] Fine-tuning on 2 custom datasets...
Epoch 1/5 - Loss: 1.943 → 1.234
[SUCCESS] Fine-tuned model saved to exports/checkpoints/finetuned/
```

---

### **Stage 6: Interactive Inference**
**Command**:
```bash
./run.sh inference
```

**What Happens:**
- **Input**: Latest checkpoint + tokenizer
- **Process**: Interactive text generation
- **Output**: Real-time chat interface

**Expected Interaction:**
```
=== Interactive Inference ===
Model loaded successfully!

Enter prompt: Write a Python function to sort a list

Generated Response:
def sort_list(input_list):
    """Sort a list in ascending order."""
    return sorted(input_list)

# Example usage:
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_numbers = sort_list(numbers)
print(sorted_numbers)  # Output: [1, 1, 2, 3, 4, 5, 6, 9]

Continue? (y/n): y
```

---

### **Stage 7: Download Pre-trained Models**
**Command**:
```bash
./run.sh download
```

**What Happens:**
- **Interactive prompt** for model name
- **Downloads** all files for specified model

**Expected Interaction:**
```
=== Downloading HuggingFace Model ===
Enter HuggingFace model name: Qwen/Qwen2.5-Coder-0.5B
Enter output directory (default: ./models/Qwen2.5-Coder-0.5B): 

Downloading config.json... ✅
Downloading model.safetensors... ✅
Downloading tokenizer.json... ✅
...
Download complete! Files saved to: ./models/Qwen2.5-Coder-0.5B/
```

---

## 📊 Complete Pipeline Run

### **All Stages at Once**
```bash
./run.sh all
```

**Execution Order:**
1. **Preprocess** → 2. **Tokenizer** → 3. **Train** → 4. **Evaluate**

**Total Time**: ~2-4 hours (depending on hardware)
**Final Outputs**:
- `data/cleaned/combined_text.txt` (cleaned corpus)
- `tokenizer/tokenizer.model` (custom tokenizer)
- `exports/checkpoints/best_model.pt` (trained model)
- Evaluation metrics in console/logs

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| "No data found" | Add files to `data/raw/` |
| "Python not found" | Install Python 3.8+ |
| "CUDA out of memory" | Use `/cpu-only` flag |
| "Model too large" | Adjust `config.json` parameters |

---

## 📈 Performance Expectations (10 sample files)

| Metric | Expected Value |
|--------|----------------|
| Training Loss | Starts ~3.5, ends ~1.2 |
| Perplexity | 10-15 |
| Model Size | ~200MB |
| Training Time | 2-4 hours (CPU) |
| Inference Speed | 50-100 tokens/sec |

---

## 🎯 Next Steps

1. **Add more data** → Better model quality
2. **Adjust hyperparameters** → Faster training
3. **Try different models** → Use `download` stage
4. **Export formats** → GGUF/ONNX (future enhancement)

**Ready to start? Run `./run.sh` (or `run.bat`) and watch your LLM come to life!**
