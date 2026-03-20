# 📚 Course Recommendation System using Sentence Transformers

## 📌 Overview
This project builds a content-based course recommendation system that suggests relevant courses based on user input skills. It uses Sentence-BERT embeddings to measure semantic similarity between courses and user queries.

The system is trained on combined datasets (Udemy + Coursera) and retrieves the top-K most similar courses using cosine similarity.

---

## 🚀 Key Features
- Skill extraction from course descriptions  
- Transformer-based semantic similarity (Sentence-BERT)  
- Custom training using positive pairs  
- Model evaluation with multiple metrics  
- Fast retrieval using precomputed embeddings  
- Supports multiple course platforms  

---

## 🗂️ Dataset
The dataset is created by merging multiple sources:

### Sources:
- Udemy Courses Dataset  
- Coursera Courses Dataset  

### Final Dataset Structure:
| Column | Description |
|--------|------------|
| Title  | Course name |
| Skills | Extracted or provided skills |
| Url    | Course link |

---

## ⚙️ Tech Stack
- Python  
- pandas  
- numpy  
- scikit-learn  
- sentence-transformers  
- PyTorch  
- matplotlib  

---

## 🧠 Model Details

### Base Model
`sentence-transformers/all-MiniLM-L6-v2`

### Training Strategy
- Positive pairs created based on skill overlap  
- Minimum overlap threshold: 2  
- Loss function: MultipleNegativesRankingLoss  

### Training Configuration
- Epochs: 4  
- Batch Size: 32  
- Warmup Ratio: 0.1  

---

## 🔍 How It Works
1. Data preprocessing and skill extraction  
2. Create training pairs based on shared skills  
3. Train Sentence-BERT model  
4. Generate embeddings for all courses  
5. Compute similarity using dot product  
6. Retrieve top-K similar courses  

---

## 📊 Evaluation Metrics
- Average Precision (AP)  
- ROC-AUC Score  
- F1 Score  
- Precision  
- Recall  
- Confusion Matrix  

---

## 🧪 Example Usage

```python
skill_query = "Machine Learning"

top_courses = retrieve_top_k_courses(
    model,
    df,
    course_embeddings,
    skill_query,
    k=5
)

print(top_courses)
```

---

## 📁 Project Structure

```
project/
│
├── data/
│   ├── Udemy_Courses.csv
│   ├── coursera_course_2024.csv
│   └── coursera_course_dataset_v3.csv
│
├── saved_course_model/
├── course_embeddings.npy
│
├── main.py
├── README.md
```

---

## 💾 Installation & Setup


### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Run the Project
```
python main.py
```

---

## 📈 Future Improvements
- Add negative sampling for better training  
- Use FAISS for faster similarity search  
- Deploy as a web app (Flask / Streamlit)  
- Improve skill extraction using NLP models  

---

## 👨‍💻 Author
Chamuditha Sheshal

---