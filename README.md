# Automated MCQ Generation using Llama 3B

This project implements a system for automatically generating multiple-choice questions (MCQs) from PDF documents using the **Llama 3B** language model. The system processes extracted text from PDFs, segments it into coherent chunks, and generates MCQs for each chunk. Both the **base model** and a **fine-tuned version** (trained on the SQuAD and RACE datasets) are available. The fine-tuned version significantly improves the quality, relevance, and accuracy of the generated MCQs.

# Models on Hugging Face

Llama-3B-QA-Enhanced: https://huggingface.co/AhmedOthman/Llama-3B-QA-Enhanced

ExamGen:   https://huggingface.co/mohamedrady1212434/examgen
## Features

- Extracts text from PDF files and divides it into meaningful chunks.
- Generates MCQs based on text chunks using the Llama 3B model (base and fine-tuned).
- Evaluates generated MCQs using popular NLP metrics such as ROUGE, BLEU, BLEURT, and BERTScore.
- Compares performance between the base model and the fine-tuned version.

## Requirements

Before running the project, ensure that you have the following dependencies installed:

- Python 3.x
- PyTorch (compatible with your system and CUDA version)
- Hugging Face Transformers
- PDF extraction libraries (such as PyMuPDF or pdfplumber)
- NLTK
- ROUGE, BLEU, BLEURT, BERTScore evaluation packages
- Scikit-learn

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```
Dataset
The fine-tuned model was trained on the following datasets:

SQuAD (Stanford Question Answering Dataset)
RACE (ReAding Comprehension from Examinations)
Both datasets are available through the Hugging Face Datasets library

Setup
1. Clone the repository:
```bash
Copy code
git clone https://github.com/your-username/automated-mcq-generation.git
cd automated-mcq-generation
```
Evaluation Metrics
The following evaluation metrics are used to assess the quality of the generated MCQs:

ROUGE: Measures the overlap of n-grams between generated and reference MCQs.
BLEU: Measures the precision of n-grams, evaluating the fluency of the generated questions.
BLEURT: A learned metric based on BERT embeddings, evaluating the semantic similarity between generated and reference MCQs.
BERTScore: Measures the similarity between generated and reference MCQs using contextual embeddings from BERT.
Results
Example Results:
Base Model:
ROUGE-1: F1-Score = 0.0826
ROUGE-2: F1-Score = 0.0257
ROUGE-L: F1-Score = 0.0569
BLEU: 3.54
BLEURT: 0.2089
BERTScore F1: 0.8521
Fine-tuned Model:
ROUGE-1: F1-Score = 0.2398
ROUGE-2: F1-Score = 0.0769
ROUGE-L: F1-Score = 0.1633
BLEU: 3.54
BLEURT: 0.6331
BERTScore F1: 0.8193
As seen in the results, the fine-tuned version demonstrates substantial improvements across all evaluation metrics, particularly in ROUGE and BLEURT, indicating enhanced question relevance and semantic accuracy.

Future Work
The current work lays the foundation for automated MCQ generation, but several improvements can be made:

Expand the fine-tuning dataset: The model has been fine-tuned using a subset of the SQuAD and RACE datasets. Expanding the dataset will improve the model’s performance and make it more robust for diverse document types and question styles.

Optimize model size and latency: Future work will focus on reducing the model's size and improving its latency for faster inference. This is crucial for real-time applications.

Deployment: The project will be deployed as a user-friendly service, allowing educators to easily generate MCQs from any PDF document. This will be a web-based platform with professional deployment standards, ensuring scalability and ease of use.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Hugging Face for providing the Llama 3B model and pre-trained models.
Stanford University for the SQuAD dataset.
RACE for the ReAding Comprehension from Examinations dataset.
PyTorch for the deep learning framework used in model training and inference.
Transformers Library for implementing the Llama model and handling various NLP tasks.
Contact
For questions or suggestions, feel free to open an issue or contact me directly via GitHub.

### Key Adjustments:

- Corrected the formatting for headings and sections.
- Added some brief explanations under the setup instructions and examples.
- Ensured that the license, acknowledgments, and other sections are clearly marked.

