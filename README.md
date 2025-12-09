# NLP-team-project
This project focuses on building and evaluating models for sentiment analysis, a core task in Natural Language Processing. The goal is to classify movie reviews as either positive or negative using the Large Movie Review Dataset (IMDb), a standard benchmark for binary sentiment classification.

# Fine-Grained IMDb Sentiment Analysis with Attention Mechanism
## 📌 Project Overview
This project implemented a Fine-Grained Sentiment Analysis model instead of a normal binary. The goal is to predict the specific star rating of movie reviews from the Large Movie Review Dataset (IMDb).

Unlike traditional approaches that only predict "Positive" or "Negative", our model handles the complexity of Ordinal Classification across 8 distinct star ratings (1, 2, 3, 4, 7, 8, 9, 10).

IDE: Jupyter Notebook, Language:Python 3.11

## 🚀 Key Features
**Advanced Architecture**: Bidirectional LSTM coupled with a custom Attention Mechanism to focus on key emotional words.

**Fine-Grained Classification**: Handles 8 specific classes, mapping raw IMDb ratings (missing 5 & 6 stars) to continuous indices.

**Embedding Strategies**: Compares performance between:

1. Model A: Embeddings trained from scratch.

2. Model B: Pre-trained GloVe word embeddings (100d).

**Optimized Data Pipeline**:

1. Subword Tokenization: Uses tokenizers (WordPiece) for better OOV handling.

2. Smart Batching: Implements BucketSampler to group sequences of similar lengths, reducing padding overhead.

3. Caching: Pre-processes and saves data to .pt files for faster restarts.

**Comprehensive Evaluation**:   
1. MAE (Mean Absolute Error): The primary metric for ordinal regression.

2. Confusion Matrix: Visualizes misclassifications across all 8 classes.

3. Error Bucket Analysis: Categorizes errors into "Off-by-one" (minor) vs. "Large Errors" (>5 stars difference) to understand failure modes.

4. Reliability Diagram: Calculates ECE to determine if the model's confidence matches its actual accuracy.

5. Robustness Analysis: Tests the model on specific linguistic challenges like Negation (e.g., "not good") and Sarcasm to gauge generalization.

## 🛠️ Requirements
After our trials, we have summarized the configuration steps for an FSC 8/F 5060 computer:
1. Open anaconda prompt 
![](<anaconda prompt.png>)and enter the following code to create the Python 3.11 environment:

    ```python
    conda create -n py311 python=3.11
    ```

2. Once the installation is complete, enter the code to activate the py311 environment:

    ```python
    conda activate py311
    ```

3. Make sure to install the relevant libraries of the code in the py311 environment ![enviroment py311](<conda activate.png>)

    ```python
    conda install pandas
    conda install scikit-learn
    conda install tqdm
    conda install matplotlib
    conda install tokenizers
    conda install seaborn
    ```

4. Right-click "Run as administrator" to open terminal![termianl](terminal.png) enter

    ```python
    nvidia-smi
    ```

    and confirm that the FSC 8/F 5060 computer has cuda version 12.9.

5. Back to the anaconda prompt, enter the code to install the corresponding version of pytorch supports 5060 cuda:

    ```python
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
    ```

6. Install ipykernel and register the py311 environment as Jupyter Kernel:

    ```python
    conda install -c conda-forge ipykernel -y

    python -m ipykernel install --user --name=py311 --display-name="Python 3.11 (py311)"
    ```
    ![register](<register kernel.png>)
7. Open the jupyter notebook and select our new kernel to run the code
    ![switch kernel](<switch kernel.png>)
    ![select kernel](<select kernel.png>)

## 📂 Dataset & Setup
1. Download Data
    IMDb Dataset: Download aclImdb_v1.tar.gz from https://ai.stanford.edu/~amaas/data/sentiment/
GloVe Embeddings: Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/
2. Configuration
Open the script and modify the CONFIG class at the top to point to your local directories:

    ```python
    class CONFIG:
    # Path to the 'train' folder inside aclImdb
    IMDB_DATA_DIR = r"E:\path\to\aclImdb\train" 
    
    # Path to the specific GloVe text file
    GLOVE_FILE_PATH = r"E:\path\to\glove.6B\glove.6B.100d.txt"
    ```
## 🧠 Model Architecture
The model consists of four main stages:
1. **Embedding Layer**: Converts token IDs into dense vectors (Random or GloVe).
2. **Bi-LSTM**: Captures sequential context in both forward and backward directions.
3. **Attention Layer**: Computes a weighted sum of LSTM hidden states.
    
    Formula:$$C = \sum \alpha_i h_i$$
    Allows the model to "attend" to specific adjectives (e.g., "terrible", "masterpiece") while ignoring noise.
4. **Classifier**: A fully connected layer mapping the context vector to 8 output logits.

## 📊 Evaluation Metrics
Since this is an ordinal task, **Accuracy is misleading**. We prioritize metrics that understand "distance":
1. **MAE** (Mean Absolute Error): Measures how far off the prediction is from the true star rating (e.g., predicting 8 stars when the truth is 9 is better than predicting 1).
2. **Confusion Matrix**: A heatmap using seaborn to show where the model gets confused (e.g., confusing 7-star with 8-star).
3. **Error Buckets**: Analyzes "Off-by-one" errors vs. "Catastrophic" errors.
4. **Reliability Diagram**: Visualizes how well the predicted probabilities align with actual accuracy.

## 🤝 Acknowledgements
We would like to express our gratitude to the following resources and individuals that made this project possible:

**Course & Lab Resources**:
1. Special thanks to our professor and teaching assistants for their guidance.
2. The project has been tested and optimized to run in the FSC 8/F laboratory environment.

**Datasets & Embeddings**:
1. IMDb Dataset: Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment Analysis*. 
2. GloVe Embeddings: Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation*.

**Theoretical References**:
The Attention Mechanism implementation was inspired by Bahdanau et al.'s work on *Neural Machine Translation by Jointly Learning to Align and Translate (2014)*.

**Open Source Libraries**:
This project is built upon the open-source ecosystem, specifically PyTorch, Hugging Face Tokenizers, Scikit-learn, Pandas, Seaborn, etc.
## 👥 Contributors
ZHANG Jinghan, XU Wenjing, LIANG Haozhe, ZHANG
Xiaowen
