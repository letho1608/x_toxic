# X Toxic: Toxic Comment Classifier In Twitter

🌟 This project is a **Toxic Comment Classifier** web application built with **TensorFlow**. The model is trained to classify comments as either toxic or non-toxic.

## 🚀 Features

- **Text classification**: The app uses a deep learning model (built with TensorFlow) to classify comments as toxic or non-toxic.
- **Language detection and translation**: If the user inputs a comment in a language other than English, the app will automatically translate it into English using Google Translate.
- **Model Evaluation**: The model's performance is evaluated using metrics like accuracy, confusion matrix, and precision-recall.

## 🎯 Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/letho1608/x_toxic
2. **Run** click the file caidat.bat -> huanluyen.bat ( or not because I've already run it ) - > run.bat

## 🧠 Training the Model

To train the model, the following steps were performed:

1. **Data Preprocessing**:  
   The text data was cleaned, tokenized, and processed to remove noise such as special characters, stopwords, and irrelevant words. This step ensures that the text is in a suitable format for training.

2. **TF-IDF Vectorization**:  
   The cleaned text data was transformed into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization. This technique helps in representing the importance of words in the corpus, which aids in improving model accuracy.

3. **Model Architecture**:  
   A deep learning model was built using **TensorFlow/Keras**. The model architecture consists of:
   - **Embedding layers**: To represent text data in a continuous vector space, capturing semantic meanings.
   - **Dense layers**: Fully connected layers to process and classify the input data.
   - **Activation functions**: Suitable functions like **ReLU** (Rectified Linear Unit) and **Softmax** (for multi-class classification) were used to introduce non-linearity and to map the model’s output to a probability distribution.

4. **Training**:  
   The model was trained on a labeled dataset of comments. It used **cross-entropy loss** (a common loss function for classification problems) and optimization algorithms like **Adam** to adjust the weights and minimize the loss function over multiple epochs.

5. **Model Evaluation**:  
   After training, the model’s performance was evaluated using several metrics:
   - **Accuracy**: To measure how often the model correctly classifies the comments.
   - **Confusion Matrix**: To visualize how the model performed on each class (toxic or non-toxic).
   - **...**

