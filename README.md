
Multimodal Retrieval System README
Introduction
This project implements a Multimodal Retrieval System using both text and images as input data. The system retrieves relevant images based on a given review text and vice versa. It utilizes various techniques for feature extraction, similarity calculation, and ranking to achieve efficient retrieval.

Project Structure
Image_Feature_Extraction.ipynb: Notebook containing code for image feature extraction using a pre-trained Convolutional Neural Network (CNN).
Text_Feature_Extraction.ipynb: Notebook containing code for text feature extraction using TF-IDF.
Image_and_Text_Retrieval.ipynb: Notebook containing code for image and text retrieval.
Combined_Retrieval.ipynb: Notebook containing code for combined retrieval and analysis.
utils.py: Python script containing utility functions for preprocessing and similarity calculation.
data/: Folder containing preprocessed data and saved features.
README.md: Readme file explaining the project and providing instructions.
Instructions for Running the Code
Image_Feature_Extraction.ipynb: Run this notebook to extract features from images. Make sure to specify the path to the preprocessed image folder and the location to save the extracted features.
Text_Feature_Extraction.ipynb: Run this notebook to extract features from text reviews. Ensure that the review data is properly preprocessed and TF-IDF scores are calculated. Save the extracted features using the pickle module.
Image_and_Text_Retrieval.ipynb: Run this notebook for image and text retrieval. Provide the input (image, review) pair and execute the code to find similar images and reviews.
Combined_Retrieval.ipynb: Run this notebook to perform combined retrieval and analyze the results. Compute composite similarity scores and rank the pairs based on these scores.
Sample Test Case
The sample test case provided in the assignment is implemented in each notebook to demonstrate the functionality of the retrieval system.
The input (image, review) pair is provided, and the output includes the most similar images and reviews along with cosine similarity scores and composite similarity scores.
Libraries Used
Pandas: For data manipulation.
NumPy: For numerical computing.
TensorFlow: For image classification using pre-trained CNNs.
Scikit-learn (sklearn): For model evaluation and preprocessing.
NLTK: For text processing, including tokenization, stemming, and lemmatization.
PIL (Python Imaging Library): For manipulating image formats.
Challenges Faced and Potential Improvements
Handling of Large Datasets: Processing a large number of images and reviews may lead to memory and computational challenges. Implementing batch processing techniques or using distributed computing frameworks can help overcome this.
Quality of Pre-trained Models: The performance of image retrieval heavily depends on the quality of pre-trained models. Fine-tuning or using more advanced architectures may improve accuracy.
Data Quality and Diversity: The effectiveness of retrieval techniques may be limited by the quality and diversity of the dataset. Collecting more diverse data and performing data augmentation can enhance performance.
Hyperparameter Tuning: Adjusting parameters such as vector dimensions, similarity thresholds, and preprocessing techniques can significantly impact retrieval performance. Conducting thorough hyperparameter tuning experiments can optimize results.
