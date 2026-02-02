Analysis of Wine Chemical Classification  
In this project, data explored and analyzed was for a wine dataset. This was to build and evaluate two 
different classification models: k-Nearest Neighbors (k-NN) and Support Vector Machine (SVM). The goal 
was to understand which model performs better on this dataset and to analyze the importance of 
features in classification tasks. Prediction was that the SVM would be more efficient. It is said that SVM 
gives fundamentally better arrangement precision and classification speed than the k-NN.  

Data Exploration   
The dataset was first inspected to understand its structure and content. The dimensions, basic stats, and 
wine class distribution provided an introduction of the dataset. Visualizations used were histograms and 
pair plots which helped in understanding the distribution of features and relations between them. 
Histograms showed that many features have a normal distribution, while the pair plot highlighted the 
relationships between features, this portrayed some clustering patterns that are beneficial for 
classification.   

Data Preprocessing   
Pre-processing involved scaling the features to ensure that they were on a comparable scale, which is 
crucial for distance-based algorithms like k-NN. The dataset was then split into training and testing sets. 
No missing values were found, simplifying the pre-processing stage.   

Model Selection   
Two models were selected for classification: k-NN and SVM. k-NN, a non-parametric algorithm, was used 
with 5 neighbors, while SVM was used with a linear kernel. Both models were trained on the scaled 
training data. 

Model Evaluation   
Classification Result Reports: the classification result reports for both k-NN and SVM showed precision, 
recall, and F1-score for each class. As predicted, SVM generally provided higher precision and recall 
scores compared to k-NN, suggesting better performance in distinguishing between classes.  
Confusion Matrices: the confusion matrices illustrate the true positives, false positives, true negatives, 
and false negatives for each model. SVM’s confusion matrix showed fewer misclassifications compared 
to k-NN in correspondence with predictions, particularly for classes that were previously challenging for 
k-NN.   
ROC Curves: the ROC curves and AUC scores further proved the SVM model's better performance. The 
SVM curve was closer to the top-left corner compared to k-NN, indicating better trade-offs between true 
positive rate and false positive rate. 

Model Comparison  
The accuracy scores for both models were computed. SVM outperformed k-NN with a higher accuracy. 
This suggests that SVM is more effective at classifying the wine types in this dataset. The higher 
performance of SVM can be attributed to its ability to create a more complex decision boundary through 
the use of hyperplanes in the feature space, whereas k-NN’s performance can be sensitive to the choice 
of k and the distance metric used.   

Interpretation   
Feature Importance: for SVM, feature importance was evaluated using the model’s coefficients.  
The analysis showed which features had the most significant impact on the decision boundary. Features 
with higher absolute coefficients were regarded as more important for classification. This insight helps in 
understanding which features are crucial in differentiating between wine classes.   

Challenges:   
• Model Complexity: SVM can be computationally expensive with larger datasets, although this 
wasn't an issue in this case due to the manageable size of the dataset.   
• k-NN Sensitivity: the performance of k-NN is highly dependent on the choice of k and can suffer 
from the curse of dimensionality if not properly tuned. 
Potential Improvements:   
• Hyperparameter Tuning: for both models, hyperparameter tuning using techniques such as grid 
search or cross-validation could further optimize performance. 
• Feature Engineering: additional feature engineering or dimensionality reduction techniques like 
PCA could improve model performance.  
• Model Diversity: exploring other classification algorithms, such as Random Forests or Gradient 
Boosting, could provide additional insights and potentially better performance.   
• Cross-validation: implementing cross-validation would offer a more robust evaluation of model 
performance compared to a single train-test split. 

Conclusion   
The SVM model demonstrated better performance compared to k-NN in classifying wine types in this 
dataset. Its ability to handle high-dimensional spaces and provide a clear decision boundary likely 
contributed to its superior performance. Feature importance analysis from SVM provided valuable 
insights into which features significantly impact classification. Improvement methods could focus on 
tuning hyperparameters, exploring alternative algorithms, and further improving preprocessing and 
feature engineering steps.
