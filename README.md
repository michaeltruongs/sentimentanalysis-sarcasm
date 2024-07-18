# nlp-boys

This is the nlp-boys implementation of the sarcasm classification model. We have implemented a modified version of the dataset and LSTM code from HW 3 for our model and training loop.

To train a model from start to finish, output the weights and best accuracy weights as 'sarcasmModel.pt' and 'sarcasmModelBest.pt', calculate the f1 score of the model from a test dataset, and output tweets and model predictions in a csv, run the main.ipynb notebook. Run this notebook from top to bottom, ignoring the cell for loading a pretrained model. There are 3 required files to train the model. 1) The glove word embeddings in 'glove.6B.50d.txt'. These embedding are used to represent the invididual tokens in each tweet. Replace the path in the glove_file variable with your path to the glove file. 2) The training data in 'train.En.csv'. We used the normal training data from the challenge to train our model. Replace the path in the input_file parameter in the cell where the train_dataset is created to your path the the training data. 3) The testing data in 'task_A_En_test.csv'. We used the normal testing data from the challenge to test our model. Replace the path in the test_file parameter when calling getEval() to your path to the testing data. 

To evaluate a model that has been already trained, calculating the f1 score of the model from a test dataset, and outputting tweets and model predictions in a csv, use the main.ipynb notebook. Run this notebook from top to bottom, skipping the cell where the 'model' variable is instantiated as a sarcasmModel() and instead running the cell where model_loaded is instantiated as a sarcasmModel(). Replace the path in the torch.load(PATH) call with the path to your saved model weights. NOTES: Before running the last cell which calls getEval(), replace the 'model' input with your loaded_model. The cells in which getMetrics() and train_sarcasm_classification() are unecessary to run when loading a pretrained model, though running them will have no impact on loading or evaluation. Make sure to update the testing data filepath to your path to your data as stated in '3' above.

getEval() both prints the f1 score of the model, and outputs both the tweets and the models predictions in a csv called 'output_file.csv'