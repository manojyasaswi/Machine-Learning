This file contains the information about how to make changes and run the files train.py and test.py
I.	train.py:
All the libraries used in this file are listed below.
•	random
•	troch
•	torch.nn
•	torch.nn.functional
•	Image from PIL
•	torch.optim
•	numpy
•	time
•	copy
•	pickle
•	matplotlib.pyplot
The file contains 2 layer neural network model. The class containing neural network model also contains the training function, validation function and some other plotting functions to visualize the data.
Use the function ‘data_curation()’ to generate a list of tuples which has paired the data with its respective class label.
The function ‘training_the_model()’ traines the model for the number epochs set by user or when the weights converge. It returns the optimal weights and parameters for the model.
The function ‘testing_the_data()’ returns the accuracy of the model when tested on validation data. 
All the parameters that can be modified are defined as lists with several values under the ‘main’ function. The parameters of the neural network class that are given as inputs are listed below.
•	lr - Learning rate of the model (float value).
•	num_of_neurons – tuple (for two-layer model) or int (for one-layer model) based on the model for ‘easy test’ or ‘hard test’.
•	train_data – data using which the model is to be trained (input must be a tuple with data zipped to its tensor).
•	Test_data – data for validation of the model (input must be a tuple with data zipped to its tensor).
•	Type_of_activation – activation function you want to use
Input: string 	‘Relu’ – relu activation function.
		‘tanH’- tanhactivation function.
		‘Sigmoid’ – sigmoid activation function.
		‘Linear’- linear activation function.
•	Optimizer – type of optimizer you want to use.
Input: string	‘Adam’ – Adam optimizer.
		‘SGD’ – Standard Gradient Descent optimizer
# if using ‘SGD’ provide the ‘moment’ argument.
•	epochs – number of epochs to before stopping the training of model (int value as input).
•	batch_size – size of each batch (int value as input).
•	num_classes – number of classes to classify (int value as input).
Adding extra layers in the neural network cannot be done by passing an argument, needs to be coded in the 


II.	test.py
All libraries used in this file are listed below.
•	random
•	troch
•	torch.nn
•	torch.nn.functional
•	Image from PIL
•	torch.optim
•	numpy
•	time
•	copy
•	pickle
•	matplotlib.pyplot
The file contains both 2 layer and one layer neural network models one for ‘hard test’ and the other for ‘easy test’ respectively. The class containing neural network model also contains the training function, validation function and some other plotting functions to visualize the data.
All the parameters that can be modified are defined as lists with several values under the ‘main’ function. The parameters of the neural network class that are given as inputs are listed below.
•	lr - Learning rate of the model (float value).
•	num_of_neurons – tuple (for two-layer model) or int (for one-layer model) based on the model for ‘easy test’ or ‘hard test’.
•	train_data – data using which the model is to be trained (input must be a tuple with data zipped to its tensor).
•	Test_data – data for validation of the model (input must be a tuple with data zipped to its tensor).
•	Type_of_activation – activation function you want to use
	Input: string 	‘Relu’ – relu activation function.
			‘tanH’- tanhactivation function.
			‘Sigmoid’ – sigmoid activation function.
			‘Linear’- linear activation function.
•	Optimizer – type of optimizer you want to use.
	Input: string	‘Adam’ – Adam optimizer.
			‘SGD’ – Standard Gradient Descent optimizer
# if using ‘SGD’ provide the ‘moment’ argument.
•	epochs – number of epochs to before stopping the training of model (int value as input).
•	batch_size – size of each batch (int value as input).
•	num_classes – number of classes to classify (int value as input).
The above mentioned parameters are set to their optimal values as observed from the training file for both single layered and two layered neural network.
Use the function ‘data_curation()’ to generate a list of tuples which has paired the data with its respective class label.
The function ‘testing_the_data()’ returns a tuple containing the accuracy of the model when tested on test data and the tensor showing the predicted class labels. 





		


	
