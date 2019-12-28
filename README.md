
'''
****************************************************************
*	Title: " Kaggle Competition to Classify Reddit Posts"

*	Category: Machine Learning

*	Authors: Ryan Mokarian, Maysam Mokarian
****************************************************************
	Description:
	This Project contains a text classifier implemented using Naive Bayes algorithm. It uses Laplace Smoothing to deal with unseen data.
'''


## Run the Project

Follow the below steps to set up your virtual environment and run the project:

1) Make sure that you have installed on your local machine:

    ```pip install virtualenv```
2) Navigate to project directory (scripts/autosuggest/api) and create a virtual environment:

    ```virtualenv venv```
3) Activate the virtual environment:

   - on Mac/Linux: ```source venv/bin/activate``` 
   - on Windows: ```venv/src/activate```
4) Install the project requirements on the virtual environment (located here: autosuggest/requirements.txt):

    ```pip install -r requirements.txt```
    
5) run the `naive_bayes_laplace_smoothing.py` file 

     ```python naive_bayes_laplace_smoothing.py```
     
6) this script generates a `submission.csv` file of the predicted classes
