# Problem 1 report

## Problem 1.1

**Used LLM** : Chat-GPT

**Code** :  
_Write down your code from First LLM_
```python
def learn_mul(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    ################# YOUR CODE COMES HERE ######################
    # training and return the multi-class logistic model
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    lr.fit(X, y)
    #############################################################
    return lr

def inference_mul(x: np.ndarray, lr_model: LogisticRegression) -> np.ndarray:
    ################# YOUR CODE COMES HERE ######################
    # inference model and return predicted y values
    y_pred = lr_model.predict(x)
    #############################################################
    return y_pred
```


**Code Description** :  
_Write down the description of the code from First LLM_   
Chat-GPT produced code by utilizing the ```LogisticRegression``` class. It passed argument 'multinomal' to ```multi_class```, 'lbfgs' to ```solver```, making it use Limited-memory Broyden–Fletcher–Goldfarb–Shanno Algorithm, and set 1000 max iterations. As for the reason for each arguement, it explained as below. 

>multi_class='multinomial': Ensures proper softmax-based logistic regression for multi-class problems.
>
>solver='lbfgs': Works well with multinomial loss and handles multi-class by default.
>
>max_iter=1000: Allows enough iterations for convergence

The code was given in a simple way, with no comments in-line. Chat-GPT seemed to give the shortest code that works for the problem.

As for the result, it produced accuracy of 0.9214. 


---

**Used LLM** : Claude AI

**Code** :  
_Write down your code from Second LLM_
```python
def learn_mul(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    ################# YOUR CODE COMES HERE ######################
    # training and return the multi-class logistic model
    
    # Initialize the logistic regression model
    # Using 'multinomial' solver for multi-class problems
    # 'lbfgs' is an optimization algorithm that works well for multi-class logistic regression
    lr = LogisticRegression(
        multi_class='multinomial',  # Use multinomial for true multi-class
        solver='lbfgs',             # Efficient optimizer for multinomial LR
        max_iter=1000,              # Increase max iterations for convergence
        C=1.0,                      # Regularization strength (inverse)
        random_state=42             # For reproducibility
    )
    
    # Train the model on the input data
    lr.fit(X, y)
    #############################################################
    return lr

def inference_mul(x: np.ndarray, lr_model: LogisticRegression) -> np.ndarray:
    ################# YOUR CODE COMES HERE ######################
    # inference model and return predicted y values
    
    # Use the trained model to predict class labels
    y_pred = lr_model.predict(x)
    
    # Alternative: If you need probabilities instead of class labels
    # y_pred_proba = lr_model.predict_proba(x)
    
    #############################################################
    return y_pred
```


**Code Description** :  
_Write down the description of the code from Second LLM with a focus on differences from first LLM_
Claude AI produced code by utilizing the ```LogisticRegression``` class. It passed the same arguments as Chat-GPT, bu additionally, it passed 1.0 to ```C```, and 42 to ```random_state```. As for the reason for each arguement, it explained as below. 

>Configured the model with the 'multinomial' option for true multi-class classification
>
>Used the 'lbfgs' solver, which is efficient for multinomial logistic regression
>
>Set a higher max_iter value (1000) to ensure convergence
>
>Added regularization parameter C and random_state for reproducibility
>
>For the inference function (inference_mul):
>
>Used the trained model's predict method to obtain class predictions

The code was a bit longer than that of Chat-GPT, especially with comments to explain what each line does. The explanation was more detailed than that of Chat-GPT, including explanation for each function. It even produced alternative codes for those who are trying to get probabilities instead of class labels. Though it is not actually included in the submitting code, Claude AI provided import statement code in case ```Logistic Regression``` class was not imported above. It also seemed to utilize as many parameters it can to make the best performance.

As for the result, it produced accuracy of 0.9214, the exact same with Chat-GPT. 


## Problem 1.2

**Used LLM** : Chat-GPT

**Code** :  
_Write down your code from First LLM_
``` python
def learn_mul2bin(X: np.ndarray, y: np.ndarray, num_classes: int) -> list[LogisticRegression]:
    lrs = []
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])
    for i in range(num_classes):
        print('training %s classifier'%(ordinal(i+1)))
        ################# YOUR CODE COMES HERE ######################
        # training and return the multi-class logistic model

        #############################################################
        
        # Create binary labels for class i vs rest
        binary_y = (y == i).astype(int)
        
        # Train binary classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, binary_y)
        
        # Save the trained model
        lrs.append(clf)

    return lrs

def inference_mul2bin(X: np.ndarray, lrs: list[LogisticRegression]) -> int:
    ################# YOUR CODE COMES HERE ######################
    # inference model and return predicted y values

    #############################################################
    
    # X is a single sample, shape (n_features,)
    # Predict probability for each classifier
    probs = [model.predict_proba(X.reshape(1, -1))[0, 1] for model in lrs]

    # Return class with highest probability
    y_pred = np.argmax(probs)
    
    return y_pred
```


**Code Description** :  
_Write down the description of the code from First LLM_   
Chat-GPT produced code by utilizing the ```LogisticRegression``` class. It passed a single argument to set 1000 max iterations. It did not give additional text to explain the resulted code, but in-line comments were included. To train the logisistic regression model, the y-data was made by converting the boolean result of ```y==i``` to integer.

The code was given in a simple way. Chat-GPT seemed to give the shortest code that works for the problem, using single-lined code when calculating probability in ```inference_mul2bin```. The highest probability was calculated using ```np.argmax()```.

As for the result, it produced accuracy of 0.9168.

---

**Used LLM** : Claude AI

**Code** :  
_Write down your code from Second LLM_
```python
def learn_mul2bin(X: np.ndarray, y: np.ndarray, num_classes: int) -> list[LogisticRegression]:
    lrs = []
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])
    for i in range(num_classes):
        print('training %s classifier'%(ordinal(i+1)))
        ################# YOUR CODE COMES HERE ######################
        # training and return the multi-class logistic model

        #############################################################
        
        # training and return the multi-class logistic model
        from sklearn.linear_model import LogisticRegression
        
        # Create binary labels for class i (1 for class i, 0 for all other classes)
        y_binary = (y == i).astype(int)
        
        # Create and train a binary logistic regression model
        lr = LogisticRegression(
            solver='liblinear',  # Efficient for binary classification
            C=1.0,               # Regularization strength (inverse)
            max_iter=1000,       # Increase iterations for better convergence
            random_state=42      # For reproducibility
        )
        
        # Train the model
        lr.fit(X, y_binary)
        
        # Add the trained model to our list
        lrs.append(lr)

    return lrs

def inference_mul2bin(X: np.ndarray, lrs: list[LogisticRegression]) -> int:
    ################# YOUR CODE COMES HERE ######################
    # inference model and return predicted y values

    #############################################################
    
    # X is a single sample in this case (from the usage context)
    # Reshape it to ensure it's 2D for sklearn (1 sample, n_features)
    if X.ndim == 1:
        X = X.reshape(1, -1)
        
    # Get probability scores for each binary classifier
    scores = []
    for lr in lrs:
        # Get probability of the positive class (class 1)
        # predict_proba returns [[prob_class0, prob_class1]]
        score = lr.predict_proba(X)[0, 1]
        scores.append(score)
    
    # Return the class with the highest probability
    y_pred = np.argmax(scores)
    
    return y_pred
```



**Code Description** :  
_Write down the description of the code from Second LLM with a focus on differences from first LLM_   
laude AI produced code by utilizing the ```LogisticRegression``` class. It passed the same arguments as Chat-GPT, bu additionally, it passed argument 'liblinear' to ```solver```, 1.0 to ```C```, and 42 to ```random_state```. Considering 'lbfs' is the default argument of ```solver```, Claude AI chose a different strategy for solving algorithm. It explained that 'liblinear' solver which works well for binary classification. However running time was much longer than that of code made by Chat-GPT (almost doubled).

The code was a bit longer than that of Chat-GPT. Rather than making single-lined codes, it seemed to make more efficient code, like reshaping ```X``` only once using ```if``` statement. It also seemed to utilize as many parameters it can to make the best performance.

As for the result, it produced accuracy of 0.9177, which is actually better than Chat-GPT. 
