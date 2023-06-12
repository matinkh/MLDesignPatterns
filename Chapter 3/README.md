# Chapter 3: Problem Representation Design Patterns

This chapter looks at different types of ML problems and analyzes how the model architecture vary depending on the problem.

* _[Reframing](#Design-Pattern-5-Reframing)_ design pattern takes a solution that is intuitively a regression problem and poses it as a classification problem (and vice versa).
* _[Multilabel](#Design-Pattern-6-Multilabel)_ design pattern handles the case that training examples can belong to more than one class.
* _[Ensemble](#Design-Pattern-7-Ensembles)_ design pattern solves a problem by training multiple models and aggregating their responses.
* _[Cascade](#Design-Pattern-8-Cascade)_ design pattern addresses situations where a ML problem can be broken into a series (or cascade) of ML problems.
* _[Neutral Class](#Design-Pattern-9-Neutral-Class)_ design pattern recommends approaches to handle highly skewed or imblalanced data.
* [Rebalancing](#Design-Pattern-10-Rebalancing) design pattern provide various ways to handle imbalance datasets.

---

## Design Pattern 5: Reframing

**Example: Rainfall Estimation**

Let's assume we are trying to train a model to estimate _rain fall_ given certain conditions. Since we are measuring rain fall, it is naturally seen as a regression problem. But we realize that our model is always off. We can calculate more features, train different models, etc. However, one way to improve the result is by _reframing_ it as a classification problem. I.e., model a discrete probability distribution.

![reframing01](img/reframing01.jpg)

Modeling a distribution this way is advantageous since rainfall does not exhibit the typical bell-shaped curve of a normal distribution. For other cases that we have _bimodal_ or even normal distribution but with high variance, this approach is suitable.



**Example: Video Recommendation**

Training a classifier to predict whether a user would click on a video title may prioritize click baits. Instead, we can reframe this problem as a regression problem of predicting the fraction of the video that will be watched. (**change of objective**)

---

When reframing a regression into a classification problem, we relax our prediction target to be instead a discrete probability distribution. We lose a little precision due to bucketing, but gain **expressiveness of a full prbability density function (PDF)**. Also, our model is more adept at learning a complex target than the more rigid regression model.

By looking at the output probability distribution we can **capture uncertainties**. The width of of the distribution tells us about the irreducible error.

* If it is very wide, that means we have a lot of variance.
* If it is very sharp and narrow, that means we do not have much error and perhaps a regression model can do just fine.

---

**Tradeoffs and alternatives**

* _Bucketized outputs_: the typical approach is to bucketize the output values and treat the problem as a multiclass classification problem. Therefore, we lose precision. This may or may not be acceptable.
* _Restricting the prediction range_: It may or may not be desired. For cases that we want to restrict the range of the prediction outputs, this approach, reframing a regression problem into a classification problem is beneficial.
* _Label bias_: in the "video recommendation" example we saw that predicting labels brings a bias with it; we may prioritize click baits. Using a regression model (estimate the watch time) resloves this issue.
* _Multitask learning_: instead of reframing a regression problem into a classification problem (or vice versa), we can do both! One model tries to optimize two different objective functions; therefore, we have _parameter sharing_ between two cases. This may be helpful, as previous cases show that it outpeforms two independent models; e.g., _predicting object types (classification prob) and the bounding boxes (regression)_.

---

## Design Pattern 6: Multilabel

_Multilabel_ classification means we can assign more than one label to a given training example. Not that this is different from _multiclass_ classification problems, where a signle example is assigned exactly one label from a group of many (>1) possible classes; if we could pick more than one label, then that would be _multilabel, multiclass classification_.

An example for _multilabel_ classification is "determining tags for Stack Overflow questions." Each question can be tagged with multiple labels; e.g., "Python", "pandas", and "visualization."

---

**Use sigmoid!**

The solution for building models that can assign more than one label to a given training example is to use the _sigmoid_ activation function in our final output layer. (Rather than generating an array where all values sum to 1 - _softmax_)

The number of units in the final layer is the same as the number of classes. Each output will be assigned a sigmoid value.

```python
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(3, activation='sigmoid')  # We can assign three labels to each example.
])
```

---

**Tradeoffs and alternatives**

* **Which loss function?** the book suggests using `binary_crossentropy` for binary and multilabel models. This is because a multilabel problem is essentially several binary classification problems.
* **How to parse sigmoid results?** for _softmax_ case, we could simply use `argmax` to find the class label. For multilabel with sigmoid values, the book suggests using a _probability threshold_. Any value greater than threshold should be considered as the predicted label.
* **Dataset considerations:** Building a balanced dataset is more nuanced for the _Multilabel design pattern_. There is usually a hierarchy to the predicted labels. We can (i) use a flat approach and have one layer with all labels; or (ii) use the _Cascade_ design patterhn; i.e., one model identifies higher-level labels, and more specific ones are assigned by other models.
* **One-versus-rest:** For _Multilabel_ classification we can train multiple binary classifiers instead on one multilabeled model. It's called _one versus rest_. Each binary classifier identifies whether or not to assign its label. The benefit is that we can use model architectures that are mostly suitable for binary classifications, such as SVM.

---

## Design Pattern 7: Ensembles

Errors in any ML model can be broken down into three parts: (i) high bias, (ii) high variance (both are _reducible errors_) and (iii) _irreducible errors_ (due to noise in data, framing of the problem, bad training examples, etc)

High bias is also referred to as _under-fitting_ problem. High variance is _overfitting_.

In practice, it's hard to lower bias and variance at the same time. Of course, this is the case for small and mid-size datasets.

**Solution:** by building several models with different inductive biases and aggregating their outputs, we hope to get a model with better performance.

* Bagging
* Boosting
* Stacking

---

**Bagging** (Bootstrap AGGregatING) - Addresses **high variance**.

The aggregation takes place on the output of the multiple ensemble models - average, majority vote, etc. Random Forest is one example of Bagging.

With bagging, the model and algorithms are the same. For example, with random forest, the submodels are all short decision trees.

<span style="color:darkgreen">Strengths:</span>

* On average, the ensemble model will perform at least as well as any of the individual models in the ensemble.
* It's even a recommended solution to fix high variance of neural networks.

<span style="color:darkred">Cons:</span>

* Bagging is typically less effective for more stable learners, such as kNN, Naive Bayes, linear models, or SVMs. That is because the size of the training set is reduced through bootstrapping.

---

**Boosting** - Address **high bias**.

Unlike bagging, boosting ultimately constructs an ensemble model with _more_ capacity than the individual member models.

The idea behind boosting is to iteratively build an ensemble of models where each successive model focuses on learning the examples the previous model got wrong. In short, boosting iteratively improves upon a sequence of weak learners taking a weighted average to ultimately yield a strong learner.

Examples: AdaBoost, Gradient Boosting Machines, XGBoost.

---

**Stacking**

Unlike bagging, the intial models are typically of different model types and are trained to completeion on the full dataset. Then, a secondray meta-model is trained using the initial model outputs as features. This second meta-model learns how to best combine the outcomes of the initial models to decrease the training error and can be any type of ML model.

---

**Tradeoffs and alternatives**

* **Increased training and design time:** _Is it best to reuse the same architecture or encourage diversity?_ _If we use different architectures, which ones should we use?_ ... (Always compare accuracy and resource usage against a linear or DNN model)
* **Dropout as bagging:** Dropout in neural network randomly turns off neurons of the network for each mini-batch, essentially evaluating a bagged ensemble of exponentially many neural networks. (It's not exactly bagging. First, models are not independent. Second, each member model would only be trained for a single training loop, not the respective training set)
* **Decreased model interpretability**
* **Choosing the right tool for the problem:** boosting for high bias. Bagging for high variance.

---

## Design Pattern 8: Cascade

The _Cascade_ design pattern addresses situations where a ML problem can be profitably broken into a series of ML problems.

Any ML problem where the output of the one model is an input to the following model or determines the selection of subsequent models is called a _Cascade_.

Say we need to predict a value during both usual and usual activity. The model will learn to ignore the unusual activity because it is rare. If the unusual activity is also associated with abnormal values, then trainability suffers.

---

For example, we would like to predict the likelihood that a customer will return an item that they have purchased. If we train a single model, the resellers' return behavior will be lost because there are millions of retail buyers, and only a few thousand resellers.

* Retail buyers usually return their item within a few weeks.
* Resellers return the item only if they cannot re-sell it. So, that would be several months.

One way to solve this problem is to overweight the reseller instances when training our model. This solution is suboptimal.

We do not want to trade off a lower accuracy on the retail buyers for a higher accuracy on the reseller use cases. It is necessary to get both types of returns as accurate as possible.

---

**How to Cascade?**

* Predict whether a data point is _usual_ or _unusual_.
* Train one model on the _usual_ cases.
* Train the second model on the _unusual_ cases.
* In production, combine the output of the three seperate models to predict the final outcome.

---

**Use the output of the first model; not actual labels**

At prediction time, we don't have true labels, just the output of the first classifier. And predictions have errors. So, the second and third models are required to make predictions on data that they might have never seen during training.

Wherever the output of a ML model needs to be fed as the input to another model, the second model needs to be trained on the predictions of the first model; not the actual labels.

---

**Tradeoffs and alternatives**

_Cascade_ is <u>NOT</u> necessarily a best practice. It adds complexity to your ML workflows, and may actually lower the performance. As much as possible, try to limit a pipeline to a single ML probel. Avoid having, as in the _Cascade_ pattern, multiple ML models in the same pipeline.

Splitting an ML problem is usually a bad idea. An ML model should learn combinations of multiple factors.

So, **when to use Cascade?**

* Cascade design pattern addresses an **unusual scenario** for which we do not have a categorical input, and for which extreme values need to be learned from multiple inputs.
  * Predicting rain falls using satellite images. For >99% of pixels, it won't rain. So, it's best to predict where it rains, and where it doesn't. And then, estimate the rain fall for positive cases.
* **Pre-trained models.** When we wish to reuse the output of a pre-trained model as an input into our model.
  * We are building a model to detect authorized entrants to a building. We wish to use an existing OCR to read license plates.
  * OCR systems will have errors. Therefore, we should train the model on the actual output of the OCR system.
  * It is also necessary to retrain the model if we change the OCR system.

---

## Design Pattern 9: Neutral Class

In many classification situations, creating a neutral class can be helpful. For example, we train a three-class classifier that outputs disjoint probabilities for Yes, No, and Maybe.

---

**Example: acetaminophen or ibrufen**

In our data, we see that 10% of of the patients are atrisk of liver damage. Thus, they were prescribed _ibrufen_ for painkiller. Another 10% had history of stomach ulcers. Therefore, they were prescribed _acetaminophen_. The rest were given either ibrufen or acetaminophen.

Our model should predict _either_ for those cases that are not in those categories.

---

**Example: apgar score**

At birth, newborns are assigned a _apgar score_ to indicate how healthy they are or if they require immediate medical attention. Scores of 9 and 10 are considered healthy.

An apgar score involves a number of subjective assessments, and whether a baby is assigned 8 or 9 often reduces to matters of physician preference. What if we create a neutral class to hold these "marginal" scores?

---

**Tradeoffs and alternatives**

Neutral class design pattern is useful for

* **When human experts disagree**. If it's not a clear signal from all our experts to label a data point, we can have a "_maybe_" class.
  * Another approach is to use the disagreement among human labelers as the weight of a data point. E.g., if experts are split 3 to 2, we give 0.6 weight to the data point, and allow our classifier to focus more on the "_sure_" cases.
* **Customer satisfaction**. If we attempt to train a binary classifier by thresholding at 6, the model will spend too much effort trying to get essentially neutral responses correct. It's better to categorize 1 to 4 as bad, 8 to 10 as good, and 5 to 7 as neutral.
* **Reframing with neutral class**. Imagine we want to have a model to predict if a stock goes up or down so we can decide wether to buy or sell. The end goal is to decide whether to purchase or sell. So, it's best to identify stocks that rise >5%, or drop >5% in the next six months for the decision, and consider the rest as neutral. (Instead of having a regression model to predict if a stock rises or drops)

---

## Design Pattern 10: Rebalancing

Many real-world problems are not so neatly balanced. For example, for _fraud detection_, we are building a model to identify fradulent credit card transactions. Such transactions are much rarer than regular transactions.

In regression cases, imbalanced datasets refer to data with outlier values that are either much higher or lower than the median in your dataset.

In cases of imbalanced dataset, accuracy values for model evaluation are misleading. Because always predicing the major case results in a high accuracy. It's important to **choose an appropriate evaluation metric**. Then, there are various techniques we can employ for handling inherently imbalanced datasets at both the dataset and model level; e.g., _downsampling_, _class weights_, etc

---

**Choose an evaluation metric**

For imbalanced datasets, it's best to use metrics like **precision**, **recall**, or **F-measure** to get a complete picture of how our model is performing.

* **Precision:** %TP out of all _predicted_ positives. ($\frac{TP}{TP + FP}$)
* **Recall:** %TP out of all _actual_ positives. ($\frac{TP}{TP+FN}$)
* **F-measure:** it's a metric that takes both precision and recall into account. ($2 \times \frac{precision \times recall}{precision + recall}$)

---

**Note:** When evaluating models trained on imbalanced datasets, we need to use _unsampled data_ when calculating success metrics. This means that no matter how we modify our dataset for training, <span style="color:blue">we should leave our test set as is</span> so that it provides an accurate representation of the original dataset.

---

**Changing dataset or model**

* **Dataset level**

  * _Downsampling_
    * we decrease the number of examples from the majority class used during model training.
    * it's usually combined with _Ensemble_ pattern.
      * `downsample the majority class` $\rightarrow$ `Train a model` $\rightarrow$ `add it to ensemble` $\rightarrow$ ..repeat..
      * During inference, we take the _median_ output of the ensemble models.
  * _Upsampling_
    * Overrepresent the minority class by both replicating some examples, and generating additional, synthetic examples.
    * One known model is SMOTE, which randomly generates a new minority class example between data points.
    * Similar logic can be applied to image datasets. `Keras.ImageDataGenerator` class does data augmentation - variations of the same image by _rotating_, _cropping_, _adjusting brightness_, and more.

* **Model level:**

  * _Weighting_

    * we can change the _weight_ our model gives to examples from each class. (different from "weights" learned by model during training). By weighting _classes_, we tell our model to treat specific classes with more importance during training.
    * the class weight values should be related to the balance of each class in our dataset.

    ```python
    minority_class_weight = 1 / (num_minority_examples/total_examples)/2
    majority_class_weight = 1 / (num_majority_examples/total_examples)/2
    ```

    * In `Keras` we can pass `class_weights` dict to our model when we train it with `fit()`.

    ```python
    class_weights = {0: majority_class_weight, 1: minority_class_weight}
    model.fit(train_data, train_labe, class_weights)
    ```

---

**Tradeoffs and alternatives**

* **Number of minority class examples available:** If the number of examples are too low, downsampling may make the resulting dataset too small for a model to learn from.
* **Upsampling for text:** generating synthetic data is less straightforward for text data. It's best to rely on downsampling and weighted classes.
* **Combining different techniques:** Downsampling and class weight techniques can be combined for optimal result. We start by downsampling our data until we find a balance that works for our use case. Then, based on the label ratios for the rebalanced dataset, use class weight technique.
  * Downsampling is also often combined with the _Ensemble_ design pattern.
    * Say we have 100 of the minority class and 1000 of the majority class.
    * We split the majority class in 10 sets and create 10 {100+100} balanced datasets.
    * We then train 10 classifiers and use bagging technique to aggregate their outputs.
* **Importance of explainability:** _attribution value_ tells us how much each feature in our model influenced the model's prediction. When dealing with imbalanced data, it's important to look beyond our model's accuracy and error metrics to verify that it's picking up on meaningful signals in our data.
