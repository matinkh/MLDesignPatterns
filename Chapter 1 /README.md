# Chapter 1: The Need for Machine Learning Design Patterns

**What are _Design Patterns_?** Each pattern describes a <span style="color:blue">problem</span> which occurs _many times_, and the core of the <span style="color:blue">solution</span> which can be reused. Having a <span style="color:blue">name</span> for a pattern saves the architect from having to continually rediscover a principle.

For each pattern, this book offers a description of an occurring problem and walks through a variety of potential solutions.

---

_Machine learning_ is a process of building models that learn from data. This is in contrast to traditional programming where we write explicit rules that tell programs how to behave.

---

## Data and Feature Engineering

Data is the hart of any machine learning problem.

* _Training data_, which is fed to the model during training process.
* _Validation data_, which is held out from training set and used to evaluate how the model is performing _after each training epoch_.
* _Test data_, which is not used in the training at all and is used to evaluate how the trained model performs.

Data can take many different forms. _Structured_ (numerical and categorical), or _unstructured_ (free-form text, images, video, and audio).

One important step is **data validation**, where we compute statistics on the data, and try to understand the schema. Furthermore, we want to evaluate the dataset to identify problems like drift and training-serving skew.

---

## The Machine Learning Process

There are two main processes: (i) _training_, where model learn to identify patterns from training data, and (ii) _prediction_, where we send new data to the model and make use of its output.

Often, the processes of collecting training data, feature engineering, training, and evaluating the model are handled seperately from the production piple line.

In other situations, we may have new data being ingested continuously and need to process them immediately before sending it to the model for training or prediction. This is known as _streaming_.

---

## Common Challenges

* **Data quality:** ML models are only as reliable as the data used to train them.
  * Data _accuracy_: how well the given data and labels are corresponding. It refers to the training data's features and the ground truth labels.
  * Data _completeness_: did the model see all the data/labels necessary to perform its job later?
  * Data _consistency_: all the feature values, sensor inputs, ... consistent in their measurement and preprocessings?
* **Reproducibility:** In traditional programming, the output of a program is reproducible and guaranteed. ML models, on the other hand, have an inherent element of randomness - model weights are initialized with random values. (We set the random seed value)
* **Data drift:** While ML models typically represent a static relationship between inputs and outputs, data can change significantly over time. Data drift refers to the challenge of ensuring our ML models stay relevant.
* **Scale:** As is the case with any software applications, we will likely encounter scaling challenges in data collection and preprocessing, training, and serving.
* **Multiple objectives:** multiple teams want the model to perform well in different areas.
  * You want your model to be more sensitive to <span style="color:darkred">false negatives</span>. Therefore, you optimize it for <span style="color:darkred">recall</span>.
  * Another team want your model to be more sensitive to <span style="color:darkgreen">false positives</span>. They optimize it for <span style="color:darkgreen">precision</span>.

---

**Summary:** Design patterns are a way to codify the knowledge and experience of experts into advice that all practitioners can follow.