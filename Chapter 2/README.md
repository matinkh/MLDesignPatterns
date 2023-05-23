# Chapter 2: Data Representation Design Patterns

At the heart of any ML model is a mathematical function that operates on specific data types. Real-world data may not be directly pluggable into the mathematical function. Therefore, we need _data representations_.

The process of creating features to represent the input data is called _feature engineering_, and so we can think of feature engineering as a way of selecting the data representation.

The process of learning features to represent the input data is called _feature extraction_, and we can think of learnable data representations (like embeddings) as automatically engineered features.

The data representation doesn't need to be learned or fixed - a hybrid is also possible.

We may represent input data of different types separately, or represent each piece of data as just one feature - _multimodal input_.

There are four data representations:

* Simple Data Representation: eg _scaling_, _normalizing_, etc
* Hashed Feature
* 



## Simple Data Representations

Not a feature representation design pattern, but a common practice in ML models.

### Numerical Inputs

For numerical values, we often scale them to take values in [-1, 1]. Why?

* ML frameworks use optimizers that are tuned to work well with numbers in this range. Thus, it improves the accuracy of the model.
* Some ML algorithms are sensitive to relative magnitude of features (eg K-Means).
* It also improves L1 and L2 regularization; we ensure that there is not much difference between variables.

What are different types of scaling?

* **Linear scaling**
  * _Min-max scaling_
  * _Clipping_
  * _Z-score normalization_
  * _Winsorizing_ (clip data outside 10 and 90 percentiles)
* **Nonlinear transformations**
  * If data is skewed or is not uniformly distributed or is not distributed like Gaussian...
  * We apply _nonlinear transformation_ before scaling (to make them look like a bell shape)
    * Custom functions: e.g. _log_ $\rightarrow$ _fourth root_ $\rightarrow$ ...
    * _Bucketize_: so bucket boundaries fit the desired distribution
    * _Box-Cox transform_: this method chooses its single parameter ($\lambda$) to control the "heteroscedasticity", so that the variance no longer depends on the magnitude.

### Categorical Inputs

Most ML models perform on numerical values. Thus, we need to transform our categorical data into numbers.

* **One-hot encoding:** converts a categorical feature into a vector of size _vocabulary_

  (eg `"English"` $\rightarrow$ `[0, 0, 1, 0, ..., 0]`)

* **Array:** if the array of categories is of fixed length, we can treat each _array position_ as a feature.

---

## Design Pattern: Hashed Feature

