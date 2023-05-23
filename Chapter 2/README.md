# Chapter 2: Data Representation Design Patterns

At the heart of any ML model is a mathematical function that operates on specific data types. Real-world data may not be directly pluggable into the mathematical function. Therefore, we need _data representations_.

The process of creating features to represent the input data is called _feature engineering_, and so we can think of feature engineering as a way of selecting the data representation.

The process of learning features to represent the input data is called _feature extraction_, and we can think of learnable data representations (like embeddings) as automatically engineered features.

The data representation doesn't need to be learned or fixed - a hybrid is also possible.

We may represent input data of different types separately, or represent each piece of data as just one feature - _multimodal input_.

There are four data representations:

* [Simple Data Representation](#simple-data-representations): eg _scaling_, _normalizing_, etc
* Design Pattern 1: [Hashed Feature](#design-pattern-1-hashed-feature)
* Design Pattern 2: Embeddings



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

## Design Pattern 1: Hashed Feature

There are certain problems with **categorical features**. Namely,

* <span style="color:darkred">_incomplete vocabulary_</span>: training data does not contain all the possible values.
* <span style="color:darkred">_high cardinality_</span>: a feature vector may have a length of thousands to millions.
* <span style="color:darkred">_cold start_</span>: after the model is placed into production, new data is introduced.

**Hashed Feature** design pattern represents categorical variables by:

1. Converting the categorical input into a unique string.
2. Applying a deterministic hashing algorithm on the string.
3. Taking the remainder of hash result divided by the desired number of buckets.

It's easy to see that all three above-mentioned issues are addressed.

---

**Tradeoffs and alternatives**

- <span style="color:darkred">_Bucket collision_</span>: different values may share the same bucket.
- <span style="color:darkred">_Skew_</span>: same bucket but vastly different? (eg. `Chicago airport` in the same bucket as `Vermont airport`)
- <span style="color:darkgreen">_Aggregate features_</span>: it may be helpful to add an aggregate feature so the difference between different-values-placed-in-the-same-bucket is preserved. (eg. `number_of_flights`)
- <span style="color:darkgreen">_Hyperparameter tuning_</span>: to find the best number of buckets.
- <span style="color:darkred">_Cryptographic hash_</span>: not reproducible. That's why we use _fingerprint_ hashing.
- <span style="color:darkgreen">_Order of operations_</span>: The order of operations (eg. `ABS(MOD(FINGERPRINT(value), num_buckets)))`) is important for reproducibility.
- <span style="color:darkred">_Empty hash buckets_</span>: It would be useful to apply L2 regularization to lower the weights associated with an empty bucket to near zero.

---

## Design Pattern 2: Embeddings

