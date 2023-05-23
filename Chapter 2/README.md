# Chapter 2: Data Representation Design Patterns

At the heart of any ML model is a mathematical function that operates on specific data types. Real-world data may not be directly pluggable into the mathematical function. Therefore, we need _data representations_.

The process of creating features to represent the input data is called _feature engineering_, and so we can think of feature engineering as a way of selecting the data representation.

The process of learning features to represent the input data is called _feature extraction_, and we can think of learnable data representations (like embeddings) as automatically engineered features.

The data representation doesn't need to be learned or fixed - a hybrid is also possible.

We may represent input data of different types separately, or represent each piece of data as just one feature - _multimodal input_.

There are four data representations:

* Simple Data Representation: eg _scaling_, _normalizing_, etc
* 