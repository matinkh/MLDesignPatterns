# Chapter 4: Model Training Patterns

ML models are usually trained iteratively, and this iterative process is informally called the _training loop_. In this chapter, we talk about the typical training loop and design patterns which are suitable for cases we have to deviate from the typical route. Namely,

* **[Useful Overfitting](#Design-Pattern-11-Useful-Overfitting):** we forgo the use of a validation or testing dataset, because we want to intentionally overfit on the training dataset.
* **Checkpoints:** we store the full state of the model periodically, so we have access to partially trained models.
* **Transfer Learning:** we take part of a previously trained model, freeze the weights, and incorporate these nontrainalbe layers intoa new model that solvess the same problem, but on a smaller dataset.
* **Distribution Strategy:** the training loop is carried out at scale over multiple workers.
* **Hyperparameter Tuning:** the training loop itself is inserted into an optimization method to find the optimal set of model hyperparameters.

## Typical Training Loop

The most common approach to determining the parameters of ML models is _gradient descent_. On large datasets, gradient descent is applied to mini-batches of the input; that is called _stochastic gradient descent_ (SGD).

Because SGD requires training to take place iteratively on small batches of the training dataset, training an ML model happens in a loop. SGD finds a minimum, but is not a closed-form solution, and so we have to detect whether the model convergence has happend. _Overfitting_ is yet another challenge that we need to detect during training. Therefore, the error (or _loss_) on the training and validation sets have to be monitored.

<img src="img/typical_trainingloop.jpg" alt="typical_trainingloop" style="zoom:50%;" />

The typical training loop in Keras looks like this

```python
model = keras.Mode(...)
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
history = model.fit(X_train, y_train,
                   batch_size=32,
                   epochs=5,
                   validation_data=(X_valid, y_valid))
results = model.evaluate(X_test, y_test, batch_size=64)
model.save(...)
```

---

## Design Pattern 11: Useful Overfitting

TBA