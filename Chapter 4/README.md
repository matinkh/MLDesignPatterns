# Chapter 4: Model Training Patterns

ML models are usually trained iteratively, and this iterative process is informally called the _training loop_. In this chapter, we talk about the typical training loop and design patterns which are suitable for cases we have to deviate from the typical route. Namely,

* **[Useful Overfitting](#Design-Pattern-11-Useful-Overfitting):** we forgo the use of a validation or testing dataset, because we want to intentionally overfit on the training dataset.
* **[Checkpoints](#Design-Pattern-12-Checkpoints):** we store the full state of the model periodically, so we have access to partially trained models.
* **[Transfer Learning](#Design-Pattern-13-Transfer-Learning):** we take part of a previously trained model, freeze the weights, and incorporate these nontrainalbe layers intoa new model that solvess the same problem, but on a smaller dataset.
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

The goal of an ML model is to generalize and make reliable predictions on new, unseen data. If the model _overfits_ the training data, its ability to generalize suffers.

However, in certain situations _overfitting_ may be required.

Consider a situation of simulating the behavior of physical or dynamical systems (as in computational biology or computational finance), where the time dependence of observations can be described by a mathematical function or set of partial differential equations (PDEs). Although the equations can be formally expressed, they don't have a closed-form solution. Classical numerical methods have been developed to approximate solutions to these systems. However, they can be too slow to be used in practice.

ML models can serve as lookup tables of inputs to outputs. In these scenarios **there is no "unseen" data** that needs to be generalized. Thus, all possible inputs are tabulated and that is not considered as overfitting.

In short, we want the model to memorize the output of a function for all possible inputs.

---

## Design Pattern 12: Checkpoints

As model sizes increase, the time it takes to fit one batch of examples also increases. When there's a problem, e.g., the model starts overfitting the data, or a hardware failure, etc, we would like to be able to resume from an intermediate point, instead of from the very beginning.

At the end of each epoch, we can save the _model state_.

What's the difference between _exported model_ and a _model state_? We export the model for deployment. An _exported model_ does not contain the entire model state, just the information necessary to create the _prediction_ function. A _model state_, on the other hand, must know the epoch number, batch number, learning rate, and in case of RNNs, a history of previous input values. In general, the full model state can be many times the size of the exported model.

Saving the full model state so that model training can resume from a point is called _checkpointing_.

To **checkpoint a model in Keras**, provide a callback to the `fit()` method:

```python
checkpoint_path = '{}/checkpoints/'.format(OUTDIR)
checkpoint_cb = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=False,
                                                verbose=1)
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=3,
                    validation_data=(X_valid, y_valid),
                    verbose=2,
                    callbacks=[checkpoint_cb])
```

The availability of partially trained models opens up a number of other usecase. This is because the partially trained models are usually more generalizable.

**Tradeoffs and alternatives**

* **Early stopping**. The longer you train, the lower the loss on the training dataset. If you are starting to overfit to the training dataset, the validation error might start to increase. In such cases, it can be helpful to look at the validation error at the end of every epoch and stop the training process when the validation error is more than that of the previous epoch.
* **Checkpoint selection**. It is not uncommon for validation error to increase a bit and then start to drop again. Thus, the book recommends training longer and choosing the optimal run as a postprocessing step.
* **Regularization**. If we apply L2 regularization, the validation error does not increase. Instead, both the training loss and validation error should plateau. It's called a _well-behaved_ training loop. Applying regularization may be better than _early stopping_, because it allows you to use the entire dataset to change the wieghts of the model, whereas early stopping requires you to waste 10%-20% of your dataset.
  * Furthermore, recent research indicates that _double descent_ happens in a variety of ML problems, and therefore, it is better to train longer rather than risk a suboptimal solution by stopping early.
* **Fine-tuning**. Imagine that you need to priodically retrain the model on fresh data. You typically want to emphasize the fresh data, not the corner cases from last month. You are often better off resuming your training some epochs earlier than the last checkpoint.
* **Redifining an epoch**. If a dataset fit in memory, we can define epochs to go over that dataset multiple times.

```python
model.fit(X_train, y_train,
          batch_size=32,
          epochs=15)
```

​		However, not all data fit in memory. To make the ocde more resilient, we supply a TensorFlow dataset - an out-of-memory dataset. It provides iteration capability and lazy loading. The code becomes

```python
checkpoint_cb = keras.callbacks.ModelCheckpoint(...)
history = model.fit(train_ds,
                    batch_size=32
                    epochs=15,
                    validation_data=valid_ds,
                    callbacks=[checkpoint_cb])
```

​		However, **using epochs on large datasets remains a bad idea**.

​		- Dataset grow over time. If you get 100K more examples and get a higher error, is it because you need to do an early stopping, or is the new data corrupt?

​		- You checkpoint once per epoch, and waiting one million examples between checkpoints might be way too long! For resilience, we may want to checkpoint more often.

​		**Steps per epoch**. Instead of trainig for 15 epochs, we may decide to train for 143,000 steps.

```python
NUM_STEPS = 143000
NUM_CHECKPOINTS = 15
history = model.fit(train_ds,
                    batch_size=32,
                    epochs=NUM_STEPS // NUM_CHECKPOINTS,
                    validation_data=valid_ds,
                    callbacks=[checkpoint_cb])
```

​		Each step involves weight updates based on a single mini-batch of data, and this gives us much more granularity; e.g., stopping at 14.3 epochs.

**Note:** It works as long as we make sure to repeat the `train_ds` infinitely, otherwise, it will exit after the first epoch.

```python
train_ds = train_ds.repeat()
```

* **Retraining with more data**. Say we get 100,000 more examples. We add it to our data warehouse but do not update the coee. Our code will still want to process the number of steps we specified, and it will get to process that much data, except that 10% of the examples are newer.

---

## Design Pattern 13: Transfer Learning

TBA