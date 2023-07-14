# Design Patterns for Resilient Serving

When a software application is deployed into production, it is expected to be resilient and require little human effort to keep it running. What's covered in this chapter

* The **[Stateless Serving Function](#Design-Pattern-16-Stateless-Serving-Function)** design pattern allows the serving infrastructure to scale and handle thousands/millions of prediction requests per second.
* The **[Batch Serving](#Design-Pattern-17-Batch-Serving)** design pattern allows the serving infrastructure to asynchronously handle occasional or preiodic requests for millions of predictions.
* The **Continued Model Evaluation** design pattern handles the common problem of detecting when a deployed model is no longer fit-for-purpose.
* The **Two-Phase Prediction** design pattern provides a way to address the problem of keeping models sophisticated and performant when they have to be deployed onto distributed devices.
* The **Keyed Prediction** design pattern is a necessity to scalably implement several of the design patterns discussed in this chapter.

---

## Design Pattern 16: Stateless Serving Function

The _Stateless Serving Function_ design pattern makes it possible for a production ML system to synchronously handle thousdands of prediction requests per second.

---

**Stateless Functions**

A stateless function is a function whose outputs are determined purely by its inputs.

Because stateless components don't have any state, the can be shared by multiple clients. On the other hand, _stateful components_ need to represent each client's conversational state.

Stateless compoenents are highly scalable, whereas stateful components are expensive and difficult to manage. When designing enterprise applications, architects are careful to minimize the number of stateful components.

We can turn a _stateful_ component into a _stateless_ one by having the client to maintain the state, or provide a longer sequence. However it makes the client code more complicated.

* E.g., a spelling correction model that takes a word and returns the corrected form will need to be stateful because it has to know the previous few words in order to correct a word "there" to "their" depending on the context.

---

**Problem with `predict()`**

Let's say we have a text classification model in Keras, and we want to use it to predict the sentiment behind user reviews.

* We have to lead the enitre model into memory. Deep learning models with many layers can be quite large.
* It imposees limits on the latency hat can be achieved because calls to the `predict()` method have to be sent one by one.
* It forces the developers to use Python to be able to call a Keras model's `predict()`.
* The model input and output that is most effective for training may not be user friendly.
  * E.g., the model output is logits because it is better for gradient descent. However, the cliants want sigmoid, so the output range is 0 to 1 and can be interpreted as a probability.

**Solution**

* Export the model into a format that captures the mathematical core of the model and is programming language agnostic.

  * In Keras, `model.save('export/mymodel')` saves the core of the model. The entire model state (learning_rate, dropout, etc) doesn't need to be saved.

* In the production system, the formula consisting of the "forward" calculations of the model is restored as a stateless function.

  ```python
  serving_fn = keras.models.load_model('export/mymodel')
  outputs = serving_fn(..inputs..)
  user_friendly_outputs = formatOutputs(outputs)
  ```

* The stateless function is deployed into a framework that provides REST end-point.

  * Either a web application, or serverless frameworks like Google App Engine, AWS Lambda, Azuer Functions, and Google Cloud Functions.

**Why it works?**

The approach of exporting a model to a stateless function and deploying the stateless function in a webl application framework works because web application frameworks offer autoscaling, can be fully managed, and are language neutral. (Every modern programming language can speak REST)

* Most modern enterprise machine learning frameworks come with a serving subsystem. For example, "TensorFlow Serving" and "TorchServe".

**Prediction Library**

Instead of deploying the serving function as a microservice that can be invoked via REST API, it is possible to impelement the prediction code as a library function. (Because of performance, or physical reasons).

The main drawback of the library approach is that maintenance and updates of the model are difficult. The secondary drawback is that the library approach is restricted to programming languages.

---

## Design Pattern 17: Batch Serving

TBA