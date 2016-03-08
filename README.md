# CNN on Spark
This is a generalised CNN(convolutional neural network) model implemented on Spark-scala. The model is abstracted down to layer models which contains models like CL(Convolutional layer), SL(subsampling layer), FL(fully-connected layer) and OL(Output layer). You can build up your own network architecture according to your need.
<h5>package structure:</h5>
./model: successfully trained LeNet5 with 5 iteration training and tested on MNIST dataset. Error rate < 2%
./src: source code of the package:
----+CNNLayer
    ----+layer.scala: generalized layer model with useful LA utilities (with learning rate and momentum)
    ----+CL.scala: convolutional layer (sigmoid activation, easy to extend in tanh or arctan)
    ----+SL.scala: subsampling layer (mean sampling, easy to extend to max/min sampling)
    ----+FL.scala: fully connected layer
    ----+OL.scala: extends FL, for output (sigmoid activation, trying to extend in softmax)
----+CNNNet
    ----+CNN.scala: well encapsulated LeNet5 model
    ----+MLP.scala: well encapsulated multi-layer-perceptron model using CNNLayer
----+NNFM
    ----deprecated, stand alone experiment for Neural network in functional programming
----+NNOO
    ----deprecated, stand alone experiment for Neural Network in OO programming
----+TestCase
    ----+CNNClassifier
        ----+cnnBatchTrain.scala: batch training of cnn
        ----+cnnLocalTrain.scala: local model(single node) training of cnn
        ----+cnnclassify.scala: feedforward operation of cnn
        ----+cnntrain.scala: mixed training for cnn
    ----+CNNCorrectnesss
        ----+deprecated, for testing
    ----+mlp
	----+mlpclassify.scala
	----+mlptrain.scala

<h5>Use language:</h5>
Spark (scala api)<br>
<h5>Based on theory:</h5>
Convolutional neural network trained with back propagation algorithm using gradient descent (parameter: learning rate and momentum).
Global training setting: batch training size, training mode(update weight when classification is wrong or not?)
<h5>Supervised by:</h5>
Dr.Eric Lo, Hong Kong PolyU

