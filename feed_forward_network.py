import numpy as np
import theano.tensor as T
import theano

class SoftmaxLayer(object):
    """
    The logistic regression class is described by two parameters (which
    we will want to learn). The first is a weight matrix. We'lll refer to
    this weight matrix as W. The second is a bias vector b. Refer to the
    text if you want to learn more about how this network works. Let's get
    started!
    """
    def __init__(self, input, input_dim, output_dim):
        """
        We first initialize the logistic network object with some important info
    
        PARAM input : theano.tensor.TensorType
        A symbolic variable that we'll use to represent one minibatch of our dataset
    
        PARAM input_dim : int
        This will represent the number of input neurons in our model
        Now as top layer, it will be the output of hidden layer(s) 
    
        PARAM output_dim : int
        This will represent the number of neurons in the output layer (i.e.
        the number of possible classification for the input) (e.g. 0-9)
        """
    
        # We initialize the weight matrix W of size (input_dim, output_dim)
        self.W = theano.shared(
            value=np.zeros((input_dim, output_dim)),
            name='W',
            borrow=True
        )
        
        # We initialize a bias vector for the neurons of the output layer
        self.b = theano.shared(
            value=np.zeros(output_dim),
            name='b',
            borrow=True
        )
        
        # Symbolic description of how to compute class membership probabilities
        self.output = T.nnet.softmax(T.dot(input, self.W) + self.b)
    
        # Symbolic description of final prediction
        self.predicted = T.argmax(self.output, axis=1)


class HiddenLayer(object):
    """
    The hidden layer class is described by two parameters (which
    we will want to learn). The first is an incoming weight matrix.
    We'll refer to this weight matrix as W. The second is a bias
    vector b. Refer to the text if you want to learn more about how
    this layer works. Let's get started!
    """
    
    def __init__(self, input, input_dim, output_dim, random_gen):
        """
        We first initialize the hidden layer object with some important
        information.
        
        PARAM input : theano.tensor.TensorType
        A symbolic variable that we'll use to describe incoming data from
        the previous layer.
        
        PARAM input_dim : int
        This will represent the number of neurons in the previous layer
        
        PARAM output_dim : int
        This will represent the number of neurons in the hidden layer
        
        PARAM random_gen : numpy.random.RandomState
        A random number generator used to properly initialize the weights.
        For a tanh activiation function, the literature suggests that the 
        incoming weights should be sampled from the uniform distribution
        [-sqrt(6./(input_dim + output_dim)), sqrt(6./(input_dim + output_dim))]
        """
        
        # We initialize the weight matrix W of size (input_dim, output_dim)
        self.W = theano.shared(
                value=np.asarray(
                        random_gen.uniform(
                                low=-np.sqrt(6. / (input_dim + output_dim)),
                                high=np.sqrt(6. / (input_dim + output_dim)),
                                size=(input_dim, output_dim)
                            ),
                        dtype=theano.config.floatX
                    ),
                name='W',
                borrow=True
            )
        # We initialize a bias vector for the neurons of the output layer
        self.b = theano.shared(
                value=np.zeros(output_dim),
                name='b',
                borrow=True
            )
        
        # Symbolic description of the incoming logits
        logit = T.dot(input, self.W) + self.b
        
        # Symbolic description of the outputs of the hidden layer neurons
        self.output = T.tanh(logit)
        
        
class FeedForwardNetwork(object):
    """
    Multi-Layer Perceptron Class
    
    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate Layers usually have as activation function tanh or the 
    sigmoid function (defined here by a 'HiddenLayer' class) while the
    top layer is a softmax Layer (defined here by a 'LogisticRegression' class)
    """
    
    def __init__(self, random_gen, input, input_dim, output_dim, hidden_layer_sizes):
        """
        :type random_gen: numpy.random.RandomState
        :param random_gem: a random number generator used to initialize weights
        
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the architecture
        (one minibatch)
        
        :type input_dim: int
        :param input_dim: number of input units, the dimension of the space in
        which the datapoints lie
        
        :type output_dim: int
        :param output_dim: number of output units, the dimension of the space in which
        the Labels lie
        
        :type hidden_layer_sizes: int
        :param hidden_layer_sizes: number of neurons in a hidden layer
        """
        
        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer tanh activation function connected to the
        # LogisticRegression Layer (Softmax layer for me); the activation function
        # can be replaced by sigmoid or any other nonlinear function
        self.hidden_layer = HiddenLayer(
            input=input,
            input_dim=input_dim,
            output_dim=output_dim,
            random_gen=random_gen
        )
        
        # The Logistic Regression Layer (or Softmax Layer gets as input the
        # hidden units of the hidden layer
        
        self.softmax_layer = SoftmaxLayer(
            input=self.hidden_layer.output,
            input_dim=hidden_layer_sizes,
            output_dim=output_dim
        )
            
        
    def feed_forward_network_cost(self, y, lambda_l2=0.0001):
        """
        Here we express the cost incurred by an example given the correct distribution
        
        PARAM y : theano.tensor.TensorType
        These are the correct answers, and we compute the cost with respect to this ground truth
        (over the entire minibatch). This means that y is of size (minibatch size, output_dim)
        
        PARAM lambda : float
        This is the L2 regularization parameter that we use to penalize large values
        for components of W, thus discouraging potential overfitting
        """
        
        # Calculate the log probabilities of the softmax output
        log_probabilities = T.log(self.softmax_layer.output)
    
        # We use these log probabilities to compute the negative log likelihood
        negative_log_likelihood = -T.mean(log_probabilities[T.arange(y.shape[0]), y])
    
        # Compute the L2 regularization component of the cost function
        l2_regularization = lambda_l2 * (
            (self.hidden_layer.W ** 2).sum() + (self.softmax_layer.W ** 2).sum()
            )
    
        # Return a symbolic description of the cost function
        return negative_log_likelihood + l2_regularization
        
    def error_rate(self, y):
        """
        Here we return the error rate of the model over a set of given labels
        (perhaps in a minibatch, in the validation set, or the test set)
    
        PARAM y : theano.tensor.TensorType
        These are the correct answers, and we compute the ost with
        respect to this ground truth (over entire minibatch). This
        means that y is of size (minibatch_size, output_dim)
        """
    
        # Make sure y of correct dimension
        assert y.ndim == self.softmax_layer.predicted.ndim
    
        # Make sure that ys contains values of correct data type (ints)
        assert y.dtype.startswith('int')
    
        # Return the error rate on the data
        return T.mean(T.neq(self.softmax_layer.predicted, y))
        