

### Components 
    - Parameter: real-valued vectors and matrices representing weight matrices
        and bias vectors. 
    - LookupParameters: sets of vectors of `Parameter`. 
    - Model: Collection of `Parameter` and `LookupParameter` objects. The model keeps track 
        of the parameters and their gradients. 
    - Trainer: implements an online update rule, such as SGD or Adam. The trainer holds a 
        pointer to the model object and, therefore, the parameters it contains. 
    - Expression: Main data type being manipulated in a Fortis program. Each expression
        represents a subcomputation in the computation graph. For instance, a `Parameter` 
        object can be added to the computation graph, resulting in an expression W or b. 

    - Operators: These are functions that act on expressions and return other expressions. 
        Crucially, they are not objects. Fortis defines many different operators, including
        addition, multiplication, softmax, tanh, etc. 
    - Builder classes: These define interfaces for building different networks. In our case, we
        will be only interested in implementing the transformer network, but one should not 
        have a hard time having a recurrent neural network builder, for instance. 
        These work on top of expressions and operators and provide easy-to-use libraries. 
        More discussion on builders below. 
    - 
