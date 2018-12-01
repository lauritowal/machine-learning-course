function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Forward Propagation
eye_matrix = eye(num_labels);
Y_Matrix = eye_matrix(y,:);

a1 = [ones(m, 1) X]; 
z2 = a1*Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
z3 = a2*Theta2';
a3 = sigmoid(z3);
H = a3;

% Recall that for linear and logistic regression, 'y' and 'h' were both vectors, so we could compute the sum of their products easily using vector multiplication. 
% After transposing one of the vectors, we get a result of size (1 x m) * (m x 1). That's a scalar value. So that worked fine, as long as 'y' and 'h' are vectors.
% But the when 'h' and 'y' are matrices, the same trick does not work as easily. --> use element wise multiplication instead.
J = 1/m * sum(sum(  -Y_Matrix .* log(H) - (1-Y_Matrix) .* log(1-H) ));

% PART 2
% Forward propagation ab
% 2. Calculate Delta of last layer
% δ3 or d3 is the difference between a3 and the y_matrix. The dimensions are the same as both, (m x r).
d3 = a3 - Y_Matrix;

% 3. Caclulate Delta of hidden layer
% Note: Excluding the first column of Theta2 via Theta2(:,2:end)' is because the hidden layer bias unit has no connection to the input layer - so we do not use backpropagation for it. See Figure 3 in ex4.pdf for a diagram showing this.
d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);
% Note that you should skip or remove 0 index of delta2.
% Note: Excluding the first column of Theta2 is because the hidden layer bias unit has no connection to the input layer - so we do not use backpropagation for it

DELTA1 = a1' * d2;
DELTA2 = a2' * d3;

Theta1_grad = 1/m * DELTA1;
Theta2_grad = 1/m * DELTA2;

% PARRT 3

% Note we should not regularize the terms that correspond to the bias. 
% For the matrices Theta1 and Theta2, this corresponds to the first column of each matrix.
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));
% Regularization
Reg = (lambda / (2*m))  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 )));
% Regularized cost function
J = J + Reg;











% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
