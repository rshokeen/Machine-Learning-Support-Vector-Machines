function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

size(X) %211 by 2 matrix
size(y) %211 by 1 matrix
size(Xval) %200 by 2 matrix
size(yval) %200 by 1 matrix

C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';


error_val_matrix = zeros(length(C_vec), length(sigma_vec)); %8 by 8 matrix
%error_val_matrix

for i = 1:length(C_vec)
  
    C_test = C_vec(i);
    
    for j = 1:length(sigma_vec) %for polynomial degree 1 to 8
    
    sigma_test = sigma_vec(j);
    
    %train on training set
    model_train= svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
    
    %measure validation error on validation set and pick the lowest validation set error
    %m x 1 column of predictions of {0, 1} values 
    prediction = svmPredict(model_train, Xval);
    
    %size(prediction) %200 by 1 matrix
    error_val_matrix(i,j) = mean(double(prediction ~= yval)); % 8 by 8 matrix
    
    
    end
    
    
    
end
%error_val_matrix
[min_val, row_index] = min(min(error_val_matrix, [ ], 2));
[min_val, col_index] = min(min(error_val_matrix, [ ], 1));
%min_val %0.03
%row_index %5
%col_index %3
C = C_vec(row_index);
sigma = sigma_vec(col_index);

% =========================================================================

end
