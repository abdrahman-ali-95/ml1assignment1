import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from plot_utils import (plot_scatterplot_and_line, plot_scatterplot_and_polynomial, 
                        plot_logistic_regression, plot_datapoints, plot_3d_surface, plot_2d_contour,
                        plot_function_over_iterations)
from linear_regression import (fit_univariate_lin_model, 
                               fit_multiple_lin_model, 
                               univariate_loss, multiple_loss,
                               calculate_pearson_correlation,
                               compute_design_matrix,
                               compute_polynomial_design_matrix)
from logistic_regression import (create_design_matrix_dataset_1,
                                 create_design_matrix_dataset_2,
                                 create_design_matrix_dataset_3,
                                 logistic_regression_params_sklearn)
from gradient_descent import griewank, gradient_griewank, gradient_descent, finite_difference_gradient_approx


def task_1(use_linalg_formulation=False):
    print('---- Task 1 ----')
    column_to_id = {"mpg": 0, "cylinders": 1,
                    "displacement": 2, "horsepower": 3, 
                    "weight": 4, "acceleration": 5, 
                    "model_year": 6, "origin": 7}

    # After loading the data, you can for example access it like this: 
    # `cars_data[:, column_to_id['weight']]`
    cars_data = np.load('data/cars_data.npy')

    # TODO: Implement Task 1.1.2: Find 3 pairs of features that have a linear relationship.
    # For each pair, fit a univariate linear regression model: If ``use_linalg_formulation`` is False,
    # call `fit_univariate_lin_model`, otherwise use the linalg formulation in `fit_multiple_lin_model` (Task 1.2.2).
    # For each pair, also calculate and report the Pearson correlation coefficient, the theta vector you found, 
    # the MSE, and plot the data points together with the linear function.
    # Repeat the process for 3 pairs of features that do not have a meaningful linear relationship.
    
    correlated_pairs = [
        ('displacement', 'weight'),
        ('horsepower', 'weight'),
        ('mpg', 'weight')
    ]

    uncorrelated_pairs = [
        ('cylinders', 'model_year'),
        ('horsepower', 'model_year'),
        ('weight', 'model_year')
    ]
    
    all_pairs = [('Correlated', correlated_pairs), ('Uncorrelated', uncorrelated_pairs)]

    for pair_type, pairs in all_pairs:
         
        for feature_name, target_name in pairs:
            
            x = cars_data[:, column_to_id[feature_name]]
            y = cars_data[:, column_to_id[target_name]]
            
            
            r = calculate_pearson_correlation(x, y)
            
            
            if not use_linalg_formulation:
                theta = fit_univariate_lin_model(x, y)
            else:
                
                X_design = compute_design_matrix(x.reshape(-1, 1))
                theta = fit_multiple_lin_model(X_design, y)
                
            loss = univariate_loss(x, y, theta)
            
            print(f"Feature: {feature_name:15} | Target: {target_name:12} | "
                  f"Pearson r: {r:.4f} | MSE: {loss:.4f} | "
                  f"Theta (b, w): ({theta[0]:.4f}, {theta[1]:.4f})")
            
            # Plot the results
            plot_scatterplot_and_line(
                x, y, theta, 
                xlabel=feature_name, ylabel=target_name, 
                title=f"{pair_type}: {feature_name} vs {target_name} (r={r:.2f})",
                figname=f"1_1_{pair_type}_{feature_name}_{target_name}"
            )


    # TODO: Implement Task 1.2.3: Multiple linear regression
    # Select two additional features, compute the design matrix, and fit the multiple linear regression model.
    # Report the MSE and the theta vector.
    
    y_mult = cars_data[:, column_to_id['weight']]

    x1 = cars_data[:, column_to_id['displacement']]
    x2 = cars_data[:, column_to_id['horsepower']]
    x3 = cars_data[:, column_to_id['mpg']]
    data_mult = np.column_stack((x1, x2, x3))

    X_mult = compute_design_matrix(data_mult)
    theta_M = fit_multiple_lin_model(X_mult, y_mult)
    loss_M = multiple_loss(X_mult, y_mult, theta_M)

    print(f"Features: displacement, horsepower, mpg | Target: weight")
    print(f"Multiple MSE Loss: {loss_M:.4f}")
    print(f"Theta M (b, w1, w2, w3): ({theta_M[0]:.4f}, {theta_M[1]:.4f}, {theta_M[2]:.4f}, {theta_M[3]:.4f})")

    x_uni = cars_data[:, column_to_id['displacement']]
    theta_U = fit_univariate_lin_model(x_uni, y_mult)
    loss_U = univariate_loss(x_uni, y_mult, theta_U)
    print(f"Univariate MSE Loss (displacement only): {loss_U:.4f}")


    # TODO: Implement Task 1.3.2: Polynomial regression
    # For the feature-target pair of choice, compute the polynomial design matrix with an appropriate degree K, 
    # fit the model, and plot the data points together with the polynomial function.
    # Report the MSE and the theta vector.
    
    x_poly = cars_data[:, column_to_id['displacement']]
    y_poly = cars_data[:, column_to_id['weight']]

    K_good = 2 
    X_poly = compute_polynomial_design_matrix(x_poly, K_good)
    theta_poly = fit_multiple_lin_model(X_poly, y_poly)
    loss_poly = multiple_loss(X_poly, y_poly, theta_poly)
    
    print(f"Polynomial Degree {K_good} | MSE Loss: {loss_poly:.4f}")
    plot_scatterplot_and_polynomial(
        x_poly, y_poly, theta_poly, 
        xlabel='displacement', ylabel='weight', 
        title=f"Polynomial fit (Degree {K_good})",
        figname="1_3_poly_fit"
    )


    # TODO: Implement Task 1.3.3: Use x_small_1 and y_small_1 to fit a polynomial model.
    # Find and report the smallest K that gets zero loss. Plot the data points and the polynomial function.
    x_small_1 = cars_data[10:15, column_to_id['displacement']]
    y_small_1 = cars_data[10:15, column_to_id['weight']]    
    
    K_perfect = 4 
    X_small_1 = compute_polynomial_design_matrix(x_small_1, K_perfect)
    theta_small_1 = fit_multiple_lin_model(X_small_1, y_small_1)
    loss_small_1 = multiple_loss(X_small_1, y_small_1, theta_small_1)
    
    print(f"Small Data 1 | Degree {K_perfect} | MSE Loss: {loss_small_1:.4f}")
    plot_scatterplot_and_polynomial(
        x_small_1, y_small_1, theta_small_1,
        xlabel='displacement', ylabel='weight',
        title=f"Perfect fit N=5 (Degree {K_perfect})",
        figname="1_3_small_1"
    )


    #TODO: Implement 1.3.4: Use the K from Task 1.3.3, x_small_2 and y_small_2 to fit a polynomial model.
    #Plot the data points and the polynomial function. Report why the loss is not zero in this case.
    x_small_2 = cars_data[169:174, column_to_id['displacement']]
    y_small_2 = cars_data[169:174, column_to_id['weight']]    
    
    X_small_2 = compute_polynomial_design_matrix(x_small_2, K_perfect)
    theta_small_2 = fit_multiple_lin_model(X_small_2, y_small_2)
    loss_small_2 = multiple_loss(X_small_2, y_small_2, theta_small_2)
    
    print(f"Small Data 2 | Degree {K_perfect} | MSE Loss: {loss_small_2:.4f}")
    plot_scatterplot_and_polynomial(
        x_small_2, y_small_2, theta_small_2,
        xlabel='displacement', ylabel='weight',
        title=f"Failed perfect fit N=5 (Degree {K_perfect})",
        figname="1_3_small_2"
    )


def task_2():
    print('\n---- Task 2 ----')

    for task in [1, 2, 3]:
        print(f'---- Logistic regression task {task} ----')
        if task == 1:
            # TODO: Load the data set 1 (X-1-data.npy and targets-dataset-1.npy)
            X_data = np.load('data/X-1-data.npy') # TODO: change me
            y = np.load('data/targets-dataset-1.npy') # TODO: change me
            create_design_matrix = create_design_matrix_dataset_1
        elif task == 2:
            # TODO: Load the data set 2 (X-2-data.npy and targets-dataset-2.npy)
            X_data = np.load('data/X-2-data.npy') # TODO: change me
            y = np.load('data/targets-dataset-2.npy') # TODO: change me
            create_design_matrix = create_design_matrix_dataset_2
        elif task == 3:
            # TODO: Load the data set 3 (X-3-data.npy and targets-dataset-3.npy)
            X_data = np.load('data/X-3-data.npy') # TODO: change me
            y = np.load('data/targets-dataset-3.npy') # TODO: change me
            create_design_matrix = create_design_matrix_dataset_3
        else:
            raise ValueError('Task not found.')

        X = create_design_matrix(X_data)

        # Plot the datapoints (just for visual inspection)
        plot_datapoints(X, y, f'Targets - Task {task}')

        # TODO: Split the dataset using the `train_test_split` function.
        # The parameter `random_state` should be set to 0.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        print(f'Shapes of: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}')

        # Train the classifier
        custom_params = logistic_regression_params_sklearn()
        clf = LogisticRegression(**custom_params)
        # TODO: Fit the model to the data using the `fit` method of the classifier `clf`

        clf.fit(X_train, y_train)

        acc_train, acc_test = clf.score(X_train, y_train), clf.score(X_test, y_test) # TODO: Use the `score` method of the classifier `clf` to calculate accuracy

        print(f'Train accuracy: {acc_train * 100:.2f}%. Test accuracy: {100 * acc_test:.2f}%.')
        
        yhat_train = clf.predict_proba(X_train)[:, 1] # TODO: Use the `predict_proba` method of the classifier `clf` to
                          #  calculate the predicted probabilities on the training set
        yhat_test = clf.predict_proba(X_test)[:, 1] # TODO: Use the `predict_proba` method of the classifier `clf` to
                         #  calculate the predicted probabilities on the test set

        # TODO: Use the `log_loss` function to calculate the cross-entropy loss
        #  (once on the training set, once on the test set).
        #  You need to pass (1) the true binary labels and (2) the probability of the *positive* class to `log_loss`.
        #  Since the output of `predict_proba` is of shape (n_samples, n_classes), you need to select the probabilities
        #  of the positive class by indexing the second column (index 1).
      
        loss_train, loss_test = log_loss(y_train, yhat_train), log_loss(y_test, yhat_test)
      
        print(f'Train loss: {loss_train}. Test loss: {loss_test}.')

        plot_logistic_regression(clf, create_design_matrix, X_train, f'(Dataset {task}) Train set predictions',
                                 figname=f'logreg_train{task}')
        plot_logistic_regression(clf, create_design_matrix, X_test,  f'(Dataset {task}) Test set predictions',
                                 figname=f'logreg_test{task}')

        # TODO: Print theta vector (and also the bias term). Hint: Check the attributes of the classifier
        classifier_weights, classifier_bias = clf.coef_, clf.intercept_
        print(f'Parameters: {classifier_weights}, {classifier_bias}')


def task_3(initial_plot=True):
    print('\n---- Task 3 ----')
    # Do *not* change this seed
    np.random.seed(0)

    # TODO: Choose a random starting point using samples from a uniform distribution
    x0 = np.random.uniform(-10, 10)
    y0 = np.random.uniform(-10, 10)
    print(f'Starting point: {x0:.4f}, {y0:.4f}')

    if initial_plot:
        # Plot the function to see how it looks like
        plot_3d_surface(griewank)
        plot_2d_contour(griewank, starting_point=(x0, y0), global_min=(0, 0))

    # TODO: Check if gradient_griewank is correct at (x0, y0). 
    # To do this, print the true gradient and the numerical approximation.
    
    true_grad = gradient_griewank(x0, y0)
    approx_grad = finite_difference_gradient_approx(griewank, x0, y0)
    print(f"True analytical gradient:  {true_grad}")
    print(f"Numerical approx gradient: {approx_grad}")

    # TODO: Call the function `gradient_descent` with a chosen configuration of hyperparameters,
    #  i.e., learning_rate, lr_decay, and num_iters. Try out lr_decay=1 as well as values for lr_decay that are < 1.
    x_list, y_list, f_list = gradient_descent(
        griewank, gradient_griewank, 
        x0, y0, 
        learning_rate = 5,
        lr_decay = 0.8,
        num_iters = 250
    )

    # Print the point that is found after `num_iters` iterations
    print(f'Solution found: f({x_list[-1]:.4f}, {y_list[-1]:.4f})= {f_list[-1]:.4f}' )
    print(f'Global optimum: f(0, 0)= {griewank(0, 0):.4f}')

    # Here we plot the contour of the function with the path taken by the gradient descent algorithm
    plot_2d_contour(griewank, starting_point=(x0, y0), global_min=(0, 0), 
                    x_list=x_list, y_list=y_list)

    # TODO: Create a plot f(x_t, y_t) over iterations t by calling `plot_function_over_iterations` with `f_list`
    
    plot_function_over_iterations(f_list)


def main():
    np.random.seed(46)
    task_1(use_linalg_formulation=False)
    task_2()
    task_3(initial_plot=True)


if __name__ == '__main__':
    main()
