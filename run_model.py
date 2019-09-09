'''
Author: Alexis Laignelet
Date: 06/09/19
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle


def u_exact(t, X):  # (N+1) x 1, (N+1) x D
    '''
    Return the closed-form solution
    '''

    # Parameters of the function
    r = 0.05
    sigma_max = 0.4

    return np.exp((r + sigma_max**2) * (model.T - t)) * np.sum(X**2, 1, keepdims=True)


def save_pickle(content, filename):
    '''
    Save outputs of the model
    '''
    with open(filename, 'wb') as f:
        pickle.dump(content, f)


def load_pickle(filename):
    '''
    Load outputs of the model
    '''
    with open(filename, 'rb') as f:
        return pickle.load(f)


def plot_model(output_run, name='', samples=2):
    '''
    Create plots from the output file
    '''

    # Retrieve the different arrays
    graph, t_test, W_test, X_pred, Y_pred, Y_test, graph_pred = output_run[0], output_run[
        1], output_run[2], output_run[3], output_run[4], output_run[5], output_run[6]

    # PLot the loss function
    lost = plt.figure()
    plt.plot(graph, 'black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss function')
    plt.title('Evolution of the training loss')
    plt.yscale('log')

    # Plot Y0 prediction
    pred = plt.figure()
    plt.plot(graph_pred, 'black')
    plt.xlabel('Iteration')
    plt.ylabel('$Y_0$')
    plt.title('Evolution of the prediction of $Y_0$')

    # Plot the stochastic paths
    path = plt.figure()
    plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T,
             'red', label='Learned $u(t,X_t)$')
    plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T,
             'black', ls='--', label='Exact $u(t,X_t)$')
    plt.scatter(t_test[0:1, -1, 0], Y_test[0:1, -1, 0],
                color='black', marker='D')
    plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, 'red')
    plt.plot(t_test[1:samples, :, 0].T,
             Y_test[1:samples, :, 0].T, 'black', ls='--')
    plt.scatter(t_test[1:samples, -1, 0],
                Y_test[1:samples, -1, 0], color='black', marker='D')
    plt.plot([0], Y_test[0, 0, 0], color='black', marker='D')
    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title('Black-Scholes-Barenblatt')
    plt.legend()

    # Compute the error
    errors = np.sqrt((Y_test - Y_pred)**2 / Y_test**2)
    mean_errors = np.mean(errors, 0)
    std_errors = np.std(errors, 0)

    # Plot the error
    error = plt.figure()
    plt.plot(t_test[0, :, 0], mean_errors, 'black', label='mean')
    plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors,
             'black', ls='--', label='mean + 2 std')
    plt.xlabel('$t$')
    plt.ylabel('Relative error')
    plt.title('Black-Scholes-Barenblatt')
    plt.legend()

    # Save the plots
    lost.savefig('lost.pdf', bbox_inches='tight')
    pred.savefig('pred.pdf', bbox_inches='tight')
    path.savefig('path.pdf', bbox_inches='tight')
    error.savefig('error.pdf', bbox_inches='tight')
