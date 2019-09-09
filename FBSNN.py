'''
Author: Alexis Laignelet
Date: 06/09/19
'''

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
from gradient_descent import *
import torch.nn.functional as F


class activation_function(nn.Module):
    '''
    Activation function of the neural network
    '''

    def __init__(self, option):
        super(activation_function, self).__init__()
        self.option = option

    def forward(self, x):
        if self.option == 'sin':
            return torch.sin(x)
        if self.option == 'relu':
            return F.relu(x)


class FBSNN:
    '''
    Create a neural network and train it
    '''

    def __init__(self, Xi, T, M, N, D, layers, **params):
        '''
        Instanciate the neural network
        '''
        # Define the attributes from the **kwargs, along with default values
        self.test = params.get('test_mode', False)
        self.opt = params.get('optimiser', 'Adam')
        self.activation = params.get('activation', 'sin')
        self.implicit = params.get('implicit', False)
        self.tau_implicit = params.get('tau_implicit', 1e-3)
        self.iter_implicit = params.get('iter_implicit', 10)
        self.seed = params.get('seed', 0)
        self.record_implicit = params.get('record_implicit', False)
        self.params = params

        # Handle CPU / GPU
        device_idx = 0
        if torch.cuda.is_available():
            self.device = torch.device(
                "cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
            torch.backends.cudnn.deterministic = True

        else:
            self.device = torch.device("cpu")

        # Set initial point
        self.Xi = torch.from_numpy(Xi).float().to(self.device)
        self.Xi.requires_grad = True

        # Define terminal time T, number of trajectories M,
        # number of time steps N and number of dimensions D
        self.T = T
        self.M = M
        self.N = N
        self.D = D

        # Create the layers: linear + activation function
        self.layers = []
        for i in range(len(layers) - 2):
            self.layers.append(
                nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
            self.layers.append(activation_function(self.activation))
        self.layers.append(
            nn.Linear(in_features=layers[-2], out_features=layers[-1]))

        # Build a sequential model from the previously defined layers
        self.model = nn.Sequential(*self.layers).to(self.device)
        self.model.apply(self.weights_init)

        # Initialise standard normal distribution
        self.m = torch.distributions.normal.Normal(0, 1)

    def weights_init(self, m):
        '''
        Weights initialisation
        '''

        # Check if the test mode is active
        if self.test == True:
            # In test mode, a random seed is used (same as TF implementation)
            np.random.seed(1234)
            if type(m) == nn.Linear:
                m.weight.data = torch.from_numpy(np.random.rand(
                    m.weight.shape[1], m.weight.shape[0]).T).float().to(self.device)
                m.weight.data.requires_grad = True
                nn.init.zeros_(m.bias)
        else:
            # Xavier initialisation is done in normal mode
            if type(m) == nn.Linear:
                torch.manual_seed(self.seed)
                torch.nn.init.xavier_uniform_(m.weight)

    def net_u(self, t, X):  # M x 1, M x D
        '''
        Make a forward pass to have Y and then compute Z
        '''

        input = torch.cat((t, X), 1)

        # Forward pass to have Y
        u = self.model(input)

        # Taking the gradient of Y with respect to X gives Z
        Du = torch.autograd.grad(outputs=[u], inputs=[X], grad_outputs=torch.ones_like(
            u), allow_unused=True, retain_graph=True, create_graph=True)[0]

        return u, Du

    def Dg_tf(self, X):  # M x D
        '''
        Compute the gradient of the terminal condition
        '''

        g = self.g_tf(X)
        Dg = torch.autograd.grad(outputs=[g], inputs=[X], grad_outputs=torch.ones_like(
            g), allow_unused=True, retain_graph=True, create_graph=True)[0]

        return Dg

    def loss_function(self, t, W, Xi):
        '''
        Make predictions and compute loss function
        '''

        loss = 0

        # Empty list to store X and Y through time
        X_list = []
        Y_list = []

        # Initial values for time t and Brownian motion W
        t0 = t[:, 0, :]
        W0 = W[:, 0, :]

        # Use the initial values to instanciate X0
        X0 = Xi.repeat(self.M, 1).view(self.M, self.D)  # M x D

        # Make a forward pass to have corresponding Y0 and Z0
        Y0, Z0 = self.net_u(t0, X0)  # M x 1, M x D

        # Store the values through time
        X_list.append(X0)
        Y_list.append(Y0)

        # Loop through all time steps
        for n in range(0, self.N):

            # Get the current time value
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]

            # Compute the next step for X1
            X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.squeeze(
                torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1)), dim=-1)

            # Compute Y1_tilde which is here the true value
            Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.sum(Z0 * torch.squeeze(
                torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1))), dim=1, keepdim=True)

            # Make prediction with the neural network
            Y1, Z1 = self.net_u(t1, X1)

            # Add the difference to the loss
            loss += torch.sum(torch.pow(Y1 - Y1_tilde, 2))

            # Move by one time step
            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1

            # Store the values of X and Y
            X_list.append(X0)
            Y_list.append(Y0)

        # For the last step the true value is the terminal condition
        loss += torch.sum(torch.pow(Y1 - self.g_tf(X1), 2))
        loss += torch.sum(torch.pow(Z1 - self.Dg_tf(X1), 2))

        X = torch.stack(X_list, dim=1)
        Y = torch.stack(Y_list, dim=1)

        return loss, X, Y, Y[0, 0, 0]

    def fetch_minibatch(self, fixed_predict=False):
        '''
        Generate time and Brownian motion
        '''

        # Assigning the parameters
        T = self.T
        M = self.M
        N = self.N
        D = self.D

        # Define time and Brownian motion between two steps
        Dt = np.zeros((M, N + 1, 1))  # M x (N+1) x 1
        DW = np.zeros((M, N + 1, D))  # M x (N+1) x D

        dt = T / N
        Dt[:, 1:, :] = dt

        # For comparision purpose
        if self.test == True or fixed_predict == True:
            np.random.seed(1234)
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))

        # Generate t and W, then to be fed into the neural network
        t = np.cumsum(Dt, axis=1)  # M x (N+1) x 1
        W = np.cumsum(DW, axis=1)  # M x (N+1) x D
        t = torch.from_numpy(t).float().to(self.device)
        W = torch.from_numpy(W).float().to(self.device)

        return t, W

    def train(self, N_Iter, learning_rate):
        '''
        Training function
        '''

        # Empty array to strore the lost function and also the prediction Y at t=0
        training_loss = np.array([])
        Y0_prediction = np.array([])

        if self.test == True:
            # In test mode the optimiser is deterministic
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=learning_rate)
            print('Optimiser: SGD')

        else:
            # For the normal mode, the optimiser is the one passed in parameter
            if self.opt == 'Adam':
                self.optimizer = optim.Adam(
                    self.model.parameters(), lr=learning_rate)

            elif self.opt == 'SGD':
                self.optimizer = optim.SGD(
                    self.model.parameters(), lr=learning_rate)

            elif self.opt == 'SGLD':
                self.optimizer = SGLD(
                    self.model.parameters(), lr=learning_rate, **self.params)

            elif self.opt == 'CTLD':
                self.optimizer = CTLD(
                    self.model.parameters(), lr=learning_rate, **self.params)

            print('Optimiser: ', self.opt)

        # Calculate the time for a certain number of iterations
        start_time = time.time()

        # Loop over the iterations
        for it in range(N_Iter):
            if self.implicit == False:
                # For explicit scheme
                t_batch, W_batch = self.fetch_minibatch()  # M x (N+1) x 1, M x (N+1) x D
                loss, X_pred, Y_pred, Y0_pred = self.loss_function(
                    t_batch, W_batch, self.Xi)

                # Desactivate the graph when parameters are updated
                self.optimizer.zero_grad()
                loss.backward()
                if self.opt == 'CTLD':
                    # The loss needs to be passed as an input for CTLD
                    self.optimizer.step(loss)
                else:
                    self.optimizer.step()

            else:
                # For implicit scheme
                t_batch, W_batch = self.fetch_minibatch()  # M x (N+1) x 1, M x (N+1) x D

                # All the weights are concatenated and then detached so they stay fixed
                weights = torch.cat([param.view(-1)
                                     for param in self.model.parameters()])
                weights_ = weights.detach()

                # Loop over the number of inner iterations
                for i in range(self.iter_implicit):
                    weights = torch.cat([param.view(-1)
                                         for param in self.model.parameters()])
                    loss, X_pred, Y_pred, Y0_pred = self.loss_function(
                        t_batch, W_batch, self.Xi)

                    # The proximal is defined
                    loss_bis = loss + 1 / \
                        (2 * self.tau_implicit) * \
                        torch.norm(weights - weights_, 2)**2

                    # Desactivate the graph when parameters are updated to solve the inner problem
                    self.optimizer.zero_grad()
                    loss_bis.backward()
                    self.optimizer.step()

                    # Save the intermediate steps for ploting purpose
                    if (self.record_implicit == True) and i < self.iter_implicit - 1:
                        training_loss = np.append(
                            training_loss, loss.cpu().detach().numpy())
                        Y0_prediction = np.append(
                            Y0_prediction, Y0_pred.cpu().detach().numpy())

            # Print info to follow the evolution
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' %
                      (it, loss, Y0_pred, elapsed, learning_rate))
                start_time = time.time()

            # Store the values of the loss and Y at time 0
            training_loss = np.append(
                training_loss, loss.cpu().detach().numpy())
            Y0_prediction = np.append(
                Y0_prediction, Y0_pred.cpu().detach().numpy())

        return training_loss, Y0_prediction

    def predict(self, Xi_star, t_star, W_star):
        '''
        Make a prediction
        '''

        # Initialise X
        Xi_star = Xi_star.float().to(self.device)
        Xi_star.requires_grad = True

        # Make a forward pass and compute loss
        loss, X_star, Y_star, Y0_pred = self.loss_function(
            t_star, W_star, Xi_star)

        return X_star, Y_star

    def u_exact(self, t, X):  # (N+1) x 1, (N+1) x D
        '''
        Return the closed-form solution
        '''

        # Parameters of the function
        r = 0.05
        sigma_max = 0.4

        return np.exp((r + sigma_max**2) * (self.T - t)) * np.sum(X**2, 1, keepdims=True)

    def run_model(self, N_Iter, learning_rate):
        '''
        Run the model over a given number of iterations at a given learning rate
        '''

        # Check if the number of iterations is a list [1000, 2000, etc]
        if isinstance(N_Iter, list):
            graph = np.array([])
            graph_pred = np.array([])

            # Go through the list and train the model
            for i in range(len(N_Iter)):
                graph_temp, graph_pred_temp = self.train(
                    N_Iter[i], learning_rate[i])
                graph = np.append(graph, graph_temp)
                graph_pred = np.append(graph_pred, graph_pred_temp)
        else:
            graph, graph_pred = self.train(N_Iter, learning_rate)

        # Generate the test data (True is passed for comparison purpose)
        t_test, W_test = self.fetch_minibatch(True)

        # Compute the prediction
        X_pred, Y_pred = self.predict(self.Xi, t_test, W_test)

        # Convert t, W, X and Y into Numpy if not already
        if type(t_test).__module__ != 'numpy':
            t_test = t_test.cpu().numpy()
        if type(W_test).__module__ != 'numpy':
            W_test = W_test.cpu().numpy()
        if type(X_pred).__module__ != 'numpy':
            X_pred = X_pred.cpu().detach().numpy()
        if type(Y_pred).__module__ != 'numpy':
            Y_pred = Y_pred.cpu().detach().numpy()

        # Compute the true value for test data
        Y_test = np.reshape(self.u_exact(np.reshape(
            t_test[0:self.M, :, :], [-1, 1]), np.reshape(X_pred[0:self.M, :, :], [-1, self.D])), [self.M, -1, 1])

        # Store the parameters of the run to access then later
        self.params['iteration'] = N_Iter
        self.params['learning_rate'] = learning_rate
        self.params['M'] = self.M
        self.params['D'] = self.D
        self.params['N'] = self.N

        output = [graph, t_test, W_test, X_pred,
                  Y_pred, Y_test, graph_pred, self.params]

        return output

    # Define the EDP to solve
    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        '''
        Drift of the stochastic process Y
        '''
        return 0.05 * (Y - torch.sum(X * Z, dim=1, keepdim=True))  # M x 1

    def g_tf(self, X):  # M x D
        '''
        Terminal condition
        '''
        return torch.sum(X ** 2, 1, keepdim=True)  # M x 1

    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        '''
        Drift of the stochastic process X
        '''
        M = self.M
        D = self.D
        return torch.zeros([M, D]).to(self.device)  # M x D

    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        '''
        Volatility of the stochastic process X
        '''
        return 0.4 * torch.diag_embed(X)  # M x D x D


class FBSNN_proxy():
    '''
    Create a neural network using proximal backpropagation and train it
    '''

    def __init__(self, Xi, T, M, N, D, layers, **params):
        '''
        Instanciate the neural network
        '''
        # Define the attributes from the **kwargs, along with default values
        self.activation = params.get('activation', 'sin')
        self.tau_prox = params.get('tau_prox', 1)
        self.seed = params.get('seed', 0)
        self.params = params

        # Handle GPU / CPU
        device_idx = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(device_idx)
                                       if torch.cuda.is_available() else "cpu")
            torch.backends.cudnn.deterministic = True

        else:
            self.device = torch.device("cpu")

        # Set itinial point
        self.Xi = torch.from_numpy(Xi).float().to(self.device)  # initial point
        self.Xi.requires_grad = True

        # Define terminal time T, number of trajectories M,
        # number of time steps N and number of dimensions D
        self.T = T
        self.M = M
        self.N = N
        self.D = D

        # Create the layers: linear + activation function
        self.layers = []
        for i in range(len(layers) - 2):
            self.layers.append(
                nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
            self.layers.append(activation_function(self.activation))
        self.layers.append(
            nn.Linear(in_features=layers[-2], out_features=layers[-1]))

        # Build a sequential model from the previously defined layers
        self.model = nn.Sequential(*self.layers).to(self.device)
        self.model.apply(self.weights_init)

        # Initialise standard normal distribution
        self.m = torch.distributions.normal.Normal(0, 1)

    def forward(self, x):
        '''
        Compute a forward pass and store intermediate results
        '''
        # Define empty list
        ai = []
        zi = []

        y = x
        for i in range(len(self.layers)):
            # Compute the output of the i-th layer
            y = self.layers[i](y)
            if (type(self.layers[i]) == nn.Linear) and (i < len(self.layers) - 1):
                # Store the result before the activation function
                zi.append(y)
            elif i < len(self.layers) - 1:
                # Store the result after the activation function
                ai.append(y)

        return ai, zi, y

    def weights_init(self, m):
        '''
        Weights initialisation
        '''
        # Xavier initialisation
        if type(m) == nn.Linear:
            torch.manual_seed(self.seed)
            torch.nn.init.xavier_uniform_(m.weight)

    def Dg_tf(self, X):  # M x D
        '''
        Compute the gradient of the terminal condition
        '''

        g = self.g_tf(X)
        Dg = torch.autograd.grad(outputs=[g], inputs=[X], grad_outputs=torch.ones_like(
            g), allow_unused=True, retain_graph=True, create_graph=True)[0]  # M x D
        return Dg

    def loss_function(self, t, W, Xi):
        '''
        Make predictions and compute loss function
        '''

        loss = 0

        # Empty list to store X and Y through time
        X_list = []
        Y_list = []

        # Initial values for time t and Brownian motion W
        t0 = t[:, 0, :]
        W0 = W[:, 0, :]

        # Use the initial values to instanciate X0
        X0 = Xi.repeat(self.M, 1).view(self.M, self.D)  # M x D

        # Compute thetaL-1
        weight = self.layers[-1].weight
        bias = self.layers[-1].bias.reshape(self.layers[-1].weight.shape[0], 1)
        thetaL1 = torch.cat((weight, bias), 1).t()

        # Input for the neural network
        x0 = torch.cat((t0, X0), 1)

        # Compute the list of intermediate results at t=0
        a0, z0, _ = self.forward(x0)

        # Add a column of 1 to a0 (to take into account the bias)
        a0_auto = torch.cat(
            (a0[-1], torch.ones(a0[-1].shape[0], 1).to(self.device)), 1)

        # Compute Y0 in doing a forward pass through the neural network
        Y0 = torch.mm(a0_auto, thetaL1)

        # Store the output of the penultimate layer at t=0
        al2tot = [a0_auto]

        # Compute Z0
        Z0 = torch.autograd.grad(outputs=[Y0], inputs=[X0], grad_outputs=torch.ones_like(
            Y0), allow_unused=True, retain_graph=True, create_graph=True)[0]

        # Now we have x0, a0, Y0, Z0
        xtot = x0
        atot = a0
        ztot = z0
        utot = Y0

        # Store the values through time
        X_list.append(X0)
        Y_list.append(Y0)

        # Loop through all time steps
        for n in range(0, self.N):

            # Get the current time value
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]

            # Compute the next step for X1
            X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.squeeze(
                torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1)), dim=-1)

            # Compute Y1_tilde which is here the true value
            Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.sum(Z0 * torch.squeeze(
                torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1))), dim=1, keepdim=True)

            # Build the input and compute the list of intermediate results
            xi = torch.cat((t1, X1), 1)
            ai, zi, _ = self.forward(xi)

            # Compute the new Y1
            ai_auto = torch.cat(
                (ai[-1], torch.ones(ai[-1].shape[0], 1).to(self.device)), 1)
            Y1 = torch.mm(ai_auto, thetaL1)
            al2tot.append(ai_auto)

            # Compute new Z1
            Z1 = torch.autograd.grad(outputs=[Y1], inputs=[X1], grad_outputs=torch.ones_like(
                Y1), allow_unused=True, retain_graph=True, create_graph=True)[0]

            # Store the values
            xtot = torch.cat([xtot, xi])
            atot = [torch.cat([atot[i], ai[i]]) for i in range(len(atot))]
            ztot = [torch.cat([ztot[i], zi[i]]) for i in range(len(ztot))]
            utot = torch.cat([utot, Y1])

            # Compute the loss
            loss += torch.sum(torch.pow(Y1 - Y1_tilde, 2))

            # Move by one time step
            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1

            # Store the values of X and Y
            X_list.append(X0)
            Y_list.append(Y0)

        # For the last step the true value is the terminal condition
        loss += torch.sum(torch.pow(Y1 - self.g_tf(X1), 2))
        loss += torch.sum(torch.pow(Z1 - self.Dg_tf(X1), 2))

        X = torch.stack(X_list, dim=1)
        Y = torch.stack(Y_list, dim=1)

        # Instanciate the tensor of gradients of the loss with respect to the output of the penultimate layers for every time steps
        grad_aL2 = torch.Tensor([]).to(self.device)

        # Go throught every time steps
        for ind in range(len(al2tot)):

            # Compute the gradient for time t
            aut = torch.autograd.grad(outputs=[loss], inputs=[al2tot[ind]], grad_outputs=torch.ones_like(
                loss), allow_unused=True, retain_graph=True, create_graph=True)[0]
            grad_aL2 = torch.cat([grad_aL2, aut])

        # Compute the gradient of the loss with respect to the weights of the last layer
        grad_thetaL1 = torch.autograd.grad(outputs=[loss], inputs=[thetaL1], grad_outputs=torch.ones_like(
            loss), allow_unused=True, retain_graph=True, create_graph=True)[0]

        return loss, X, Y, Y[0, 0, 0], xtot, atot, ztot, utot, grad_aL2, grad_thetaL1

    def fetch_minibatch(self, fixed_predict=False):
        '''
        Generate time and Brownian motion
        '''

        # Assigning the parameters
        T = self.T
        M = self.M
        N = self.N
        D = self.D

        # Define time and Brownian motion between two steps
        Dt = np.zeros((M, N + 1, 1))  # M x (N+1) x 1
        DW = np.zeros((M, N + 1, D))  # M x (N+1) x D

        dt = T / N
        Dt[:, 1:, :] = dt

        # For comparison purpose
        if fixed_predict == True:
            np.random.seed(1234)
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))

        # Generate t and W, then to be fed into the neural network
        t = np.cumsum(Dt, axis=1)  # M x (N+1) x 1
        W = np.cumsum(DW, axis=1)  # M x (N+1) x D
        t = torch.from_numpy(t).float().to(self.device)
        W = torch.from_numpy(W).float().to(self.device)

        return t, W

    def solve_implicit(self, A, Y, Z, beta):
        '''
        Compute the exact solution of the proximal
        '''
        AA = A.t().mm(A)
        I = torch.eye(A.size(1)).type_as(A)
        A_tilde = AA + beta * I
        b_tilde = A.t().mm(Y) + beta * Z
        X, _ = torch.solve(b_tilde, A_tilde)

        return X

    def train(self, N_Iter, learning_rate):
        '''
        Training function
        '''

        # Empty array to strore the lost function and also the prediction Y at t=0
        training_loss = np.array([])
        Y0_prediction = np.array([])

        # Use the value of tau passed as a parameter
        tau = self.tau_prox

        # Use the activation function and make a forward pass
        activ = activation_function(self.activation)
        def act(x): return activ.forward(x)

        # Calculate the time for a certain number of iterations
        start_time = time.time()

        # Loop over the iterations
        for it in range(N_Iter):
            # Generate the data for training
            t_batch, W_batch = self.fetch_minibatch()  # M x (N+1) x 1, M x (N+1) x D

            # Retreive all the tensors from the loss function including the intermediate results and gradients
            loss, X_pred, Y_pred, Y0_pred, xtot, atot, ztot, utot, grad_aL2, grad_thetaL1 = self.loss_function(
                t_batch, W_batch, self.Xi)

            #########################################
            #     Update a_{L-2} and theta_{L-1}    #
            #########################################

            # aL-2 is the output of the penultimate layer
            aL2 = atot[-1]

            # Add a column of ones to take into account the bias
            aL2 = torch.cat(
                (aL2, torch.ones(aL2.shape[0], 1).to(self.device)), 1)

            # Compute the weights of the last layer thetaL-1
            weight = self.layers[-1].weight
            bias = self.layers[-1].bias.reshape(
                self.layers[-1].weight.shape[0], 1)
            thetaL1 = torch.cat((weight, bias), 1).t()

            # Update aL-2 with the gradient from loss function
            aL2_update = aL2 - learning_rate * grad_aL2

            # Update the list with the updated aL-2
            atot[-1] = aL2_update

            # Update thetaL-1 with the gradient from loss function
            thetaL1_update = thetaL1 - learning_rate * grad_thetaL1

            # Assign the new weights
            self.layers[-1].weight = torch.nn.Parameter(
                thetaL1_update.t()[:, :-1])
            self.layers[-1].bias = torch.nn.Parameter(
                thetaL1_update.t()[:, -1])

            #########################################
            # Update every z_i, a_{i-1} and theta_i #
            #########################################

            for i in range(len(atot) - 1):

                # Define current zi and ai
                zi = ztot[-i - 1]
                ai = atot[-i - 1]

                # Compute the derivative of the activation function with respect to zi
                activ_fun = act(zi)
                grad_act = torch.autograd.grad(outputs=[activ_fun], inputs=[zi], grad_outputs=torch.ones_like(activ_fun),
                                               allow_unused=True, retain_graph=True, create_graph=True)[0]

                # Update zi
                zi_update = zi - grad_act * (act(zi) - ai[:, :-1])

                # Retrieve ai-1 from the list of all ai
                ai1 = atot[-i - 2]
                ai1 = torch.cat(
                    (ai1, torch.ones(ai1.shape[0], 1).to(self.device)), 1)

                # Compute thetai
                weight = self.layers[-1 - (2 * (i + 1))].weight
                bias = self.layers[-1 - (2 * (i + 1))].bias.reshape(self.layers[-1 -
                                                                                (2 * (i + 1))].weight.shape[0], 1)
                thetai = torch.cat((weight, bias), 1).t()

                # Update ai-1
                ai1_update = ai1 - \
                    torch.mm(torch.mm(ai1, thetai) - zi_update, thetai.t())

                # Update the list with the updated ai-1
                atot[-i - 2] = ai1_update

                # Update thetai
                thetai_update = self.solve_implicit(ai1, zi_update, thetai, 1 / tau)

                # Assign the weights
                self.layers[-1 - (2 * (i + 1))
                            ].weight = torch.nn.Parameter(thetai_update.t()[:, :-1])
                self.layers[-1 - (2 * (i + 1))
                            ].bias = torch.nn.Parameter(thetai_update.t()[:, -1])

            #########################################
            #              Update theta1            #
            #########################################

            # Retrieve z1 from the list
            z1 = ztot[0]

            # Compute the gradient of the activation function
            activ_fun = act(z1)
            grad_act = torch.autograd.grad(outputs=[activ_fun], inputs=[z1], grad_outputs=torch.ones_like(activ_fun),
                                           allow_unused=True, retain_graph=True, create_graph=True)[0]

            # Update z1
            z1_update = z1 - grad_act * (act(z1) - atot[0][:, :-1])

            # Compute theta1
            weight = self.layers[0].weight
            bias = self.layers[0].bias.reshape(
                self.layers[0].weight.shape[0], 1)
            theta1 = torch.cat((weight, bias), 1).t()

            # a0 is the input of the neural network
            a0 = torch.cat(
                (xtot, torch.ones(xtot.shape[0], 1).to(self.device)), 1)

            # Update theta1
            theta1_update = self.solve_implicit(a0, z1_update, theta1, 1 / tau)

            # Assign the weights
            self.layers[0].weight = torch.nn.Parameter(
                theta1_update.t()[:, :-1])
            self.layers[0].bias = torch.nn.Parameter(theta1_update.t()[:, -1])

            # Print info to follow the evolution
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' %
                      (it, loss, Y0_pred, elapsed, learning_rate))
                start_time = time.time()

            # Store the values of the loss and Y at time 0
            training_loss = np.append(
                training_loss, loss.cpu().detach().numpy())
            Y0_prediction = np.append(
                Y0_prediction, Y0_pred.cpu().detach().numpy())

        return training_loss, Y0_prediction

    def predict(self, Xi_star, t_star, W_star):
        '''
        Make a prediction
        '''

        # Initialise X
        Xi_star = Xi_star.float().to(self.device)
        Xi_star.requires_grad = True

        # Make a forward pass and compute loss
        loss, X_star, Y_star, Y0_pred, _, _, _, _, _, _ = self.loss_function(
            t_star, W_star, Xi_star)

        return X_star, Y_star

    def u_exact(self, t, X):  # (N+1) x 1, (N+1) x D
        '''
        Return the closed-form solution
        '''

        # Parameters of the function
        r = 0.05
        sigma_max = 0.4

        return np.exp((r + sigma_max**2) * (self.T - t)) * np.sum(X**2, 1, keepdims=True)

    def run_model(self, N_Iter, learning_rate):
        '''
        Run the model over a given number of iterations at a given learning rate
        '''

        # Check if the number of iterations is a list [1000, 2000, etc]
        if isinstance(N_Iter, list):
            graph = np.array([])
            graph_pred = np.array([])

            # Go through the list and train the model
            for i in range(len(N_Iter)):
                graph_temp, graph_pred_temp = self.train(
                    N_Iter[i], learning_rate[i])
                graph = np.append(graph, graph_temp)
                graph_pred = np.append(graph_pred, graph_pred_temp)
        else:
            graph, graph_pred = self.train(N_Iter, learning_rate)

        # Generate the test data (True is passed for comparison purpose)
        t_test, W_test = self.fetch_minibatch(True)

        # Compute the prediction
        X_pred, Y_pred = self.predict(self.Xi, t_test, W_test)

        # Convert t, W, X and Y into Numpy if not already
        if type(t_test).__module__ != 'numpy':
            t_test = t_test.cpu().numpy()
        if type(W_test).__module__ != 'numpy':
            W_test = W_test.cpu().numpy()
        if type(X_pred).__module__ != 'numpy':
            X_pred = X_pred.cpu().detach().numpy()
        if type(Y_pred).__module__ != 'numpy':
            Y_pred = Y_pred.cpu().detach().numpy()

        # Compute the true value for test data
        Y_test = np.reshape(self.u_exact(np.reshape(
            t_test[0:self.M, :, :], [-1, 1]), np.reshape(X_pred[0:self.M, :, :], [-1, self.D])), [self.M, -1, 1])

        # Store the parameters of the run to access then later
        self.params['iteration'] = N_Iter
        self.params['learning_rate'] = learning_rate
        self.params['M'] = self.M
        self.params['D'] = self.D
        self.params['N'] = self.N

        output = [graph, t_test, W_test, X_pred,
                  Y_pred, Y_test, graph_pred, self.params]

        return output

    # Define the EDP to solve
    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        '''
        Drift of the stochastic process Y
        '''
        return 0.05 * (Y - torch.sum(X * Z, dim=1, keepdim=True))  # M x 1

    def g_tf(self, X):  # M x D
        '''
        Terminal condition
        '''
        return torch.sum(X ** 2, 1, keepdim=True)  # M x 1

    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        '''
        Drift of the stochastic process X
        '''
        M = self.M
        D = self.D
        return torch.zeros([M, D]).to(self.device)  # M x D

    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        '''
        Volatility of the stochastic process X
        '''
        return 0.4 * torch.diag_embed(X)  # M x D x D
