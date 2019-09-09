'''
Author: Alexis Laignelet
Date: 06/09/19
'''

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.distributions import Normal
import numpy as np


class SGLD(Optimizer):
    '''
    Stochastic Gradient Langevin Dynamics
    '''

    def __init__(self, params, lr=required, **kwargs):
        '''
        Instanciate with beta parameter
        '''
        # Take the value of beta from the **kwargs
        beta = kwargs.get('beta', 1)

        # Create a dictionary of parameters
        defaults = dict(lr=lr, beta=beta, addnoise=True)
        super(SGLD, self).__init__(params, defaults)

        # Handle CPU / GPU
        device_idx = 0
        if torch.cuda.is_available():
            self.device = torch.device(
                "cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
            torch.backends.cudnn.deterministic = True

        else:
            self.device = torch.device("cpu")

    def step(self):
        """
        Performs a single optimization step (code from Dr Panos Parpas)
        """
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Save the grad for every tensors with requires_grad
                d_p = p.grad.data

                if group['addnoise']:
                    # Add the langevin noise
                    size = d_p.size()
                    beta = group['beta']
                    langevin_noise = Normal(
                        torch.zeros(size).to(self.device),
                        torch.ones(size).to(self.device) /
                        np.sqrt(beta * group['lr'])
                    )
                    p.data.add_(-group['lr'], d_p +
                                langevin_noise.sample().to(self.device))
                else:
                    # Performs vanilla SGD
                    p.data.add_(-group['lr'], d_p)

        return loss


class CTLD(Optimizer):
    '''
    Stochastic Gradient Langevin Dynamics
    '''

    def __init__(self, params, lr=required, **kwargs):
        '''
        Instanciate the optimiser with the given parameters
        '''

        # Ls is the number of iterations for the exploration phase
        Ls = kwargs.get('Ls', 300)

        # Save the weights of the neural networks to optimise
        save = list(params)

        # Handle CPU / GPU
        device_idx = 0
        if torch.cuda.is_available():
            self.device = torch.device(
                "cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device("cpu")

        # Initialise the parameters for every layers
        list_r_0 = []
        list_r_alpha_0 = []

        # Choose a random seed
        torch.manual_seed(0)
        np.random.seed(0)

        # Go throught every layers
        for weight in save:

            # Create a list of r_0
            size = weight.shape
            gaussian_r = Normal(torch.zeros(size).to(
                self.device), torch.ones(size).to(self.device))
            r_0 = gaussian_r.sample().to(self.device)
            list_r_0.append(r_0)

            # Create a list of r_alpha
            r_alpha_0 = np.random.normal(0, 1, 1)[0]
            list_r_alpha_0.append(r_alpha_0)

        list_alpha = np.zeros(len(list_r_0))

        # Chose the Hyperparameters
        # Here is the default values

        # gamma = (1 - c_m) / lr with c_m between 0 and 1
        # gamma_alpha = 1 / lr
        # K = 300
        # sigma = 0.04
        # C = delta_prime / lr**2
        # w = 20/(lr**2 * Ls * K)

        delta_prime = 1.5
        cm = 0.5

        defaults = dict(lr=lr,
                        gamma=(1 - cm) / lr,
                        list_r=list_r_0,
                        list_alpha=list_alpha,
                        delta=0.4,
                        delta_prime=delta_prime,
                        S=0.85,
                        C=delta_prime / lr**2,
                        K=300,
                        V=0,
                        list_r_alpha=list_r_alpha_0,
                        gamma_alpha=1 / lr,
                        w=1,
                        sigma=0.04,
                        Vk=0,
                        Vk1=0,
                        verbose=False,
                        it=0,
                        Ls=Ls)

        print('####################################\n'
              '######### Hyperparameters ##########\n'
              '####################################\n'
              'Gamma: ', defaults['gamma'], '\n'
              'Delta: ',  defaults['delta'], '\n'
              'Delta_prime: ',  defaults['delta_prime'], '\n'
              'S: ', defaults['S'], '\n'
              'C: ', defaults['C'], '\n'
              'K: ', defaults['K'], '\n'
              'w: ', defaults['w'], '\n'
              'sigma: ', defaults['sigma'], '\n')

        super(CTLD, self).__init__(save, defaults)

    def step(self, loss_fun):
        """
        Performs a single optimization step
        """
        for group in self.param_groups:

            # Retrieve all the Hyperparameters
            lr = group['lr']
            list_r = group['list_r']
            gamma = group['gamma']
            list_alpha = group['list_alpha']
            delta = group['delta']
            delta_prime = group['delta_prime']
            S = group['S']
            C = group['C']
            K = group['K']
            V = group['V']
            list_r_alpha = group['list_r_alpha']
            gamma_alpha = group['gamma_alpha']
            w = group['w']
            sigma = group['sigma']
            Vk = group['Vk']
            Vk1 = group['Vk1']
            Ls = group['Ls']
            it = group['it']

            # Counter of the number of iterations
            it += 1
            if it == Ls:
                print('End of stochastic regime')
            group['it'] = it

            # Define K intervals in [-delta_prime, delta_prime]
            alpha_range = np.linspace(-delta_prime, delta_prime, K + 1)

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    # If there is no requires_grad
                    continue

                # Retreive r, r_alpha and alpha for the current layer
                r = list_r[i]
                r_alpha = list_r_alpha[i]
                alpha = list_alpha[i]

                if it >= Ls:
                    # Convergence phase
                    # Update the weights
                    p.data += lr * r

                    # Update r
                    d_p = p.grad.data
                    list_r[i] = (1 - gamma * lr) * r - lr * d_p

                else:
                    # Exploration phase
                    # Compute gradient of loss respect to the parameters
                    d_p = p.grad.data
                    size = d_p.size()

                    # Sample eps and eps_alpha
                    gaussian_epsilon = Normal(torch.zeros(size).to(
                        self.device), torch.ones(size).to(self.device))
                    epsilon = gaussian_epsilon.sample().to(self.device)

                    epsilon_alpha = np.random.normal(0, 1, 1)

                    # Update theta
                    p.data += lr * r

                    # Update r
                    factor = np.sqrt((2 * lr * gamma) /
                                     self.fun_g(alpha, delta, delta_prime, S))
                    list_r[i] = (1 - gamma * lr) * r - \
                        lr * d_p + factor * epsilon

                    # Update alpha
                    list_alpha[i] += lr * r_alpha

                    # Update V
                    if alpha > delta_prime:
                        alpha_k = alpha_range[-2]
                        alpha_k1 = alpha_range[-1]
                    elif alpha < -delta_prime:
                        alpha_k = alpha_range[0]
                        alpha_k1 = alpha_range[1]
                    elif abs(alpha) < delta_prime:
                        alpha_k = alpha_range[alpha_range < alpha].max()
                        alpha_k1 = alpha_range[alpha_range > alpha].min()

                    # Update Vk / Vk+1
                    Vk += w * np.exp(-(alpha_k - alpha)**2 / (2 * sigma**2))
                    Vk1 += w * np.exp(-(alpha_k1 - alpha)**2 / (2 * sigma**2))

                    # Evaluate h_tilde
                    h_tilde = - \
                        self.grad_fun_g(alpha, delta, delta_prime,
                                        S) * self.hamiltonian(loss_fun, r)
                    h_tilde += - \
                        self.grad_phi(alpha, delta_prime, C) - \
                        (Vk1 - Vk) / (2 * delta_prime / K)

                    # Update r_alpha
                    list_r_alpha[i] = (1 - lr * gamma_alpha) * r_alpha + h_tilde * \
                        lr + np.sqrt(2 * lr * gamma_alpha) * epsilon_alpha

                    # For debugin purpose
                    if group['verbose'] == True:
                        print('####################################\n'
                              '######### Hyperparameters ##########\n'
                              '####################################\n'
                              'Learning rate: ', lr, '\n'
                              'Gamma: ', gamma, '\n'
                              'Gamma_alpha: ', gamma_alpha, '\n'
                              'C: ', C, '\n'
                              'w: ', w, '\n'
                              'sigma: ', sigma, '\n'
                              'K: ', K, '\n'
                              'Delta: ', delta, '\n'
                              'Delta_prime: ', delta_prime, '\n'
                              'S: ', S, '\n'
                              '\n'
                              '####################################\n'
                              '###### Intermediate results ########\n'
                              '####################################\n'
                              'g(alpha): ', self.fun_g(
                                  alpha, delta, delta_prime, S), '\n'
                              'grad_phi: ', self.grad_phi(
                                  alpha, delta_prime, C), '\n'
                              'epsilon: ', epsilon, '\n'
                              'epsilon_alpha: ', epsilon_alpha, '\n'
                              'factor eps: ', factor, '\n'
                              '\n'
                              '####################################\n'
                              '############## Updates #############\n'
                              '####################################\n'
                              'Theta: ', p.data, '\n'
                              'r: ', list_r[i], '\n'
                              'Alpha: ', list_r[i], '\n'
                              'h_tilde: ', h_tilde, '\n'
                              'r_alpha', list_r_alpha[i], '\n'
                              '\n'
                              '\n'
                              )

                # Save the new parameters
                group['list_r'] = list_r
                group['list_r_alpha'] = list_r_alpha
                group['list_alpha'] = list_alpha

    def fun_z(self, alpha, delta, delta_prime):
        '''
        Function z to build the piecewise function g
        '''
        return (abs(alpha) - delta) / (delta_prime - delta)

    def fun_g(self, alpha, delta, delta_prime, S):
        '''
        Function g
        '''

        # Piecewise function
        if abs(alpha) <= delta:
            return 1
        elif (delta < abs(alpha)) and (abs(alpha) < delta_prime):
            transition = 1 - S * (3 * self.fun_z(alpha, delta, delta_prime)
                                  ** 2 - 2 * self.fun_z(alpha, delta, delta_prime)**3)
            return transition
        elif abs(alpha) >= delta_prime:
            return 1 - S

    def grad_fun_g(self, alpha, delta, delta_prime, S):
        '''
        Gradient of the function g
        '''

        # Piecewise function
        if abs(alpha) <= delta:
            return 0
        elif (delta < abs(alpha)) and (abs(alpha) < delta_prime):
            return - 6 * S / (delta_prime - delta) * (self.fun_z(alpha, delta, delta_prime) - self.fun_z(alpha, delta, delta_prime)**2)
        elif abs(alpha) >= delta_prime:
            return 0

    def grad_phi(self, alpha, delta_prime, C):
        '''
        Gradient of the function phi meant to induce a force
        '''

        # Induces a force whene alpha is out of bounds [-delta_prime, delta_prime]
        if alpha > delta_prime:
            return C
        elif alpha < -delta_prime:
            return -C
        else:
            return 0

    def hamiltonian(self, U, r):
        '''
        Compute the Hamiltonian
        '''
        h = U.cpu().detach().numpy() + 1 / 2 * \
            np.linalg.norm(r.cpu().detach().numpy(), 2)**2
        return h
