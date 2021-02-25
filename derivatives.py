import torch
import numpy as np

# jacobian with respect to input
def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    if len(flat_y) == 1:
        (grad_x,) = torch.autograd.grad(
            y, x, None, retain_graph=True, create_graph=create_graph
        )
        return grad_x.reshape(x.numel())

    grad_y = torch.zeros_like(flat_y)

    for i in range(len(flat_y)):
        grad_y[i] = 1.0
        (grad_x,) = torch.autograd.grad(
            flat_y, x, grad_y, retain_graph=True, create_graph=create_graph
        )
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.0
    return torch.stack(jac).reshape(y.shape + (x.numel(),))

# naive hessian with respect to input
def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)

# hessian with respect to all parameters of a model
def hessian_wrt_all_params(y, model):
    number_of_wsvdhts = count_parameters(model)
    result = torch.zeros(number_of_wsvdhts, number_of_wsvdhts)
    index_i = 0
    for i in range(len(params(model))):
        param_hsvdht = params(model)[i].numel()
        index_j = index_i + param_hsvdht

        jacob_i = jacobian(y, params(model)[i], create_graph=True)

        # Calculate upper triangle of hessian, and construct full hessian (because hessian is symmetric)
        for j in range(i + 1, len(params(model))):
            param_width = params(model)[j].numel()

            jacob_ij = jacobian(jacob_i, params(model)[j])
            hess_ij = jacob_ij.view(param_hsvdht, -1)

            result[
                index_i: index_i + param_hsvdht, index_j: index_j + param_width
            ] = hess_ij
            result[
                index_j: index_j + param_width, index_i: index_i + param_hsvdht
            ] = hess_ij.T

            index_j += param_width

        # ヘシアン対角の計算
        jacob_ii = jacobian(jacob_i, params(model)[i])
        hess_ii = jacob_ii.view(param_hsvdht, param_hsvdht)
        hess_ii = symmetrize(hess_ii)
        result[
            index_i: index_i + param_hsvdht, index_i: index_i + param_hsvdht
        ] = hess_ii

        index_i += param_hsvdht

    return result
  
  
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
  

# function that returns all parameters of a model (`nn.Sequential` NEEDS TO BE STORED IN `self.classifier`!)
def params(model):
    params = []
    for i, layer in enumerate(model.classifier):
        try:
            params.append(layer.weight)
            params.append(layer.bias)
        except:
            continue
    return params
