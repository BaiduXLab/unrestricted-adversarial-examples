import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm
from torch.autograd import Variable

import torch
import torch.nn as nn

import multiprocessing

from adversarialbox.utils import to_var
from imagenet_c import corrupt
import multiprocessing

import pdb

# --- White-box attacks ---

class Attack(object):
	name = None

	# TODO: Refactor this out of this object
	_stop_after_n_datapoints = None	# An attack can optionally run on only a subset of the dataset

	def __call__(self, *args, **kwargs):
		raise NotImplementedError()

def corrupt_float32_image(x, corruption_name, severity):
	"""Convert to uint8 and back to conform to corruption API"""
	x = np.copy(x)	# We make a copy to avoid changing things in-place
	x = (x * 255).astype(np.uint8)

	corrupt_x = corrupt(
		x,
		corruption_name=corruption_name,
		severity=severity)
	return corrupt_x.astype(np.float32) / 255.


def _corrupt_float32_image_star(args):
	return corrupt_float32_image(*args)


class CommonCorruptionsAttack(Attack):
    name = "common_corruptions"

    def __init__(self, return_all, severity=1):
        self.corruption_names = [
            'gaussian_noise',
            'shot_noise',
            'impulse_noise',
            'defocus_blur',
            'glass_blur',
            'motion_blur',
            'zoom_blur',
            # 'snow', # Snow does not work in python 2.7
            # 'frost', # Frost is not working correctly
            'fog',
            'brightness',
            'contrast',
            'elastic_transform',
            'pixelate',
            'jpeg_compression',
            'speckle_noise',
            'gaussian_blur',
            'spatter',
            'saturate']
        self.severity = severity
        self.pool = multiprocessing.Pool(len(self.corruption_names))

    def __call__(self, images_batch_nhwc):
        assert images_batch_nhwc.shape[1:] == (224, 224, 3), \
            "Image shape must equal (N, 224, 224, 3)"
        batch_size = len(images_batch_nhwc)

        # Keep track of the worst corruption for each image
        worst_corruption = np.copy(images_batch_nhwc)
        worst_loss = [np.NINF] * batch_size

        corrupt_out = []
        # Iterate through each image in the batch
        for batch_idx, x in enumerate(images_batch_nhwc):
            corrupt_args = [(x, corruption_name, self.severity)
                                            for corruption_name in self.corruption_names]
            corrupt_x_batch = self.pool.map(_corrupt_float32_image_star, corrupt_args)
            if batch_idx == 0:
                corrupt_out = corrupt_x_batch
            else:
                corrupt_out = np.concatenate((corrupt_out, corrupt_x_batch), axis=0)
            

        return corrupt_out

class FGSMAttack(object):
    def __init__(self, model=None, epsilon=None):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y, epsilons=None):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons

        X = np.copy(X_nat)

        X_var = to_var(torch.from_numpy(X), requires_grad=True)
        y_var = to_var(torch.LongTensor(y))

        scores = self.model(X_var)
        loss = self.loss_fn(scores, y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()

        X += self.epsilon * grad_sign
        X = np.clip(X, 0, 1)

        return X


class LinfPGDAttack_v2(object):
    def __init__(self, epsilon=0.3, k=40, a=0.01, 
        random_start=True, multi_out=False):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """

        self.multi_out = multi_out
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start

    def __call__(self, X_nat, y, model, loss_fn, mode):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        X_nat_np = X_nat.numpy()
        
        model_cp = copy.deepcopy(model)
        for p in model_cp.parameters():
            p.requires_grad = False
        
        if mode == 'train':
            model_cp.train()
        elif mode == 'eval': 
            model_cp.eval()
        else:
            raise ValueError('Invalid mode.')
        
        model_cp.eval()
        if self.rand:
            X = X_nat_np + np.random.uniform(-self.epsilon, self.epsilon,
                X_nat_np.shape).astype('float32')
        else:
            X = np.copy(X_nat_np)
        

        for i in range(self.k):
            X_var = Variable(torch.from_numpy(X).cuda(), requires_grad=True, volatile=False)
            y_var = y.cuda()
            scores = model_cp(X_var)
            if self.multi_out:
                scores = scores[0]
            
            loss = loss_fn(scores, y_var)
            model_cp.zero_grad()
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()
            X_var.grad.zero_()

            X += self.a * np.sign(grad)
            X = np.clip(X, X_nat_np - self.epsilon, X_nat_np + self.epsilon)
            X = np.clip(X, 0, 1) # ensure valid pixel range
        return torch.from_numpy(X)

class LinfPGDAttack(object):
    def __init__(self, model=None, epsilon=0.3, k=40, a=0.01, 
        random_start=True, multi_out=False):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """

        self.multi_out = multi_out
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        if self.rand:
            X = X_nat + np.random.uniform(-self.epsilon, self.epsilon,
                X_nat.shape).astype('float32')
        else:
            X = np.copy(X_nat)

        for i in range(self.k):
            X_var = to_var(torch.from_numpy(X), requires_grad=True)
            y_var = to_var(torch.LongTensor(y))

            scores = self.model(X_var)
            if self.multi_out:
                scores = scores[0]
            loss = self.loss_fn(scores, y_var)
            #self.model.zero_grad()
            loss.backward(retain_graph=True)
            grad = X_var.grad.data.cpu().numpy()

            X += self.a * np.sign(grad)

            X = np.clip(X, X_nat - self.epsilon, X_nat + self.epsilon)
            X = np.clip(X, 0, 1) # ensure valid pixel range

        return X


# --- Black-box attacks ---

def jacobian(model, x, nb_classes=10):
    """
    This function will return a list of PyTorch gradients
    """
    list_derivatives = []
    x_var = to_var(torch.from_numpy(x), requires_grad=True)

    # derivatives for each class
    for class_ind in range(nb_classes):
        score = model(x_var)[:, class_ind]
        score.backward()
        list_derivatives.append(x_var.grad.data.cpu().numpy())
        x_var.grad.data.zero_()

    return list_derivatives


def jacobian_augmentation(model, X_sub_prev, Y_sub, lmbda=0.1):
    """
    Create new numpy array for adversary training data
    with twice as many components on the first dimension.
    """
    X_sub = np.vstack([X_sub_prev, X_sub_prev])

    # For each input in the previous' substitute training iteration
    for ind, x in enumerate(X_sub_prev):
        grads = jacobian(model, x)
        # Select gradient corresponding to the label predicted by the oracle
        grad = grads[Y_sub[ind]]

        # Compute sign matrix
        grad_val = np.sign(grad)

        # Create new synthetic point in adversary substitute training set
        X_sub[len(X_sub_prev)+ind] = X_sub[ind] + lmbda * grad_val #???

    # Return augmented training data (needs to be labeled afterwards)
    return X_sub
