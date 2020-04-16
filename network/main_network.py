"""
Title: main_network.py
Description: Build networks.
Reference: https://github.com/lukasruff/Deep-SAD-PyTorch/tree/master/src/networks
"""

from fmnist_LeNet import FashionMNISTLeNet, FashionMNISTLeNetAutoencoder


# #########################################################################
# 1. Build the Network Used for Training
# #########################################################################
def build_network(net_name='fmnist_LeNet_one_class'):
    known_networks = ('fmnist_LeNet_one_class', 'fmnist_LeNet_rec')
    assert net_name in known_networks

    net_name = net_name.strip()

    # The network for the one-class model training
    if net_name == 'fmnist_LeNet_one_class':
        return FashionMNISTLeNet(rep_dim=64)

    # The network for the reconstruction model training
    if net_name == 'fmnist_LeNet_rec':
        return FashionMNISTLeNetAutoencoder(rep_dim=64)
    return None


# #########################################################################
# 2. Build the Network Used for Pre-Training (Only for One-Class Model)
# #########################################################################
def build_autoencoder(net_name='fmnist_LeNet_one_class'):
    known_networks = ('fmnist_LeNet_one_class')
    assert net_name in known_networks

    net_name = net_name.strip()

    # The network for the one-class model pretraining
    if net_name == 'fmnist_LeNet_one_class':
        return FashionMNISTLeNetAutoencoder(rep_dim=64)

    return None
