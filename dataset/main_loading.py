"""
Title: main_loading.py
Description: The loading functions.
Author: Lek'Sai Ye, University of Chicago
"""


from fmnist_loader import FashionMNISTLoader, FashionMNISTLoaderEval

# #########################################################################
# 1. Load Dataset in One Function
# #########################################################################
def load_dataset(loader_name: str='fmnist',
                 root: str='/net/leksai/data/FashionMNIST',
                 label_normal: tuple=(0,),
                 label_abnormal: tuple=(),
                 label_eval:tuple=(1,),
                 ratio_abnormal: float=1.,
                 test_eval: bool=False):

    known_loaders = ('fmnist', 'fmnist_eval', 'cifar-10')
    assert loader_name in known_loaders

    if loader_name == 'fmnist':
        return FashionMNISTLoader(root,
                                  label_normal,
                                  label_abnormal,
                                  ratio_abnormal)

    if loader_name == 'fmnist_eval':
        return FashionMNISTLoaderEval(root,
                                      label_eval,
                                      test_eval)

    return None
