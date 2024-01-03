#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: zhaitongqing
@file: hparam.py
@time: 2023/4/19 11:36
@desc: 
"""
import yaml


def load_hparam(filename):
    stream = open(filename, 'r')
    docs = yaml.safe_load_all(stream)
    hparam_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = v
    return hparam_dict


def merge_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_dict(user[k], v)
    return user


class Dotdict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct  # 如果空，就进行dct = dict()初始化，否则就用传进来的参数dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):  # hasattr() 函数用于判断对象是否包含对应的属性。用于字典嵌套字典的转换
                value = Dotdict(value)
            self[key] = value


class Hparam(Dotdict):

    def __init__(self, file='config/config.yaml'):
        super(Dotdict, self).__init__()
        hp_dict = load_hparam(file)
        hp_dotdict = Dotdict(hp_dict)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)

    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__


hparam = Hparam()
