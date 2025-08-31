import os
import math
import mlconfig
import logging
import GPUtil
import argparse
from datetime import datetime

class V():
    cfg: mlconfig.Config
    logger: logging.Logger

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(V, cls).__new__(cls)
        return cls.instance
    
    @staticmethod
    def info(msg, **kwargs):
        s = f'{datetime.now()} - INFO: {msg}'
        for k, v in kwargs.items(): s += f', {k}={v}'
        V().logger.info(s)

def pick_gpu():
    gpu = GPUtil.getAvailable(order='memory', limit=math.inf, maxLoad=2, maxMemory=0.1, includeNan=False,
                                    excludeID=[], excludeUUID=[])
    return gpu

def reserve_gpu(gpu_id):
    if gpu_id is None:
        gpu_id = pick_gpu()
    
    if gpu_id:
        V().cfg.device = gpu_id[0]
        V().info(f"Using CUDA {gpu_id[0]}")

def init_config(args):
    cfg = mlconfig.load(args.config)
    cfg.task_name = os.path.basename(args.config)
    os.makedirs(os.path.join(cfg.output_dir, cfg.task_name), exist_ok=True)
    V().cfg = cfg
    V().logger = logging.getLogger("")
    V().logger.addHandler(logging.FileHandler(os.path.join(cfg.output_dir, cfg.task_name, "log.txt")))
    V().logger.setLevel(logging.INFO)
    reserve_gpu(None)