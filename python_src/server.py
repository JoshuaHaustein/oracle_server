import os
import time
import logging
from multiprocessing import Process
logging.basicConfig(level=logging.INFO)

import numpy as np

from models import Oracle, Feasibility, Pushability
from oracle_pb2 import (
    ActionRequest,
    ActionResponse,
    FeasibilityRequest,
    FeasibilitySampleRequest,
    PushabilityRequest
)


def get_pipe_path(name):
    path = os.environ.get(name)
    if not path:
        raise EnvironmentError('Environment variable {} not set'.format(name))
    else:
        return path


def oracle_loop():
    try:
        logging.info('Oracle loop started')
        action_request = ActionRequest()
        for attr in action_request.FindInitializationErrors():
            setattr(action_request, attr, 0.0)
        message_size = action_request.ByteSize()
        oracle = Oracle()
        request_path = get_pipe_path('ORACLE_REQUEST_PIPE_PATH')
        response_path = get_pipe_path('ORACLE_RESPONSE_PIPE_PATH')

        while True:
            with open(request_path, 'rb') as request_pipe:
                data = request_pipe.read(message_size)
                ar = ActionRequest.FromString(data)
                response = oracle.sample(ar)
                if len(data) == 0:
                    break
                pass
            with open(response_path, 'wb') as response_pipe:
                response_pipe.write(response.SerializeToString())
    except KeyboardInterrupt:
        pass

def feasibility_sample_loop():
    try:
        logging.info('Feasibility sample loop started')
        request = FeasibilitySampleRequest()
        for attr in request.FindInitializationErrors():
            setattr(request, attr, 0.0)
        message_size = request.ByteSize()
        feasibility = Feasibility()
        request_path = get_pipe_path('FEASIBILITY_SAMPLE_REQUEST_PIPE_PATH')
        response_path = get_pipe_path('FEASIBILITY_SAMPLE_RESPONSE_PIPE_PATH')

        while True:
            with open(request_path, 'rb') as request_pipe:
                data = request_pipe.read(message_size)
                request = FeasibilitySampleRequest.FromString(data)
                response = feasibility.sample(request)
                if len(data) == 0:
                    break
                pass
            with open(response_path, 'wb') as response_pipe:
                response_pipe.write(response.SerializeToString())
    except KeyboardInterrupt:
        pass


def pushability_loop():
    try:
        logging.info('Pushability loop started')
        pushability_request = PushabilityRequest()
        for attr in pushability_request.FindInitializationErrors():
            setattr(pushability_request, attr, 0.0)
        message_size = pushability_request.ByteSize()
        request_path = get_pipe_path('PUSHABILITY_REQUEST_PIPE_PATH')
        response_path = get_pipe_path('PUSHABILITY_RESPONSE_PIPE_PATH')
        pushability = Pushability()

        while True:
            with open(request_path, 'rb') as request_pipe:
                data = request_pipe.read(message_size)
                request = PushabilityRequest.FromString(data)
                response = pushability.mahalanobis(request)
                if len(data) == 0:
                    break
                pass
            with open(response_path, 'wb') as response_pipe:
                response_pipe.write(response.SerializeToString())
    except KeyboardInterrupt:
        pass


def feasibility_loop():
    try:
        logging.info('Feasibility loop started')
        feasibility_request = FeasibilityRequest()
        for attr in feasibility_request.FindInitializationErrors():
            setattr(feasibility_request, attr, 0.0)
        message_size = feasibility_request.ByteSize()
        feasibility = Feasibility()
        request_path = get_pipe_path('FEASIBILITY_REQUEST_PIPE_PATH')
        response_path = get_pipe_path('FEASIBILITY_RESPONSE_PIPE_PATH')

        while True:
            with open(request_path, 'rb') as request_pipe:
                data = request_pipe.read(message_size)
                ar = ActionRequest.FromString(data)
                response = feasibility.mahalanobis(ar)
                if len(data) == 0:
                    break
                pass
            with open(response_path, 'wb') as response_pipe:
                response_pipe.write(response.SerializeToString())
    except KeyboardInterrupt:
        pass
        

if __name__ == '__main__':
    oracle_proc = Process(target=oracle_loop)
    oracle_proc.start()
    pushability_proc = Process(target=pushability_loop)
    pushability_proc.start()
    feasibility_proc = Process(target=feasibility_loop)
    feasibility_proc.start()
    feasibility_sample_proc = Process(target=feasibility_sample_loop)
    feasibility_sample_proc.start()
    

#while True:
