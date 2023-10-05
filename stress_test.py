
#  _____________________________
#
#  TRELLIS DATA CONFIDENTIAL
#  _____________________________
#
#   [2016] - [2023] TRELLIS DATA PTY LTD
#   All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains
#  the property of TRELLIS DATA PTY LTD.
#  The intellectual and technical concepts contained
#  herein are proprietary to TRELLIS DATA PTY LTD
#  and may be covered by Australian and Foreign Patents,
#  patents in process, and are protected by trade secret or copyright law.
#  Dissemination of this information or reproduction of this material
#  is strictly forbidden unless prior written permission is obtained
#  from the CEO of TRELLIS DATA PTY LTD.

import argparse
import copy
import functools
import json
import random
import time
import uuid
import os
import time
import sys
import datetime
from multiprocessing import Pool, current_process
from typing import List, Dict, Any, Tuple, Optional

from absl import logging
from tqdm import tqdm

import api_helper
from file_load import InferenceFileLoader, ImageFileLoader, VideoFileLoader, AudioFileLoader

def wait_for_hours(hours):
    """Wait for the specified number of hours."""
    wait_time = datetime.timedelta(hours=hours)
    time.sleep(wait_time.total_seconds())

def stress_test(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    model_id = config["model_id"]
    
    # Redirecting print statements to a .txt file
    original_stdout = sys.stdout
    with open('stress_test_results.txt', 'a') as f:
        sys.stdout = f
        
        # Simulating the stress test
        print(f"Starting stress test for model {model_id}...")
        
        # Resetting stdout to original
        sys.stdout = original_stdout
    
    # Stopping the Docker container associated with the model
    os.system(f"docker stop {model_id}")
    
    # Starting the next model (assuming it's defined in the next config file)
    next_config_file = config_file.replace('.json', '2.json')
    if os.path.exists(next_config_file):
        with open(next_config_file, 'r') as f:
            next_config = json.load(f)
        next_model_id = next_config["model_id"]
        os.system(f"docker start {next_model_id}")

def run_stress_tests():
    # List of config files
    config_files = ["stress_test_config.json", "stress_test_config2.json"]  # Add more files as needed
    for config_file in config_files:
        stress_test(config_file)

if __name__ == "__main__":
    # Load the number of hours to wait from stress_test_config.json
    with open("stress_test_config.json", 'r') as f:
        config = json.load(f)
    hours_to_wait = config["hours_to_wait"]
    
    # Wait for the specified number of hours
    wait_for_hours(hours_to_wait)
    
    run_stress_tests()

def get_file_loader(file_path_or_dir: str, file_num: int, file_types: List[str]):
    if len(file_types) > 1:
        raise ValueError("currently not support for mixed file type inference")
    file_type: str = file_types[0]

    if file_type == "image":
        return ImageFileLoader(file_path_or_dir, file_num, AutoInferenceConstant.ACCEPTED_IMAGE_EXTENSIONS)
    elif file_type == "video":
        return VideoFileLoader(file_path_or_dir, file_num, AutoInferenceConstant.ACCEPTED_VIDEO_EXTENSIONS)
    elif file_type =="audio":
        return AudioFileLoader(file_path_or_dir, file_num, AutoInferenceConstant.ACCEPTED_AUDIO_EXTENSIONS)
    else:
        raise ValueError(f"currently not support file types other than {AutoInferenceConstant.ACCEPTED_FILE_TYPES},"
                         f"got {file_type} as inference_file_types in config")

class AutoInferenceConstant:
    """constants that are used for auto inference:
    LAST_RESULT_ONLY: after all inference requests are sent, only retrieve the status of the last request.
        When last request is done inference, record the total time and the test is over.

    FIXED_POOL: maintain the number of requests which are sent but not done inference to a fixed number.
        If one request is done inference, remove it from the pool; if number of request is less than pool size,
        send more requests to fill the pool. Test is finished when all requests are done inference

    WARM_UP_TIME: how many inferences to do on each model to warm up before benchmarking the time.
        Through experiment, it is found sometimes for image 1 is not enough, suggested 2 or 3

    ACCEPTED_[FILE_TYPE]_EXTENSIONS: the accepted extension for each [FILE_TYPE]. Currently only support
        image, video and audios. Files with these extension will be detected and used for inference
        when configured in "inference_file_type" in json config file

    RANDOM: when inference to multiple models, each request will pick a random model to be sent to
    EVEN: when inference to multiple models, requests will be evenly spread across all models

    """
    # inference types
    LAST_RESULT_ONLY: str = "last_result_only"
    FIXED_POOL: str = "fixed_pool"
    ALL_TYPE: List[str] = [LAST_RESULT_ONLY, FIXED_POOL]

    # how many inference to do on each model to warm up
    WARM_UP_TIME: int = 1

    # acceptable file types and extensions from tip.constants
    ACCEPTED_FILE_TYPES: List[str] = ["image", "video", "audio"]
    ACCEPTED_IMAGE_EXTENSIONS: List[str] = [
        # PNG
        ".png",
        # JPEG
        ".jpg", ".jpeg", ".jpe", ".jif", ".jfif", ".jfi",
        # JPEG 2000
        ".jp2", ".j2k", ".jpf", ".jpm", ".jpg2", ".j2c", ".jpc", ".jpx", ".mj2",
        # BMP
        ".bmp", ".dib"
    ]
    ACCEPTED_VIDEO_EXTENSIONS: List[str] = [".avi", ".mp4", ".gif", ".flv", ".mov"]
    ACCEPTED_AUDIO_EXTENSIONS: List[str] = [".mp3", ".wav", ".flv", ".ogg", ".wma", ".aac", ".flac", ".m4a", ".amr"]

    # how should the inference requests be sent to multiple models
    RANDOM: str = "random"  # send to randomly chosen model
    EVEN: str = "even"  # evenly distribute requests across the model


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """validate input configs before using for inference, return the adjusted config
        for now the possible adjusts are
        1. if number of workers is larger than inference number, lower number of workers = inference number
        2. if inference number is not divisible by number of workers, adjust inference number to make it divisible

    Args:
        config (Dict[str, Any]): configurations input for stress test

    Return:
        Dict[str, Any]: the adjusted configurations

    """
    model_ids: Optional[List[int]] = config["model_ids"]
    model_network_ids: Optional[List[int]] = config["model_network_ids"]
    dataset_ids: Optional[List[int]] = config["dataset_ids"]
    inference_file_types: List[str] = config["inference_file_types"]
    result_pulling_type: str = config["result_pulling_type"]

    def check_list_of_dtype(params_name: str, dtype: type) -> None:
        """a little helper function to help with logging error. The passed in params should be a list of dtype

        Args:
            params_name (str): the name of the parameter to check
            dtype (type): the type object that param's element should be

        """
        params: List[Any] = config[params_name]
        if not isinstance(params, list):
            raise TypeError(f"config['{params_name}'] should be a list, got {type(params)} instead")
        if not all(isinstance(param, dtype) for param in params):
            raise TypeError(f"config['{params_name}'] should be a list of {dtype}, got at least 1 element that is not")

    def check_positive_param(param_name: str, dtype: type) -> None:
        """another helper to check and logging positive number error

        Args:
            param_name: str the parameter that suppose to be a number
            dtype (type): the type object that param should be

        """
        param: Any = config[param_name]
        if not param:
            raise TypeError(f"config[{param_name}] cannot be null")
        if not isinstance(param, dtype):
            raise TypeError(f"config['{param_name}'] should be {dtype}, got {type(param)} instead")
        if param <= 0:
            raise ValueError(f"config['{param_name}'] should be positive, got {param} instead")

    # exactly one of model_ids and model_network_ids should be provided
    if not (model_ids or model_network_ids):
        raise TypeError(f"config['model_ids'] and config['model_network_ids'] cannot be null at the same time")

    if model_ids and model_network_ids:
        raise TypeError(f"only one of config['model_ids'] and config['model_network_ids'] is needed")

    # model_ids should be a list of int
    if model_ids:
        check_list_of_dtype("model_ids", int)

    # model_network_ids should be a list of int
    if model_network_ids:
        check_list_of_dtype("model_network_ids", int)

    # dataset_ids should be a list of int
    if dataset_ids:
        check_list_of_dtype("dataset_ids", int)

    # inference_file_types should be a list of string with certain values
    if inference_file_types:
        check_list_of_dtype("inference_file_types", str)
        for inference_file_type in inference_file_types:
            if inference_file_type not in AutoInferenceConstant.ACCEPTED_FILE_TYPES:
                raise ValueError(f"inference_file_type should be one of {AutoInferenceConstant.ACCEPTED_FILE_TYPES}, "
                                 f"got {inference_file_type} instead")
    else:
        raise TypeError(f"config['inference_file_types'] cannot be null")

    # num_workers should be a positive integer
    check_positive_param("num_workers", int)
    # total_inference_num should be a positive integer
    check_positive_param("total_inference_num", int)

    # explain should be "true" or "false", string, not boolean
    explain: str = config["explain"]
    if explain:
        if not (explain == "true" or explain == "false"):
            raise ValueError(f"explain should be string value takes either 'true' or 'false',"
                             f"got {explain} instead")
    else:
        logging.warning("explain is not set, default to true")

    # some parameters are needed when result_pulling_type == fixed_pool
    if result_pulling_type == AutoInferenceConstant.FIXED_POOL:
        check_positive_param("pool_size", int)
        check_positive_param("pool_monitor_time", float)


    total_inference_num: int = config["total_inference_num"]
    num_workers: int = config["num_workers"]
    # when number of workers are more than total inference number, it seems unnecessary and creates some bugs
    if total_inference_num < num_workers:
        logging.warning(f"total number of inference ({total_inference_num}) is smaller than "
                        f"number of workers specified ({num_workers}). "
                        f"Adjusting number of workers to {total_inference_num}")
        # one inference one worker should be sufficient
        num_workers: int = total_inference_num
        # update the config as well
        config["num_workers"]: int = num_workers

    # when total inference num is not divisible by num workers, make it divisible
    # this is for the purpose of spreading requests evenly across models, and more accurate performance evaluation
    if total_inference_num % num_workers != 0:
        # subtract the remainder to make inference number a multiple of number of workers
        config["total_inference_num"] = config["total_inference_num"] - (total_inference_num % num_workers)
        logging.warning(f"total number of inference ({total_inference_num}) is not divisible by "
                        f"number of workers specified ({num_workers}). "
                        f"Adjusting total number of inference to {config['total_inference_num']}")

    return config

def choose_model_id(request_spread_type: str, model_ids: List[int], index: int) -> int:
    """choose a model id from all models ids. Depending on request spread type configured,
    could be either evenly or randomly.

    Args:
        request_spread_type (str): the way to spread all the request across all models. Could be "even" or "random"
        model_ids (List[int]): all model ids deployed for testing
        index (int): index of current file, used when request are spread evenly

    Returns:
        int: model id that chose

    """
    if request_spread_type == AutoInferenceConstant.EVEN:
        return model_ids[(index % len(model_ids))]
    else:
        # if request_spread_type is invalid or unspecified, just send requests randomly to models
        return random.choice(model_ids)

def send_single_request(api: api_helper.InferenceServerAPIHelper,
                        model_id: int,
                        file_path: str,
                        is_model_network: bool,
                        dataset_ids: Optional[List[int]],
                        explain: str) -> uuid.UUID:
    """send a single inference request to model
    Args:
        api (api_helper.InferenceServerAPIHelper): class instance contains API methods
        model_id (int): model id to send inference request to
        file_path (str): file to send for inference
        is_model_network (bool): whether the inference is done on model or model network
        dataset_ids (Optional[List[int]]: for POI inference, a dataset id must be specified, otherwise set to None
        explain (str): explain flag in inference API body

    Returns:
        uuid.UUID: correlation id of the request sent

    """

    # generate a random correlation id for the current request
    cor_id: uuid.UUID = uuid.uuid4()

    # only POI inference needs a dataset id, other inference does not need this value
    dataset_id: Optional[int] = None
    if dataset_ids:
        # for dataset_id, just choose it randomly, has minor affect on performance
        dataset_id: int = random.choice(dataset_ids)

    # inference on model and inference on model network have different APIs
    if is_model_network:
        network_id: int = model_id
        api.model_network_inference(network_id, file_path, cor_id, explain)
    else:
        api.model_inference(model_id, dataset_id, file_path, cor_id, explain)

    return cor_id


def send_all_requests(file_path_list: List[str],
                      api: api_helper.InferenceServerAPIHelper,
                      config: Dict[str, Any]) -> Tuple[Dict[uuid.UUID, int], Dict[int, uuid.UUID]]:
    """sending all files in file path list for inference one by one

    Args:
        file_path_list (List[str]): list of all file paths
        api (api_helper.InferenceServerAPIHelper): class instance contains API methods
        config (Dict[str, Any]): configurations for stress test

    Returns:
        Tuple[Dict[uuid.UUID, int], Dict[int, uuid.UUID]]: a tuple of two dictionary
        where the first dictionary record each uuid and its corresponding model id,
        second dictionary record each model id and each model's last request's uuid,

    """
    num_file: int = len(file_path_list)
    request_spread_type: str = config["request_spread_type"]
    explain: str = config["explain"]
    # flag to remember do we inference on models or model network
    is_model_network: bool = False
    # for convenient, no matter inference is done on model or model network, let's just call the variable model_ids
    if config["model_ids"]:
        model_ids: List[int] = config["model_ids"]
    else:
        model_ids: List[int] = config["model_network_ids"]
        is_model_network: bool = True
    dataset_ids: Optional[List[int]] = config["dataset_ids"]
    # dictionary to keep track of every single request sent
    cor_id_to_model_id: Dict[uuid.UUID, int] = dict()
    # dictionary to keep track of the last request sent to each model
    model_id_to_last_cor_id: Dict[int, uuid.UUID] = dict()

    if len(current_process()._identity) > 0:
        # get current process id, purely for the purpose of evenly spread request across all models
        current_process_id: int = current_process()._identity[0]
    else:
        # no multiprocess, set id = 1
        current_process_id: int = 1

    for i in tqdm(range(num_file)):
        # revert to the index in original file list to spread requests evenly
        model_id: int = choose_model_id(request_spread_type, model_ids, i + (current_process_id - 1) * num_file)
        cor_id: uuid.UUID = send_single_request(api, model_id, file_path_list[i],
                                                is_model_network, dataset_ids, explain)
        # pair the correlation id and the model id together
        cor_id_to_model_id[cor_id]: int = model_id
        model_id_to_last_cor_id[model_id]: uuid.UUID = cor_id

    return cor_id_to_model_id, model_id_to_last_cor_id

def split(file_path_list: List[str], chunk_size: int) -> List[str]:
    """helper function to cut part of file paths out for multiprocess inference request sending

    Args:
        file_path_list (List[str]): list of all file paths
        chunk_size (int): size of chunk returned

    Returns:
        List[str]: list of chopped file_path_list with size equals to chuck_size
    """

    for i in range(0, len(file_path_list), chunk_size):
        yield file_path_list[i:i + chunk_size]

def send_all_requests_multiprocess(send_request_partial: functools.partial,
                                   file_path_list: List[str], config: Dict[str, Any]) -> float:
    """send all files in file_path_list for inference with multiprocess defined by user in config
    Args:
        send_request_partial (functools.partial): a partial function that send requests
        file_path_list (List[str]): list of all file paths
        config (Dict[str, Any]): configurations for stress test

    Returns:
        float: how long does it take to send all requests

    """
    num_workers: int = config["num_workers"]
    chunk_size: int = len(file_path_list) // num_workers
    time_start: float = time.time()
    # cut the big file_path_list into pieces and each process deal with one piece
    with Pool(num_workers) as p:
        p.map(send_request_partial, split(file_path_list, chunk_size))
    request_time: float = time.time() - time_start
    logging.info(f"All requests have been sent through.\n"
                 f"Total number of requests: {len(file_path_list)}.\n"
                 f"Total time of send requests: {request_time :.2f}s.\n"
                 f"Average time for sending 1 request: {(request_time / len(file_path_list)):.2f}s\n")
    return  request_time

def check_last_result(api: api_helper.InferenceServerAPIHelper,
                      model_id_to_last_cor_id: Dict[int, uuid.UUID],
                      config: Dict[str, Any]) -> float:
    """this function is meant to be used after sending all the inference request. It will find the correlation id
        of the last request and continuously checking the status until inference is done

    Args:
        api (api_helper.InferenceServerAPIHelper): class instance contains API methods
        model_id_to_last_cor_id (Dict[int, uuid.UUID]): dictionary where keys are model id for each model,
            values are the correlation id of the last request sent to this model
        config (Dict[str, Any]): configurations for stress test

    Returns
        float: how long does it take to finish all inference since all requests are sent

    """
    start_time: float = time.time()
    # flag to remember do we inference on models or model network
    is_model_network: bool = True if config["model_network_ids"] else False
    logging.info("start to fetch the last inference result...")
    while True:
        model_ids: List[int] = list(model_id_to_last_cor_id.keys())
        if len(model_ids) == 0:
            inference_time: float = time.time() - start_time
            logging.info(f"{inference_time:.2f} seconds have passed, inference complete")
            return inference_time

        # check the last inference request sent to each model
        for model_id in model_ids:
            if is_inference_complete(api, model_id, model_id_to_last_cor_id[model_id], is_model_network):
                # if it is done, remove it from the dictionary
                model_id_to_last_cor_id.pop(model_id)
        # do it once per second
        time.sleep(1)
        logging.info(f"{(time.time() - start_time):.2f} seconds have passed, still fetching...")

def is_inference_complete(api: api_helper.InferenceServerAPIHelper,
                          model_id: int, cor_id: uuid.UUID, is_model_network: bool) -> bool:
    """check whether an inference is complete

    Args:
        api (api_helper.InferenceServerAPIHelper): class instance contains API methods
        model_id (int): model id of the request sent
        cor_id (uuid.UUID): uuid of the inference request
        is_model_network (bool): whether the inference is done on model or model network

    Return:
        bool: whether the inference has completed or not

    """
    # model and model network uses different APIs to get response
    if is_model_network:
        network_id: int = model_id
        response: Dict[str, Any] = api.get_model_network_responses(network_id, cor_id)
        return api.is_model_network_inference_complete(response)
    else:
        response: Dict[str, Any] = api.get_model_responses(model_id, cor_id)
        return api.is_model_inference_complete(response)

def send_request_fill_pool(api,
                           config: Dict[str, Any],
                           file_path_list: List[str],
                           current_file_index: int,
                           cor_id_to_model_id: Dict[uuid.UUID, int]) -> Tuple[int, Dict[uuid.UUID, int]]:
    """keep sending the request until reaches pool size or reaches total inference number in config
        used for result_pulling_type = fixed_pool

    Args:
        api (api_helper.TrellisIntelligencePlatformAPI): class instance contains API methods
        config (Dict[str, Any]): configurations for stress test
        file_path_list (List[str]): list of all file paths
        current_file_index (int): the index of file_path_list that points to which file to send inference request
        cor_id_to_model_id (Dict[uuid.UUID, int]): dictionary to keep track of the requests in the pool

    Return:
        Tuple[int, Dict[uuid.UUID, int]]: a tuple of the updated current_file_index and updated cor_id_to_model_id

    """
    num_file: int = len(file_path_list)
    pool_size: int = config["pool_size"]
    request_spread_type: str = config["request_spread_type"]
    dataset_ids: Optional[List[int]] = config["dataset_ids"]
    explain: str = config["explain"]
    # flag to remember do we inference on models or model network
    is_model_network: bool = False
    # for convenient, no matter inference is done on model or model network, let's just call the variable model_ids
    if config["model_ids"]:
        model_ids: List[int] = config["model_ids"]
    else:
        model_ids: List[int] = config["model_network_ids"]
        is_model_network: bool = True

    # keep sending request when:
    # 1. not all requests are sent already
    # 2. the pool is not filled yet
    while current_file_index < num_file and len(cor_id_to_model_id) < pool_size:
        model_id: int = choose_model_id(request_spread_type, model_ids, current_file_index)
        cor_id = send_single_request(api, model_id, file_path_list[current_file_index],
                                     is_model_network, dataset_ids, explain)
        # update the current requests in the pool and the current pointer to file list
        cor_id_to_model_id[cor_id]: int = model_id
        current_file_index: int = current_file_index + 1

    return current_file_index, cor_id_to_model_id


def update_pool(api, cor_id_to_model_id: Dict[uuid.UUID, int], config: Dict[str, Any]) -> int:
    """ check each request within the current monitored pool, remove the ones that have done inference

    Args:
        api (api_helper.TrellisIntelligencePlatformAPI): class instance contains API methods
        cor_id_to_model_id (Dict[uuid.UUID, int]): dictionary to keep track of the requests in the pool
        config (Dict[str, Any]): configurations for stress test

    Return:
        int: the number of inferences that are done in the current pool

    """
    inference_count: int = 0
    cor_id_to_model_id_copy: Dict[uuid.UUID, int] = copy.deepcopy(cor_id_to_model_id)
    is_model_network: bool = True if config["model_network_ids"] else False
    for cor_id, model_id in cor_id_to_model_id_copy.items():
        if is_inference_complete(api, model_id, cor_id, is_model_network):
            # remove the correlation id that has done inference
            cor_id_to_model_id.pop(cor_id)
            inference_count: int = inference_count + 1
    logging.info(f"{inference_count} inferences have done")
    return inference_count


def inference_fix_pool(api,
                       file_path_list: List[str],
                       config: Dict[str, Any],
                       file_loader: InferenceFileLoader) -> None:
    """maintain a fixed number of request to monitor. If some requests have inference done, remove them.
        if size of pool is smaller than it meant to be, fill the pool by sending more request.
        time between each monitor can be adjusted in config

    Args:
        api (api_helper.TrellisIntelligencePlatformAPI): class instance contains API methods
        file_path_list (List[str]): list of all file paths
        config (Dict[str, Any]): configurations for stress test
        file_loader (InferenceFileLoader): file loader used for file preparation

    """
    num_file: int = len(file_path_list)
    current_file_index: int = 0
    total_inference_count: int = 0
    start_time: float = time.time()
    # dictionary to keep track of each request sent
    cor_id_to_model_id: Dict[uuid.UUID, int] = {}
    pool_size: int = config["pool_size"]
    while total_inference_count < num_file:
        # keep sending requests until pool is full or no more files left
        current_file_index, cor_id_to_model_id = send_request_fill_pool(api, config, file_path_list, current_file_index,
                                                                        cor_id_to_model_id)
        logging.info(f"{current_file_index} / {num_file} requests in total have been sent")
        logging.info(f"{len(cor_id_to_model_id)} requests are in the pool under monitoring")
        # update the pool, remove correlation ids that finish inference
        current_iteration_inference_count: int = update_pool(api, cor_id_to_model_id, config)
        if current_iteration_inference_count > pool_size * 0.8:
            logging.warning("Number of inference done in the current iteration exceeds 80% of requests in the pool. "
                            "Consider increasing pool size or change inference method to pull last result only "
                            "to get more accurate benchmark result")
        total_inference_count: int = total_inference_count + current_iteration_inference_count
        end_time: float = time.time()
        logging.info(f"having {len(cor_id_to_model_id)} left in the pool within the current iteration")
        logging.info(f"total inference progress: {total_inference_count} / {num_file}")
        logging.info(f"{(end_time - start_time):.2f}s have elapsed since starting")
        file_loader.evaluate_performance_fix_pool(total_inference_count, (end_time - start_time))

        # how often to monitor the pool
        time.sleep(config["pool_monitor_time"])


def warm_up(api, file_path_list: List[str], config: Dict[str, Any]) -> None:
    """do some inference on each model and wait until the Last inference sent is done to warm up the model
    Note: before this step, model should be manually placed evenly across all GPU to optimise performance

    Args:
        api (api_helper.TrellisIntelligencePlatformAPI): class instance contains API methods
        file_path_list (List[str]): list of all file paths
        config (Dict[str, Any]): configurations for stress test

    """
    # for convenient, no matter inference is done on model or model network, let's just call the variable model_ids
    if config["model_ids"]:
        model_ids: List[int] = config["model_ids"]
    else:
        model_ids: List[int] = config["model_network_ids"]

    # send the first file for inference to each model twice for warm up
    warm_up_time: int = AutoInferenceConstant.WARM_UP_TIME
    warm_up_files: List[str] = [file_path_list[0]] * warm_up_time * len(model_ids)

    # make a copy of config, change request_spread_type to even
    config_copy: Dict[str, Any] = copy.deepcopy(config)
    config_copy["request_spread_type"]: str = AutoInferenceConstant.EVEN
    logging.info("#---------start warm up---------#")
    _, model_id_to_cor_id = send_all_requests(warm_up_files, api, config_copy)
    check_last_result(api, model_id_to_cor_id, config_copy)
    logging.info("#---------warm up finished---------#")

def main(api,
         file_path_list: List[str],
         config: Dict[str, Any],
         file_loader: InferenceFileLoader) -> None:
    """perform test based on config given

    Args:
        api (api_helper.TrellisIntelligencePlatformAPI): class instance contains API methods
        file_path_list (List[str]): list of all file paths
        config (Dict[str, Any]): configurations for stress test
        file_loader (InferenceFileLoader): file loader used for file preparation

    """
    result_pulling_type: str = config["result_pulling_type"]
    # partial function prepared for multiprocess requests sending
    # make sure that the time spent for sending the request won't be the bottleneck of our inference speed
    send_request_partial: functools.partial = functools.partial(send_all_requests, api=api, config=config)

    if result_pulling_type not in AutoInferenceConstant.ALL_TYPE:
        # no inference result pulling at all, just keep sending the request
        logging.warning(f"inference type is not one of {AutoInferenceConstant.ALL_TYPE}, "
                        f"no inference results will be pulled,only inference requests will be sent")
        send_all_requests_multiprocess(send_request_partial, file_path_list, config)

    elif result_pulling_type == AutoInferenceConstant.LAST_RESULT_ONLY:
        # send all request
        request_time: float = send_all_requests_multiprocess(send_request_partial, file_path_list, config)

        # send an addition (len(model_ids or model_network_ids)) inference
        # so that we know exactly which ones are sent last
        time_start: float = time.time()
        if config["model_ids"]:
            additional_inference_num: int = len(config["model_ids"])
        else:
            additional_inference_num: int = len(config["model_network_ids"])
        additional_inference_file: List[str] = [file_path_list[0]] * additional_inference_num

        _, model_id_to_last_cor_id = send_request_partial(additional_inference_file)
        request_time: float = request_time + (time.time() - time_start)
        total_inference_num: int = config['total_inference_num'] + additional_inference_num
        logging.info(f"Additional {additional_inference_num} inference(s) are sent, 1 to each model. \n"
                     f"Total request number: {total_inference_num}\n"
                     f"Total request time: {request_time}")

        # now all inference requests are sent, and we know exactly which ones are sent last, start checking last ones
        inference_time: float = check_last_result(api, model_id_to_last_cor_id, config)
        total_inference_time: float = request_time + inference_time
        file_loader.evaluate_performance_last_result_only(additional_inference_file, total_inference_time)

    elif result_pulling_type == AutoInferenceConstant.FIXED_POOL:
        # the number of requests that in the status of "sent but not done" is maintained to be fixed.
        # hard to use multiple process to send requests that keep filling the pool
        # so instead just hint the user when the pool is too empty
        inference_fix_pool(api, file_path_list, config, file_loader)
    else:
        # TODO: more test types in the future if needed
        pass


if __name__ == '__main__':
    """
    Before using the script, make sure all the inference models are started and evenly placed across GPUs
    Currently support only the following test cases
    1. upload 1 image at a time to a single model until total_image_num reaches.
    2. upload 1 image at a time to a single model, 
       upload all images in input_dir repeatedly until total_image_num reaches.
       Pull the result from the last inference and record the total time
    3. upload 1 image at a time to a single model, 
       upload all images in input_dir repeatedly until total_image_num reaches.
       Having exactly inference_pool_size images go through inference at the same time. 
       When 1 image finishes inference, pull the result and add a image into the pool
    
    Usage:
    python stress_test.py [-c JSON_CONFIG_PATH]. Config file sample is sitting in the same directory as this script
    
    Parameters in stress_test_config.json explained:
      - "header" (required): affect logging information but does not affect functionality 
      - "server" (required): server end point
      - "username" (required): username of the server
      - "password" (required): password of the server
      - "inference_server_api_version" (required): version of inference server API. Use "2_50_0" at the moment.
          If API calls fail, contact R&D team to fix.
      - "input_path_or_dir" (required): either a single file path or a directory contains multiple files to inference
      - "result_pulling_type" (optional): currently support "last_result_only" and "fixed_pool".
          last_result_only: send all request then monitor only the last inference.
          fixed_pool: maintain the number of requests which are sent but not done inference to a fixed number.
            If one request is done inference, remove it from the pool; if number of request is less than pool size,
            send more requests to fill the pool. Test is finished when all requests are done inference
          Putting other values will only send inference requests and not getting response back
      - "model_ids" (optional): list of model ids. Provide when model_network_ids = null
      - "model_network_ids" (optional): list of model network ids. Provide when model_ids = null
      - "dataset_ids" (optional): list of dataset ids. For POI model test, this parameter is required
      - "inference_file_types" (required): list of strings, currently support "image", "video", "audio"
      - "total_inference_num" (required): total number of inference requests to send
      - "request_spread_type" (optional): currently support "even" and "random". 
          even: inference requests are evenly spread across all models
          random: inference requests are randomly assign to models
          All other values are equivalent to "random"
      - "pool_size" (optional): required when result_pulling_type == "fixed_pool"
      - "num_workers" (required): number of parallel processes when sending inference requests
      - "pool_monitor_time" (optional): required when result_pulling_type == "fixed_pool"
          The time gap between updating the pool
          Note that this value is a float, for example if the gap wanted is 2 seconds, please input 2.0
      - "explain" (optional): whether to turn explain on or not in inference, default to "true"
          true: explain
          false: not explain. This will save significant amount of GPU and inference time in some cases
          
    """
    logging.set_verbosity(logging.INFO)

    parser = argparse.ArgumentParser(description="Inference via Trellis API. Makes API calls specified by config")
    parser.add_argument("-c", '--json_config_path',
                        type=str,
                        default="stress_test_config.json",
                        help="the configration file path")
    args: argparse.Namespace = parser.parse_args()

    # read config file as a python dictionary
    config_path: str = args.json_config_path
    with open(config_path, "rb") as config_file:
        config: Dict[str, Any] = json.load(config_file)

    # validate input configs
    config: Dict[str, Any] = validate_config(config)

    # credential information based on inference server version
    # 2_50_0 can be used as long as no modifications are made in API parameters in newer version
    # in the case of API parameters are changed, a new class should be created in api_helper.py to adapt
    version: str = config["inference_server_api_version"]
    class_name: str = "InferenceServerAPIHelper" + version
    api_class: type = getattr(__import__('api_helper'), class_name)

    api: api_helper.InferenceServerAPIHelper = api_class(config["header"],
                                                         config["server"],
                                                         config["username"],
                                                         config["password"])
    api.log_in()

    logging.info('Configs:')
    for k, v in config.items():
        logging.info(f'    {k} : {v}')

    # prepare a file loader based on inference file type
    # currently only support single file type like ["image"], does not support mixed type like ["image", "audio"]
    file_loader: InferenceFileLoader = get_file_loader(config["input_path_or_dir"],
                                                       config["total_inference_num"],
                                                       config["inference_file_types"])

    all_file_paths: List[str] = file_loader.file_path_list
    # do a few inference on each model to warm up before start to benchmark
    warm_up(api, all_file_paths, config)
    # start testing
    main(api, all_file_paths, config, file_loader)
