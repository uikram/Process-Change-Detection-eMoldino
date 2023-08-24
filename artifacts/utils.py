"""This module contains utility function for process change
"""
import numpy as np
from scipy.spatial.distance import cosine
from datetime import datetime, timedelta
import logging
from typing import Optional, Union, Literal

from artifacts.acceleration import (
    AccRecord,
    pad_zeroes,
)


logger = logging.getLogger("AI")


def get_hourly_similarity_metric(
    records_all: list[AccRecord],
) -> dict[str, Optional[float]]:
    """Get the hourly similarity metric from previous inferences
    Args:
        records_all (list[AccRecord]): list of  AccRecord instances
    Returns:
        dict[str : Optional[float]]: hourly sim. metric with the hr
                                     as key in the dictionary
    """
    unique_hours = list({record.measurement_hour for record in records_all})
    unique_hours = sorted(unique_hours, reverse=False)

    # get sim_metric_hr from previous inference
    sim_metric_hr_all = {}
    for hour in unique_hours:
        unique_sim_metric_hr = list(
            {
                record.sim_metric_hr
                for record in records_all
                if (record.measurement_hour) == hour
                and (record.sim_metric_hr is not None)
            }
        )
        # There should only be one unique similarity metric
        if len(unique_sim_metric_hr) > 1:
            logger.error(
                "Warning: similarityMetricHr has multiple values for single hour",
                stack_info=False,
            )
            sim_metric_hr = unique_sim_metric_hr[0]

        elif len(unique_sim_metric_hr) == 1:
            sim_metric_hr = unique_sim_metric_hr[0]
        else:
            sim_metric_hr = None

        sim_metric_hr_all[hour] = sim_metric_hr

    return sim_metric_hr_all


def generate_pairs(lst: list) -> list[tuple]:
    """Generates list of consecutive pairs given a list of objects
    Args:
        lst (list): input list
    Returns:
        list[tuple]: list of consecutive pairs
    """
    return [(lst[i], lst[i + 1]) for i in range(len(lst) - 1)]


def generate_n_pairs(lst: list, n: int) -> list[list]:
    """Generates a list of items of size n given a list
    Args:
        lst (list): original list
        n (int): size of the sublist
    Returns:
        list[list]: a list of list of the sublists
    Doc test:
    >>> generate_n_pairs(lst=[1, 2, 3, 4], n=2)
    >>> [[None, 1], [1, 2], [2, 3], [3, 4]]
    """
    all_pairs = []
    # if n <= len(lst):
    for idx in range(len(lst)):
        idx_start = idx - n + 1
        idx_end = idx + 1

        if idx_start < 0:
            pairs = lst[0:idx_end]
            pairs = [None for _ in range(n - len(pairs))] + pairs
            all_pairs.append(pairs)
        else:
            all_pairs.append(lst[idx_start:idx_end])

    return all_pairs


def extract_hourly_data(data: list[AccRecord], target_hr: str) -> list[AccRecord]:
    """Extracts acceleration data for the specified hour

    Args:
        data (list[AccRecord]): list of dict containing id, measurement_time,
                                      measurement_hour, and acc_feature
        target_hr (str): hr for which data would be extracted for

    Returns:
        list[AccRecord]: filtered data for the specific hour
    """
    data_filtered: list[AccRecord] = list(
        filter(lambda record: record.measurement_hour == target_hr, data)
    )
    data_filtered_sorted = sorted(
        data_filtered, key=lambda record: record.measurement_time, reverse=False
    )
    return data_filtered_sorted


def add_hours(
    start_time: str, hrs_to_add: float, input_fmt: str, output_fmt: str
) -> str:
    """Adds hrs_to_add hours on start_time
    Args:
        start_time (str): reference time
        hrs_to_add (float): additional hours to add
        input_fmt (str):  the format of the start_time
        output_fmt (str): the format of the outout when extracted
    Returns:
        str: start_time + hrs_to_add
    """
    start_time_dt = datetime.strptime(start_time, input_fmt)
    shifted_time_dt = start_time_dt + timedelta(hours=hrs_to_add)
    shifted_time_str = shifted_time_dt.strftime(output_fmt)
    "%Y%m%d%H"

    return shifted_time_str


def subtract_hours(
    start_time: str, hrs_to_sub: float, input_fmt: str, output_fmt: str
) -> str:
    """Subtracts hrs_to_add hours from start_time
    Args:
        start_time (str): reference time
        hrs_to_sub (float): additional hours to subtract
        input_fmt (str):  the format of the start_time
        output_fmt (str): the format of the outout when extracted
    Returns:
        str: start_time - hrs_to_sub
    """
    shifted_time_str = add_hours(
        start_time=start_time,
        hrs_to_add=(-1 * hrs_to_sub),
        input_fmt=input_fmt,
        output_fmt=output_fmt,
    )
    return shifted_time_str


def cos_similarity_metric(
    vector_1: Union[list, np.ndarray], vector_2: Union[list, np.ndarray]
) -> float:
    """Calculates the cosine similarity between two vectors
    Args:
        vector_1 (np.ndarray): first acc feature vector
        vector_2 (np.ndarray): second acc feature vector
    Returns:
        float: cosine product
    """
    expected_len = len(vector_1) if len(vector_1) >= len(vector_2) else len(vector_2)
    # Zero pad the vectors so that they could have the same dimensions
    vector_1 = pad_zeroes(input=vector_1, expected_len=expected_len)
    vector_2 = pad_zeroes(input=vector_2, expected_len=expected_len)
    # Calculate the cosine distance
    cosine_distance = cosine(u=vector_1, v=vector_2)
    return 1 - cosine_distance


def generate_hrs_template(
    start_hr: str, target_hrs_len: int = 4, max_distance: int = 2
) -> dict:
    """Based on the start_hr, generates a template for hours
       of data fethed from MMS
    Args:
        start_hr (str): reference hour when fetching data
        target_hrs_len (int, optional): target duration of fetched data.
                                        Defaults to 4.
        max_distance (int, optional): the additional hrs of data fetched at
                                      the beginning and end of the fetched data
                                      Defaults to 2.
    Returns:
        dict: hour template and an indictor of whether the result will be returned
    """
    start_hr_dt = datetime.strptime(start_hr, "%Y%m%d%H")

    total_hrs = target_hrs_len + (2 * max_distance)
    hrs_template = []
    for hr in range(total_hrs):
        next_hr_dt = start_hr_dt + timedelta(hours=hr)
        next_hr_str = datetime.strftime(next_hr_dt, "%Y%m%d%H")
        hrs_template.append(next_hr_str)

    # Create boolean to identify if analysis should be returned
    padded_hrs = [False for _ in range(max_distance)]
    to_return = padded_hrs + [True for _ in range(target_hrs_len)] + padded_hrs

    return {"hrs_template": hrs_template, "to_return": to_return}


def get_neighbor_record(
    data: list[AccRecord],
    curr_time: str,
    max_hr_dist: float,
    pos: int,
    orientation=Literal["before", "after"],
) -> Optional[AccRecord]:
    """Gets records in the vicinity of a time of interest

    Args:
        data (list[AccRecord]): list of AccRecord objs
        curr_time (str): time of interest to get neighbor records
        max_hr_dist(float): distance from current time
        pos (int): distance of the neighbor from curr_time
        orientation: whether the neighbor record is before or after curr_time

    Returns:
        Optional[AccRecord]: neighbor record to be returned
    """

    if orientation == "after":
        # Upper boundary of time within the max_hr_dist
        upper_boundary = add_hours(
            start_time=curr_time,
            hrs_to_add=max_hr_dist,
            input_fmt="%Y%m%d%H%M%S",
            output_fmt="%Y%m%d%H%M%S",
        )
        # Filter records that come after the curr_time
        data_next = [
            record
            for record in data
            if (
                record.measurement_time > curr_time
                and record.measurement_time <= upper_boundary
            )
        ]
        # sort with respect to time
        data_next_sorted = sorted(
            data_next, key=lambda x: x.measurement_time, reverse=False
        )

        try:
            pos_after = pos - 1
            record_neighbor = data_next_sorted[pos_after]
        except IndexError:
            record_neighbor = None

    elif orientation == "before":
        lower_boundary = subtract_hours(
            start_time=curr_time,
            hrs_to_sub=max_hr_dist,
            input_fmt="%Y%m%d%H%M%S",
            output_fmt="%Y%m%d%H%M%S",
        )
        data_prev = [
            record
            for record in data
            if (
                record.measurement_time < curr_time
                and record.measurement_time >= lower_boundary
            )
        ]
        data_prev_sorted = sorted(
            data_prev, key=lambda x: x.measurement_time, reverse=False
        )
        try:
            pos_before = -1 * (pos)
            record_neighbor = data_prev_sorted[pos_before]
        except IndexError:
            record_neighbor = None

        return record_neighbor


def get_n_neighbor_records(
    data: list[AccRecord],
    curr_time: str,
    max_hr_dist: float,
    n: int,
    orientation: Literal["before", "after"],
) -> list[Optional[AccRecord]]:
    """gets n neighbors before or after the current record
    Args:
        data (list[AccRecord]): list of acceleration records
        curr_time (str): current time
        max_hr_dist (float): max distance from current time to search for neighbors
        n (int): number of neighbors to extract
        orientation (Literal["before", "after"]): direction to search for records

    Returns:
        list[Optional[AccRecord]]: list of n neighbors fetched
    """
    data_neighbor = []
    for pos in range(n):
        neighbor_record = get_neighbor_record(
            data=data,
            curr_time=curr_time,
            max_hr_dist=max_hr_dist,
            pos=pos,
            orientation=orientation,
        )
        data_neighbor.append(neighbor_record)

    return data_neighbor
