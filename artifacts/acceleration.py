"""
This module contains a class for Acceleration data type
The class defines methods that could be applied on the acceleration object
"""

import math
import json
import numpy as np
from functools import reduce
from typing import Optional, Literal, Union
import logging

logger = logging.getLogger("AI")


class MeasurementTime:
    """Object to model the measurement time data"""

    def __init__(self, measurement_time: str):
        self.measurement_time = measurement_time
        self.year = self.get_year()
        self.month = self.get_month()
        self.day = self.get_day()
        self.hour = self.get_hour()
        self.minute = self.get_minute()
        self.second = self.get_second()
        self.upto_hour = self.get_upto_hour()

    def get_year(self):
        return self.measurement_time[:4]

    def get_month(self):
        return self.measurement_time[4:6]

    def get_day(self):
        return self.measurement_time[6:8]

    def get_hour(self):
        return self.measurement_time[8:10]

    def get_minute(self):
        return self.measurement_time[10:12]

    def get_second(self):
        return self.measurement_time[12:14]

    def get_upto_hour(self):
        upto_hour = self.year + self.month + self.day + self.hour
        return upto_hour


class Acceleration:
    """Models the acceleration data for one cycle and extracts features from it"""

    def __init__(
        self,
        acc_data: str,
        approved_ct: int,
        FIRST_SECTION_PROP: float,
        BIN_WIDTH_FIRST_SECTION: float,
        BIN_WIDTH_SECOND_SECTION: float,
    ):
        self.acc_data = acc_data
        self.approved_ct = approved_ct
        self.acc_feature = self.get_acc_feature(
            BIN_WIDTH_FIRST_SECTION=BIN_WIDTH_FIRST_SECTION,
            BIN_WIDTH_SECOND_SECTION=BIN_WIDTH_SECOND_SECTION,
            FIRST_SECTION_PROP=FIRST_SECTION_PROP,
        )["acc"]
        self.time_feature = self.get_acc_feature(
            BIN_WIDTH_FIRST_SECTION=BIN_WIDTH_FIRST_SECTION,
            BIN_WIDTH_SECOND_SECTION=BIN_WIDTH_SECOND_SECTION,
            FIRST_SECTION_PROP=FIRST_SECTION_PROP,
        )["time"]

    def parse_acc_to_dict(self) -> dict[str, list]:
        acc_data = self.acc_data
        acc_dict = json.loads(acc_data)
        # Get the time and acc values
        time = list(map(lambda x: float(x["time"]), acc_dict))
        acc = list(map(lambda x: float(x["value"]), acc_dict))
        return {"time": time, "acc": acc}

    def get_acc_feature(
        self,
        BIN_WIDTH_FIRST_SECTION: float,
        BIN_WIDTH_SECOND_SECTION: float,
        FIRST_SECTION_PROP: float,
    ) -> dict:
        """Method that extracts the acceleration feature vector
        from the accleration data

        Args:
            BIN_WIDTH_FIRST_SECTION (float): bin width in seconds for the duration (0, approved_ct/2]
            BIN_WIDTH_SECOND_SECTION (float): bin width in seconds for the duration (approved_ct/2, max(time)]
            FIRST_SECTION_PROP (float): the proportion of first section of the approved cycle time

        Returns:
            dict: the sampled acceleration data and the time filters
        """

        approved_ct = self.approved_ct
        if approved_ct:
            approved_ct_sec = approved_ct / 10
            CUT_SECTION_SEC = approved_ct_sec * FIRST_SECTION_PROP
        else:
            # No approved cycle time for mold
            CUT_SECTION_SEC = 10
            error_msg = (
                "Warning: Mold doesn't have approved CT."
                "\nDefault CT of 20s is used to extract features."
                "\n---- ---- ---- ---- ---- ---- ---- ---- ----"
            )
            # logger.error(
            #     error_msg,
            #     stack_info=False,
            # )

        # Get the time and acceleration arrays
        acc_dict = self.parse_acc_to_dict()
        time, acc = acc_dict["time"], acc_dict["acc"]

        # Sort the data w.r.t. measurement time
        time_acc_pair = list(zip(time, acc))
        time_acc_pair_sorted = sorted(time_acc_pair, key=lambda x: x[0])

        MAX_TIME = max(time)
        BIN_WIDTH_FIRST_SECTION = (
            BIN_WIDTH_FIRST_SECTION
            if CUT_SECTION_SEC >= BIN_WIDTH_FIRST_SECTION
            else (CUT_SECTION_SEC / 10)
        )  # bin width for the section before CUT_SECTION_SEC

        binned_acc_data = []
        binned_time_data = []
        # First half section of cycle
        for t_boundary in np.arange(
            start=0,
            stop=CUT_SECTION_SEC if MAX_TIME > CUT_SECTION_SEC else MAX_TIME,
            step=BIN_WIDTH_FIRST_SECTION,
        ):
            time_min = t_boundary
            time_max = t_boundary + BIN_WIDTH_FIRST_SECTION
            # Filter the data points where (time_min <= time < time_max)
            filtered_data = list(
                filter(
                    lambda pair: (pair[0] >= time_min) and (pair[0] < time_max),
                    time_acc_pair_sorted,
                )
            )
            # Calculate the avg acceleration value of the filtered data points
            filtered_acc = [pair[1] for pair in filtered_data]

            if filtered_acc:
                acc_avg = np.mean(filtered_acc)
            else:
                acc_avg = 0
            binned_acc_data.append(acc_avg)
            binned_time_data.append((time_min + time_max) / 2)

        # Second half section of cycle
        if MAX_TIME > CUT_SECTION_SEC:  # check there is data in second half
            for t_boundary in np.arange(
                start=CUT_SECTION_SEC,
                stop=MAX_TIME + BIN_WIDTH_SECOND_SECTION,
                step=BIN_WIDTH_SECOND_SECTION,
            ):
                time_min = t_boundary
                time_max = t_boundary + BIN_WIDTH_SECOND_SECTION
                # Filter the data points where (time_min < time <= time_max)
                filtered_data = list(
                    filter(
                        lambda pair: (pair[0] >= time_min) and (pair[0] < time_max),
                        time_acc_pair_sorted,
                    )
                )
                # Calculate the avg acceleration value of the filtered data points
                filtered_acc = [pair[1] for pair in filtered_data]

                if filtered_acc:
                    acc_avg = np.mean(filtered_acc)
                else:
                    acc_avg = 0
                binned_acc_data.append(acc_avg)
                binned_time_data.append((time_min + time_max) / 2)

        return {"time": binned_time_data, "acc": binned_acc_data}


class AccRecord:
    """Represents a single acceleration measurement record"""

    def __init__(
        self,
        id: int,
        measurement_time: str,
        measurement_hour: str,
        acc_feature: list,
        sim_metric: Optional[float],
        sim_metric_hr: Optional[float],
        proc_changed: Literal[None, 1, 0],
    ):
        self.id = id
        self.measurement_time = measurement_time
        self.measurement_hour = measurement_hour
        self.acc_feature = acc_feature
        self.sim_metric = sim_metric
        self.sim_metric_hr = sim_metric_hr
        self.proc_changed = proc_changed

    def __repr__(self):
        str_repr = str(
            {
                "id": self.id,
                "measurement_time": self.measurement_time,
                "measurement_hour": self.measurement_hour,
                "acc_feature": self.acc_feature,
                "sim_metric": self.sim_metric,
                "sim_metric_hr": self.sim_metric_hr,
                "proc_changed": self.proc_changed,
            }
        )
        return str_repr


def pad_zeroes(
    input: Union[list, np.ndarray], expected_len: int
) -> Union[list, np.ndarray]:
    """Adds trailing zeros to a list/array
    Args:
        input (Union[list, np.ndarray]): input list/array
        expected_len (int): expected length of the list/array
    Returns:
        Union[list, np.ndarray]: zero padded list/array
    """

    if isinstance(input, list):
        input_c: list = input.copy()
        while len(input_c) < expected_len:
            input_c.append(0)
    elif isinstance(input, np.ndarray):
        input_c: np.ndarray = np.copy(input)
        while len(input_c) < expected_len:
            input_c = np.append(input_c, 0)
    return input_c


# Stand alone function because it is imported in the ai_pc.py file
def calculate_avg_acc_feature(
    data: list[AccRecord], avg_method: Literal["mean", "exp_mean"] = "mean", **kwargs
) -> Optional[np.ndarray]:
    """Calculates the average of the acc features extracted
    Args:
        data (list[AccRecord]): list of dict containing id,
                                measurement_time, measurement_hour,
                                and acc_feature
        avg_method (Literal["mean", "exp_mean"]): the method used when taking the
                                                  the average of the features
    Returns:
        np.ndarray: it is the average acceleration feature for a given hour
    """

    # Acceleration data needs to be present
    if data:
        # Order data with respect to time
        data = sorted(data, key=lambda x: x.measurement_time, reverse=False)

        len_acc_features = np.array([len(record.acc_feature) for record in data])
        max_acc_feature_len = len_acc_features.max()
        # Pad acc features with zero
        acc_feature_all = np.asarray(
            [
                pad_zeroes(input=record.acc_feature, expected_len=max_acc_feature_len)
                for record in data
            ]
        )
        # Calculate the average feature
        if avg_method == "mean":
            avg_acc_feature = acc_feature_all.mean(axis=0)

        elif avg_method == "exp_mean":
            n_samples = len(data)
            alpha = 1 / n_samples
            factors = [(alpha) * (1 - alpha) ** k for k in range(n_samples)]
            sum_factors = sum(factors)
            factors_norm = [(x / sum_factors) for x in factors]

            # Set the order of the weights
            weight_order = kwargs.get("weight_order", None)
            if weight_order == "increasing":
                factors_norm = sorted(factors_norm, reverse=False)
            elif weight_order == "decreasing":
                # No need to reorder the factors
                pass
            factors_norm = np.array(factors_norm).reshape(len(factors_norm), 1)
            # The features are summed across each dimension after taking the weights
            avg_acc_feature = (factors_norm * acc_feature_all).sum(axis=0)

    else:
        avg_acc_feature = None
    return avg_acc_feature


class AccSummary:
    """Class that represent the summary of multiple acceleration measurements"""

    def __init__(
        self,
        acc_data: list[Optional[AccRecord]],
        avg_method: Literal["mean", "exp_mean"],
        **kwargs
    ):
        # Remove any None values in the list
        acc_data_filtered = AccSummary.remove_none_from_list(acc_data)
        weight_order = kwargs.get("weight_order")
        self.acc_data = acc_data_filtered
        self.num_records = len(acc_data_filtered)
        self.avg_acc_feature = self.get_avg_acc_feature(
            avg_method=avg_method, weight_order=weight_order
        )

    def __repr__(self):
        str_repr = str(
            {
                "acc_data": self.acc_data,
                "num_records": self.num_records,
                "avg_acc_feature": self.avg_acc_feature,
            }
        )
        return str_repr

    def get_avg_acc_feature(
        self,
        avg_method=Literal["mean", "exp_mean"],
        weight_order=Literal["increasing", "decreasing"],
    ) -> Optional[np.ndarray]:
        avg_acc_feature = calculate_avg_acc_feature(
            data=self.acc_data, avg_method=avg_method, weight_order=weight_order
        )
        return avg_acc_feature

    @staticmethod
    def remove_none_from_list(acc_records: list[Optional[AccRecord]]):
        acc_records_filtered = [
            acc_record for acc_record in acc_records if acc_record is not None
        ]
        return acc_records_filtered


class AccHourlySummary(AccSummary):
    """Represents the summary of a 1 hour acceleration data"""

    def __init__(
        self,
        hour: str,
        exists: bool,
        sim_metric_hr: Optional[float],
        proc_changed: Literal[None, 0, 1],
        to_return: bool,
        acc_data: list[AccRecord],
        avg_method: Literal["mean", "exp_mean"],
        **kwargs
    ):
        weight_order = kwargs.get("weight_order")
        super().__init__(
            acc_data=acc_data, avg_method=avg_method, weight_order=weight_order
        )
        # Additional attributes
        self.hour = hour
        self.exists = exists
        self.sim_metric_hr = sim_metric_hr
        self.proc_changed = proc_changed
        self.to_return = to_return

    def __repr__(self):
        str_repr = str(
            {
                "hour": self.hour,
                "exists": self.exists,
                "sim_metric_hr": self.sim_metric_hr,
                "proc_changed": self.proc_changed,
                "to_return": self.to_return,
                "acc_data": self.acc_data,
                "num_records": self.num_records,
                "avg_acc_feature": self.avg_acc_feature,
            }
        )
        return str_repr
