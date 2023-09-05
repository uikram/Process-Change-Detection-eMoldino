"""
This module contains functions that calculate the similarity betwween 
acceleration data points and identify if a process change occured
"""

import numpy as np
from typing import Optional
import logging
import math
from datetime import datetime

from artifacts.acceleration import (
    Acceleration,
    MeasurementTime,
    AccRecord,
    AccSummary,
    AccHourlySummary,
    calculate_avg_acc_feature,
)
from artifacts.utils import (
    
    extract_hourly_data,
    cos_similarity_metric,
    generate_hrs_template,
    generate_n_pairs,
    add_hours,
    subtract_hours,
    get_neighbor_record,
    get_n_neighbor_records,
)

# Logger for errors
logger = logging.getLogger("AI")

# Decimal place for similarity metrics
DECIMAL_PLACE = 6


from artifacts.acceleration import AccRecord
from typing import Optional, Union, Literal
import logging
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
        sim_metric_hr_all[hour] = 0.988 #Dummy Value
        # sim_metric_hr_all[hour] = sim_metric_hr

    return sim_metric_hr_all



def get_hourly_summary(
    start_hr: str, records_all: list[AccRecord]
) -> dict[str, AccHourlySummary]:
    """Gets hourly similarity metrics
    Args:
        start_hr (str): the left hr boundary of the data fetched
        records_all (list[AccRecord]): list of AccRecord for all data fetched
    Returns:
        dict[str: AccHourlySummary]: dict of AccHourlySummary objects
    """
    # Get hour template
    template = generate_hrs_template(
        start_hr=start_hr, target_hrs_len=4, max_distance=2
    )
    hrs_template = template["hrs_template"]
    to_return = template["to_return"]

    # Get similarity metric for the current
    sim_metric_existing_hrs = get_hourly_similarity_metric(records_all=records_all)
    unique_hrs = list(sim_metric_existing_hrs.keys())

    # fmt: off
    # Comparison is made with prev hr or with the prev_prev hr
    pair_3_hrs = generate_n_pairs(lst=hrs_template, n=3)  # pair up the hours
    pair_3_return = generate_n_pairs(lst=to_return, n=3)  # pair up the return status
    
    all_hr_summary = {}
    for pairs in zip(pair_3_hrs, pair_3_return):
        # Extract hours and whether they should be returned
        prev_prev_hr, prev_hr, curr_hr = pairs[0] #This creates pairs like if pairs[0]= [1,2,3] then prev_prev_hr=1 , prev_hour=2, curr_hour=3
        prev_prev_rtrn, prev_rtrn, curr_rtrn = pairs[1]

        # Extract data for current hour
        data_curr_hr = extract_hourly_data(data=records_all, target_hr=curr_hr)

        # Intialize object for hourly acceleration summary
        sim_metric_hr = sim_metric_existing_hrs.get(curr_hr, None)
        acc_hr_summary = AccHourlySummary(
            hour=curr_hr,
            exists=True,
            sim_metric_hr=sim_metric_hr, #Similarity metric from previous inference
            proc_changed=0,
            to_return=curr_rtrn,
            acc_data=data_curr_hr,
            avg_method="exp_mean",
            weight_order="increasing"
        )

        # data for specified hour doesn't exist
        if curr_hr not in unique_hrs:
            # Update values for exists and sim_metric_hr
            acc_hr_summary.exists = False
            acc_hr_summary.sim_metric_hr = None
        # data for specififed hour exists
        else:
            acc_hr_summary.exists = True
            # prev and prev_prev hours can't exist for curr_hr
            # This is the first record in the template hours
            if (prev_hr is None) and (prev_prev_hr is None):
                # No need to update the acc_hr_summary
                pass 

            # Either prev or prev_prev hour exists in current data
            else:
                # Average acc feature for current hour
                avg_acc_feature_curr_hr = calculate_avg_acc_feature(
                    data=data_curr_hr,
                    avg_method="mean",
                )
                # prev_hr exists in data
                if prev_hr in unique_hrs:
                    # Filter data for prev_hr and calculate avg acc feature
                    data_prev_hr = extract_hourly_data(data=records_all, target_hr=prev_hr)
                    avg_acc_feature_prev_hr = calculate_avg_acc_feature(
                        data=data_prev_hr,
                        avg_method="mean", 
                    )
                    # Calculate similarity metric
                    sim_metric_hr = cos_similarity_metric(
                        vector_1=avg_acc_feature_prev_hr,
                        vector_2=avg_acc_feature_curr_hr
                    )
                    acc_hr_summary.sim_metric_hr = sim_metric_hr

                # prev_prev_hr exists in data
                elif prev_prev_hr in unique_hrs:
                    # Filter data for prev_prev_hr and calculate avg acc feature
                    data_prev_prev_hr = extract_hourly_data(data=records_all, target_hr=prev_prev_hr)
                    avg_acc_feature_prev_prev_hr = calculate_avg_acc_feature(
                        data=data_prev_prev_hr,
                        avg_method="mean",
                    )
                    # Calculate avg acc feature
                    sim_metric_hr = cos_similarity_metric(
                        vector_1=avg_acc_feature_prev_prev_hr,
                        vector_2=avg_acc_feature_curr_hr
                    )
                    acc_hr_summary.sim_metric_hr = sim_metric_hr

                # Both prev and prev_prev don't exist in the data 
                else:
                    acc_hr_summary.sim_metric_hr = 2

        # Append hourly summary
        all_hr_summary[curr_hr] = acc_hr_summary

    # fmt: on
    sorted_all_hr_summary = dict(sorted(all_hr_summary.items()))
    return sorted_all_hr_summary


def identify_process_change(
    all_hr_summary: dict[str, AccHourlySummary], THRESHOLD_SIM_METRIC: float
) -> dict[str, AccHourlySummary]:
    """Identifies if process change occurs from hourly summary data
    Args:
        dict[str,AccHourlySummary]: summary of hourly acceleration data
    Returns:
        dict[str,AccHourlySummary]: modified summary to indicate if there is process change
    """
    THRESHOLD_NUM_RECORDS = 0.5

    for curr_hr, sum_curr_hr in all_hr_summary.items():
        # Get prev and next hour data
        next_hr = add_hours(
            start_time=curr_hr,
            hrs_to_add=1,
            input_fmt="%Y%m%d%H",
            output_fmt="%Y%m%d%H",
        )
        prev_hr = subtract_hours(
            start_time=curr_hr,
            hrs_to_sub=1,
            input_fmt="%Y%m%d%H",
            output_fmt="%Y%m%d%H",
        )
        sum_next_hr = all_hr_summary.get(next_hr, None)
        sum_prev_hr = all_hr_summary.get(prev_hr, None)
        # If data for prev hour does not exist, set the prev_prev_hr as prev_hr
        if (not sum_prev_hr) or (not sum_prev_hr.exists):
            # prev_prev_hr = subtract_hours(start_hr=curr_hr, hrs_to_sub=2)
            prev_prev_hr = subtract_hours(
                start_time=curr_hr,
                hrs_to_sub=2,
                input_fmt="%Y%m%d%H",
                output_fmt="%Y%m%d%H",
            )
            sum_prev_hr = all_hr_summary.get(prev_prev_hr, None)

        # Summary exists for the prev and next hr
        if sum_prev_hr and sum_next_hr:
            # There should be data for current, prev and next hrs
            if sum_curr_hr.exists and sum_prev_hr.exists and sum_next_hr.exists:
                if (
                    (sum_curr_hr.sim_metric_hr is not None)  # First ever data
                    and (sum_curr_hr.sim_metric_hr != 2)  # Curr hr is start of prod
                    and (
                        sum_prev_hr.sim_metric_hr not in [None, 2]
                    )  # 2: Prev hr. is start of production
                    and (sum_curr_hr.sim_metric_hr < THRESHOLD_SIM_METRIC)
                    # and (sum_next_hr.sim_metric_hr >= THRESHOLD_SIM_METRIC) # No Need to check
                    and (
                        sum_prev_hr.num_records
                        >= math.floor(sum_curr_hr.num_records * THRESHOLD_NUM_RECORDS)
                    )  # restriction on the number of records in prev_hr
                    and (
                        sum_curr_hr.num_records
                        >= math.floor(sum_prev_hr.num_records * THRESHOLD_NUM_RECORDS)
                    )  # restriction on the number of records in curr_hr
                ):
                    all_hr_summary[curr_hr].proc_changed = 1
        else:
            all_hr_summary[curr_hr].proc_changed = 0

    return all_hr_summary


def parse_individual_records(
    fetchResult: dict,
    FIRST_SECTION_PROP: float,
    BIN_WIDTH_FIRST_SECTION: float,
    BIN_WIDTH_SECOND_SECTION: float,
) -> list[AccRecord]:
    """Parses each acceleration record received from the MMS server
    Args:
        fetchResult (dict): data received from MMS server
        FIRST_SECTION_PROP (float): proportion of the approved time to devide the data into 2 sections
        BIN_WIDTH_FIRST_SECTION (float): sampling width in the first half of the approved ct
        BIN_WIDTH_SECOND_SECTION (float): sampling widht in the second half of the approved ct
    Returns:
        list[AccRecord]: list of AccRecord objects after processing the fetchResult data
    """
    # Get relevant data
    data: dict = fetchResult["data"]
    counter_id: str = data["counterId"]
    acc_all: list[str] = data["accelerations"]
    approved_ct: int = data["contractedCycleTime"]
    data_acc_id_all: list[int] = data["dataAccId"]
    measurement_time_all: list[str] = data["measurementDate"]
    proc_changed_all: list[Optional[float]] = data["procChanged"]
    sim_metric_all: list[Optional[float]] = data["similarityMetric"]
    sim_metric_existing_hrs: list[Optional[float]] = data["similarityMetricHr"]

    # Create ordered list of each acc record with proper formatting
    records_all = []
    for id, time, acc, sim_metric, sim_metric_hr, proc_changed in zip(
        data_acc_id_all,
        measurement_time_all,
        acc_all,
        sim_metric_all,
        sim_metric_existing_hrs,
        proc_changed_all,
    ):
        try:
            # Initalize time and acceleration object
            time_obj = MeasurementTime(time)
            acc_obj = Acceleration(
                acc_data=acc,
                approved_ct=approved_ct,
                FIRST_SECTION_PROP=FIRST_SECTION_PROP,
                BIN_WIDTH_FIRST_SECTION=BIN_WIDTH_FIRST_SECTION,
                BIN_WIDTH_SECOND_SECTION=BIN_WIDTH_SECOND_SECTION,
            )

            # Intialize object for a single record
            acc_record = AccRecord(
                id=id,
                measurement_time=time_obj.measurement_time,
                measurement_hour=time_obj.upto_hour,
                acc_feature=acc_obj.acc_feature,
                sim_metric=sim_metric,
                sim_metric_hr=sim_metric_hr,
                proc_changed=proc_changed,
            )
            records_all.append(acc_record)

        except Exception as er:
            error_msg = (
                "ERROR in Parsing Acceleration Data"
                f"\nID: {id}"
                f"\nCOUNTER_ID: {counter_id}"
                f"\nMEASUREMENT_DATE:{time}"
                f"\nACCELERATIONS: {acc}"
                "\n---- ---- ---- ---- ---- ---- ----"
            )
            logger.error(
                error_msg,
                exc_info=er,
            )

    # Sort the records chronologically
    if records_all:
        records_all = sorted(
            records_all, key=lambda x: x.measurement_time, reverse=False
        )

    return records_all


def main_proc_change(
    fetchResult: dict,
    THRESHOLD_SIM_METRIC: float,
    N_PREV_RECS: int,
    FIRST_SECTION_PROP: float,
    BIN_WIDTH_FIRST_SECTION: float,
    BIN_WIDTH_SECOND_SECTION: float,
) -> dict:
    """Main function that processes the input data from the MMS server
    Args:
        fetchResult (dict): data received from the MMS server
        THRESHOLD_SIM_METRIC (float): threshold for the similarity metric
        N_PREV_RECS (int): the number of previous records to consider when
                           extracting acc features
        BIN_WIDTH_FIRST_SECTION (float): sampling width in the first section of the process time
        BIN_WIDTH_SECOND_SECTION (float): sampling widht in the second section of the process time
    Returns:
        dict: returns the response of the process change model
    """

    # Get relevant data
    data: dict = fetchResult["data"]
    start_hr: str = data["startHour"]
    measurement_date: str = data["measurementDate"]
    # Create ordered list of each acc record with proper formatting
    records_all = parse_individual_records(
        fetchResult=fetchResult,
        FIRST_SECTION_PROP=FIRST_SECTION_PROP,
        BIN_WIDTH_FIRST_SECTION=BIN_WIDTH_FIRST_SECTION,
        BIN_WIDTH_SECOND_SECTION=BIN_WIDTH_SECOND_SECTION,
    )
    # Get hourly similarity metrics
    all_hr_summary = get_hourly_summary(start_hr=start_hr, records_all=records_all)
    # Identify the existence of process change based on hourly data
    all_hr_summary = identify_process_change(
        all_hr_summary, THRESHOLD_SIM_METRIC=THRESHOLD_SIM_METRIC
    )

    # List to save processed results
    records_processed = []
    # Check if hour summary exists
    for curr_hr, sum_curr_hr in all_hr_summary.items():
        # Get the hourly sim metric and acceleration data
        sim_met_curr_hr = sum_curr_hr.sim_metric_hr
        acc_data_curr_hr = sum_curr_hr.acc_data
        avg_acc_feature_curr = sum_curr_hr.avg_acc_feature

        # Find prev and prev_prev hours
        prev_hr = subtract_hours(
            start_time=curr_hr,
            hrs_to_sub=1,
            input_fmt="%Y%m%d%H",
            output_fmt="%Y%m%d%H",
        )
        prev_prev_hr = subtract_hours(
            start_time=curr_hr,
            hrs_to_sub=2,
            input_fmt="%Y%m%d%H",
            output_fmt="%Y%m%d%H",
        )

        # Find summary of prev and prev_prev hours
        sum_prev_hr = all_hr_summary.get(prev_hr)
        sum_prev_prev_hr = all_hr_summary.get(prev_prev_hr)

        # Get the acc data from prev and prev_prev hours
        acc_data_prev = sum_prev_hr.acc_data if sum_prev_hr else []
        acc_data_prev_prev = sum_prev_prev_hr.acc_data if sum_prev_prev_hr else []
        acc_data_comb = acc_data_prev_prev + acc_data_prev

        if sum_curr_hr.exists and sum_curr_hr.to_return:
            # Process each record in the current hour
            for record in acc_data_curr_hr:
                # Get the previous three records in the previous 2 hours
                last3_acc_records = get_n_neighbor_records(
                    data=acc_data_comb,
                    curr_time=record.measurement_time,
                    max_hr_dist=2,
                    n=N_PREV_RECS,
                    orientation="before",
                )
                # Append the current record to the previous two hours data
                acc_data_comb.append(record)
                # Summary of the acc record of the last 3 records
                sum_last3_records = AccSummary(
                    acc_data=last3_acc_records,
                    avg_method="mean",
                )
                # Get the avg acc feature vector in the last 3 hours
                avg_acc_feature_last3 = sum_last3_records.avg_acc_feature
                acc_feature_record = record.acc_feature

                # Similarity of the 3 previous records with the current one
                if avg_acc_feature_last3 is not None:
                    # No records in the last two hrs
                    sim_met_record = cos_similarity_metric(
                        vector_1=avg_acc_feature_last3, vector_2=acc_feature_record
                    )
                else:
                    sim_met_record = 2

                # Process change detected
                if sum_curr_hr.proc_changed == 1:
                    if sim_met_record < THRESHOLD_SIM_METRIC:
                        proc_change_val = 1
                    else:
                        proc_change_val = 0
                # No process change detected
                else:
                    proc_change_val = 0

                # Result for the record
                result_to_append = {
                    "dataAccId": record.id,
                    "similarityMetric": round(sim_met_record, DECIMAL_PLACE),
                    "similarityMetricHr": round(sim_met_curr_hr, DECIMAL_PLACE),
                    "procChanged": proc_change_val,
                    "avgAccVectorHr": avg_acc_feature_curr,
                }
                records_processed.append(result_to_append)

    # Final model response
    records_processed = sorted(
        records_processed, key=lambda x: x["dataAccId"], reverse=False
    )
    dataAccId = [int(record["dataAccId"]) for record in records_processed]
    similarityMetric = [record["similarityMetric"] for record in records_processed]
    similarityMetricHr = [record["similarityMetricHr"] for record in records_processed]
    procChanged = [record["procChanged"] for record in records_processed]
    avgAccVectorHr = [record["avgAccVectorHr"] for record in records_processed]

    # print(data)
    # print(dataAccId)

    dataAccId_to_measurementDate = dict(zip(data['dataAccId'], measurement_date))
    measurementDate_remaining = [dataAccId_to_measurementDate[data_id] for data_id in dataAccId]

    formatted_dates = []

    for timestamp in measurementDate_remaining:
        dt_obj = datetime.strptime(timestamp, '%Y%m%d%H%M%S')

        formatted_date = dt_obj.strftime("Timestamp('%H:00:')")
    
        formatted_dates.append(formatted_date)


    model_response = {
        "moldId": data["moldId"],
        "counterId": data["counterId"],
        "dataAccId": dataAccId,
        "similarityMetric": similarityMetric,
        "similarityMetricHr": similarityMetricHr,
        "procChanged": procChanged,
        "measurementDate": formatted_dates,
    }

    return model_response
