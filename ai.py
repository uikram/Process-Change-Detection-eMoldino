"""
This module contains a class to run the model for process change
"""

from ai.model.EM_AI_PC.artifacts.ai_pc import main_proc_change
from ai.model.EM_AI_PC.config.params import ProcChangeConfig
from ai.model.meta.model import BaseModel


class AI_PC(BaseModel):
    def __init__(self, aiType):
        self.aiType = aiType

    def run(self, fetchResult):
        proc_change_result = main_proc_change(
            fetchResult=fetchResult,
            THRESHOLD_SIM_METRIC=ProcChangeConfig.THRESHOLD_SIM_METRIC,
            N_PREV_RECS=ProcChangeConfig.N_PREV_RECS,
            FIRST_SECTION_PROP=ProcChangeConfig.FIRST_SECTION_PROP,
            BIN_WIDTH_FIRST_SECTION=ProcChangeConfig.BIN_WIDTH_FIRST_SECTION,
            BIN_WIDTH_SECOND_SECTION=ProcChangeConfig.BIN_WIDTH_SECOND_SECTION,
        )
        model_response = {"aiType": self.aiType, "data": proc_change_result}
        return model_response
