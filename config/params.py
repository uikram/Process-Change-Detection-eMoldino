class ProcChangeConfig:
    """configurations for the proc change algorithm"""

    THRESHOLD_SIM_METRIC: float = 0.7  # Cosine similarity threshold
    N_PREV_RECS: int = 3  # Number of prev acc measurements used for comparison
    FIRST_SECTION_PROP: float = 0.5  # value is between [0, 1]
    BIN_WIDTH_FIRST_SECTION: float = 0.5  # Sampling width for first section in secs
    BIN_WIDTH_SECOND_SECTION: float = 1.0  # Sampling width for the second section
