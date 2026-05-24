import os

import pandas as pd

from vulscan.utils.project_info import PROJECT_PATH

cwe_descriptions = None


def _init_cwe_descriptions():
    global cwe_descriptions
    cwe_df = pd.read_csv(os.path.join(PROJECT_PATH, "datasets/cwe-details.csv"))
    cwe_descriptions = {}
    for _, row in cwe_df.iterrows():
        name = str(row["Name"]) if not pd.isna(row["Name"]) else ""
        if pd.isna(row["Extended Description"]):
            desc = str(row["Description"]) if not pd.isna(row["Description"]) else ""
            cwe_descriptions[row["CWE-ID"]] = name + ". " + desc
        else:
            cwe_descriptions[row["CWE-ID"]] = name + ". " + str(row["Extended Description"])


def get_cwe_info(cwe_id: int) -> str:
    """
    Describe the CWE with the given ID.

    Args:
        cwe_id: The ID of the CWE to describe.

    Returns:
        A string containing the description of the CWE.

    Example:
    input: 120
    output: 'Buffer Copy without Checking Size of Input (Classic Buffer Overflow)'
    """
    if cwe_descriptions is None:
        _init_cwe_descriptions()

    return cwe_descriptions.get(cwe_id, "Unknown CWE")
