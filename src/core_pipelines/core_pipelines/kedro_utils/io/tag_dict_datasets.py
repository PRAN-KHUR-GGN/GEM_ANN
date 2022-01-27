# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.
"""
io classes for TagDict
"""

from kedro.extras.datasets.pandas import ExcelDataSet, CSVDataSet

from optimus_core.core.tag_management.tag_dict import TagDict


class TagDictCSVLocalDataSet(CSVDataSet):
    """ Loads and saves a TagDict object from/to csv
        This is an extension of the Kedro Pandas CSV Dataset
    """

    def _load(self) -> TagDict:
        df = super()._load()
        return TagDict(df)

    def _save(self, data: TagDict) -> None:
        df = data.to_frame()
        super()._save(df)


class TagDictExcelLocalDataSet(ExcelDataSet):
    """
    Loads and saves a TagDict object from/to excel

    This is an extension of the Kedro Text Dataset

    To load from a specific sheet, add "sheet_name" to the
    "load_args" in your catalog entry. To save to a specific
    sheet, add "sheet_name" to the "save_args" in your catalog entry.

    """

    def _load(self) -> TagDict:
        df = super()._load()
        return TagDict(df)

    def _save(self, data: TagDict) -> None:
        df = data.to_frame()
        super()._save(df)
