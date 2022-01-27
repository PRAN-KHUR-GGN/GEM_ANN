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
"""Functions to generate an html report."""

import markdown
from pathlib import Path

import logging
import os
from jinja2 import Environment, FileSystemLoader
from typing import List, Dict
from markdown.extensions.toc import TocExtension


def generate_html_report(
    figures: List[Dict], params: Dict,
):
    """Generates an html report for figures specified in `figures` at the location in
    `params`. Check README file for more details.
    """
    report_meta_data = params.get("meta_data", "")
    _report_template_path = os.path.abspath(
        os.path.join(params.get("report_template_path", ""))
    )
    file_dir = _report_template_path or Path(__file__).resolve().parent
    html_file = os.path.abspath(
        os.path.join(params.get("report_path", "report_output.html"))
    )

    file_loader = FileSystemLoader(file_dir)
    env = Environment(loader=file_loader)

    template = env.get_template(params.get("template_markdown", "template.md"))
    md_output = template.render(figures=figures, report=report_meta_data)

    html_output = markdown.markdown(
        md_output, extensions=[TocExtension(toc_depth="2-6"), "md_in_html"]
    )
    logging.info("Writing to {}".format(str(html_file)))
    with open(html_file, "w") as file:
        file.write(html_output)


def generate_report(params: Dict, figures_dict: Dict):
    """
    Write a html report to disk.

    Path to saved file taken from params. Path specified in params at
    params['train_model']['report']['output_path'].
    Args:
        params: report generation params
        figures_dict: Dict of figures, keyed by values
    Returns: None
    """
    temp_dir = params.get("temp_out_dir", "data/08_reporting/")
    os.makedirs(temp_dir, exist_ok=True)
    fig_description = params.get("fig_desc", {})

    figures_list = []
    for name, fig in figures_dict.items():
        save_path = os.path.join(temp_dir, name + ".html")
        fig.write_html(save_path)
        figures_list.append(
            {
                "name": name,
                "description": fig_description.get(
                    name, f"This is a default text for plot {name}"
                ),
                "path": save_path,
            }
        )
    generate_html_report(figures_list, params)
