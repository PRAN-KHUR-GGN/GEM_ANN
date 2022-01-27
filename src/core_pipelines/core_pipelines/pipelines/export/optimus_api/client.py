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
import json
import logging
import requests


logger = logging.getLogger(__name__)


class Client:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def create_runs(self, runs):
        logging.info("creating %i runs", len(runs))
        for run in runs:
            self.create_run(run)

    def create_run(self, run) -> dict:
        requests.post(
            f"{self.endpoint}/api/v1/runs", data=json.dumps(run), headers=self.headers()
        )

    def create_tags(self, tags):
        logging.info("creating %i tags", len(tags))
        for tag in tags:
            self.create_tag(tag)

    def create_tag(self, tag):
        requests.post(
            f"{self.endpoint}/api/v1/tags", data=json.dumps(tag), headers=self.headers()
        )

    def create_states(self, states):
        logging.info("creating %i states", len(states))
        for state in states:
            self.create_state(state)

    def create_state(self, state):
        requests.post(
            f"{self.endpoint}/api/v1/states",
            data=json.dumps(state),
            headers=self.headers(),
        )

    def create_recommendations(self, recommendations):
        logging.info("creating %i recommendations", len(recommendations))
        for recommendation in recommendations:
            self.create_recommendation(recommendation)

    def create_recommendation(self, recommendation):
        requests.post(
            f"{self.endpoint}/api/v1/recommendations",
            data=json.dumps(recommendation),
            headers=self.headers(),
        )

    def create_predictions(self, predictions):
        logging.info("creating %i predictions", len(predictions))
        for prediction in predictions:
            self.create_prediction(prediction)

    def create_prediction(self, prediction):
        requests.post(
            f"{self.endpoint}/api/v1/predictions",
            data=json.dumps(prediction),
            headers=self.headers(),
        )

    def create_control_sensitivities(self, control_sensitivities):
        logging.info("creating %i control_sensitivities", len(control_sensitivities))
        requests.post(
            f"{self.endpoint}/api/v1/control_sensitivities",
            data=json.dumps(control_sensitivities),
            headers=self.headers(),
        )

    def headers(self):
        return {"Content-type": "application/json", "Accept": "text/plain"}
