# -*- coding: utf-8 -*- #
# Author: yinzhangyue
# Created: 2024/6/10
import re
from collections import Counter
from math import isclose


class Metric:
    def __init__(self) -> None:
        pass

    def most_common(self, lst):
        assert lst != [], "The list is empty!"
        new_lst = [i for i in lst if i != ""]
        return Counter(new_lst).most_common(1)[0][0] if new_lst != [] else ""

    def get_consistency(self, response_list: list):
        lst = self.process_pred_list(response_list)
        assert lst != [], "The list is empty!"
        new_lst = [_ for _ in lst if _ != ""]
        return Counter(new_lst).most_common(1)[0][1] if new_lst != [] else 0

    def process_pred(self, response: str) -> str:
        return response

    def process_pred_list(self, response_list: list) -> list:
        pred_list = []
        for response in response_list:
            pred = self.process_pred(response)
            pred_list.append(pred)
        return pred_list

    def cal_acc(self, pred: str, answer: str) -> int:
        return 1 if pred == answer else 0

    def get_acc(self, response_list: list, answer: str):
        pred = self.most_common(self.process_pred_list(response_list))
        return self.cal_acc(pred, answer)


# Math Reasoning
class GSM8K_Metric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def process_pred(self, response: str) -> str:
        # Extract the part after "the answer is"
        pred = response.split("the answer is")[-1]
        # Remove commas and superfluous whitespace
        pred = pred.replace(",", "").strip()
        # Match numeric values (integers and decimals)
        pred_numbers = re.findall(r"-?\d+(?:\.\d+)?", pred)
        # Use the last matched number if present
        pred = pred_numbers[-1] if pred_numbers else ""
        return pred

    def cal_acc(self, pred: str, answer: str) -> int:
        if pred == "":
            return 0
        try:
            pred_value = float(pred)
            # Extract final numeric answer from ground truth (after '####')
            answer_match = re.search(r'#### (\d+)', answer)
            true_answer = answer_match.group(1) if answer_match else answer
            answer_value = float(true_answer.replace(",", ""))
            # Compare floats using isclose
            return 1 if isclose(pred_value, answer_value, rel_tol=1e-5) else 0
        except ValueError:
            return 0


class MultiArith_Metric(GSM8K_Metric):
    def __init__(self) -> None:
        super().__init__()


class SingleEq_Metric(GSM8K_Metric):
    def __init__(self) -> None:
        super().__init__()


class AddSub_Metric(GSM8K_Metric):
    def __init__(self) -> None:
        super().__init__()


class AQuA_Metric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def process_pred(self, response: str) -> str:
        # Extract the part after "the answer is" and convert to upper-case
        pred = response.split("the answer is")[-1].strip().upper()
        # Match options A-E
        pred_options = re.findall(r"[A-E]", pred)
        # Use the last matched option if present
        pred = pred_options[-1] if pred_options else ""
        return pred

    def cal_acc(self, pred: str, answer: str) -> int:
        if pred == "":
            return 0
        # Case-insensitive comparison ignoring superfluous whitespace
        return 1 if pred == answer.strip().upper() else 0


class SVAMP_Metric(GSM8K_Metric):
    def __init__(self) -> None:
        super().__init__()


# Commonsense Reasoning
class CSQA_Metric(AQuA_Metric):
    def __init__(self) -> None:
        super().__init__()


class StrategyQA_Metric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def process_pred(self, response: str) -> str:
        # Extract the part after "the answer is" and convert to lower-case
        pred = response.split("the answer is")[-1].lower()
        # Match "yes" or "no"
        pred_options = re.findall(r"\b(yes|no)\b", pred)
        # Use the last matched token if present
        pred = pred_options[-1] if pred_options else ""
        return pred

    def cal_acc(self, pred: str, answer) -> int:
        if pred == "":
            return 0
        # Handle boolean ground-truth
        if isinstance(answer, bool):
            answer_str = "yes" if answer else "no"
        else:
            answer_str = str(answer).strip().lower()
        # Case-insensitive comparison ignoring superfluous whitespace
        return 1 if pred == answer_str else 0
