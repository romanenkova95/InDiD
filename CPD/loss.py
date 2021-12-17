"""Methods for calculating principle loss function for Change Point Detection."""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class CPDLoss(nn.Module):
    """Class implementing loss function for Change Point Detection."""

    def __init__(self, len_segment: int, alpha: float) -> None:
        """Initialize parameters for CPDLoss.

        :param len_segment: parameter restricted the size of a considered segment in delay loss (T in the paper)
        """
        super().__init__()

        self.len_segment = len_segment
        self.alpha = alpha

    @staticmethod
    def calculate_delays_(prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate delays (detection or false alarms) for probabilities.

        :param prob: probabilities of changing in time moment t
        :return: tuple of
            - delays
            - products of ones minus probabilities
        """
        device = prob.device
        len_prob = prob.size(0)

        prob_no_change = torch.ones(len_prob).to(device)

        # 1, 1 - p_0, 1 - p_1, ..., 1 - p_{n-1}
        prob_no_change[1:] -= prob[:-1]

        # 1, 1 - p_0, (1 - p_0) * (1 - p_1), ..., prod_1^{N-1}(1 - p_i)
        prob_no_change_before = torch.cumprod(prob_no_change, dim=0).to(device)

        # 1*p_0, 2 * p_1 * (1 - p_0), 3 * p_2 * (1 - p_1) * (1 - p_0), ....
        delays = torch.arange(1, len_prob + 1).to(device) * prob * prob_no_change_before

        # (1 - p_0) * (1 - p_1) * ... * (1 - p_N)
        prod_prob_no_change = torch.prod(prob_no_change) * (
            torch.ones(1).to(device) - prob[-1]
        )

        return delays, prod_prob_no_change

    @staticmethod
    def delay_detection_loss_(prob: torch.Tensor) -> torch.Tensor:
        """Calculate detection delay loss.

        :param prob: probabilities of changing in time moment t
        :return: loss expressing delay detection
        """
        device = prob.device
        delays, prod_prob_no_change = CPDLoss.calculate_delays_(prob)

        delays = delays.to(device)

        # calculate delay loss
        delay_loss = torch.sum(delays) + (len(prob) + 1) * torch.prod(
            prod_prob_no_change
        )
        return delay_loss

    @staticmethod
    def false_alarms_loss_(
        prob: torch.Tensor, on_intervals: bool = False
    ) -> torch.Tensor:
        """Calculate time to false alarms loss.

        :param prob: probabilities of changing in time moment t
        :param on_intervals: if True separate intervals on sub-slices and calculate loss on them
        :return: loss expressing time to false alarms
        """
        len_prob = len(prob)
        device = prob.device
        fp_loss = torch.zeros(1).to(device)

        # calculate FA loss in random sub-intervals
        if on_intervals:
            start_ind = 0
            end_ind = 0

            while end_ind < len_prob:

                # we want non-overlapping random interval
                start_ind = max(0, end_ind)
                len_interval = np.random.randint(max(len_prob // 16, 1), len_prob)
                end_ind = min(len_prob, start_ind + len_interval)

                delays, prod_prob_no_change = CPDLoss.calculate_delays_(
                    prob[start_ind:end_ind]
                )
                fp_loss += -torch.sum(delays) - (len_prob + 1) * torch.prod(
                    prod_prob_no_change
                )

        else:
            # calculate FA loss
            delays, prod_prob_no_change = CPDLoss.calculate_delays_(prob)
            fp_loss = -torch.sum(delays) - (len_prob + 1) * torch.prod(
                prod_prob_no_change
            )
        return fp_loss

    def forward(self, prob: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        """Calculate CPD loss.

        :param prob: probabilities of changing in time moment t
        :param true_labels: true labels is there is change in moment t
        :return: CPD loss
        """
        device = prob.device
        loss = torch.zeros(1).to(device)
        label: torch.Tensor
        loss_curr = 0

        for i, label in enumerate(true_labels):
            change_ind = torch.nonzero(label != label[0])

            # calculate false alarms part before change moment
            # calculate delay detection part after change moment (if there is change)
            if change_ind.size()[0] == 0:
                # if normal data without change
                fp_loss = CPDLoss.false_alarms_loss_(prob[i, :])
                loss_curr = fp_loss

            else:
                change_ind = change_ind[0]
                delay_loss = CPDLoss.delay_detection_loss_(
                    prob[i, change_ind : (change_ind + self.len_segment)]
                )
                fp_loss = CPDLoss.false_alarms_loss_(prob[i, :change_ind])
                loss_curr = self.alpha * len(prob) / self.len_segment * delay_loss + fp_loss

            loss += loss_curr

        loss = loss / true_labels.size(0)
        return loss
