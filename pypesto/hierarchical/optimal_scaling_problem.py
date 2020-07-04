import numpy as np
from typing import List
import petab
import amici

from .problem import InnerProblem
from .parameter import InnerParameter
from.problem import (inner_parameters_from_parameter_df,
                     ixs_for_measurement_specific_parameters,
                     ix_matrices_from_arrays)


class OptimalScalingProblem(InnerProblem):
    def __init__(self,
                 xs: List[InnerParameter],
                 data: List[np.ndarray]):
        super().__init__(xs, data)

        self.groups = {}

        for idx, gr in enumerate(self.get_groups_for_xs(InnerParameter.OPTIMALSCALING)):
            self.groups[gr] = {}
            xs = self.get_xs_for_group(gr)
            self.num_categories = len(xs)
            self.num_datapoints = \
                np.sum([np.size(data[idx]) for idx in range(len(data))])

            self.num_inner_params = self.num_datapoints + 2*self.num_categories

            self.num_constr_full = 2*self.num_datapoints + self.num_categories - 1

            self.lb_indices = \
                list(range(self.num_datapoints,
                           self.num_datapoints + self.num_categories))

            self.ub_indices = \
                list(range(self.num_datapoints + self.num_categories,
                           self.num_inner_params))

            self.cat_ixs = {}
            self.get_cat_indices()

            self.C = self.initialize_c(xs)

            self.W = self.initialize_w()

    @staticmethod
    def from_petab_amici(
            petab_problem: petab.Problem,
            amici_model: 'amici.Model',
            edatas: List['amici.ExpData']):
        return qualitative_inner_problem_from_petab_problem(
            petab_problem, amici_model, edatas)

    def initialize_c(self, xs):
        constr = np.zeros([self.num_constr_full, self.num_inner_params])
        data_idx = 0
        for cat_idx, x in enumerate(xs):
            num_data_in_cat = int(np.sum(
                [np.sum(x.ixs[idx]) for idx in range(len(x.ixs))]
            ))
            for data_in_cat_idx in range(num_data_in_cat):
                # x_lower - y_surr <= 0
                constr[data_idx, data_idx] = -1
                constr[data_idx, cat_idx + self.num_datapoints] = 1

                # y_surr - x_upper <= 0
                constr[data_idx + self.num_datapoints, data_idx] = 1
                constr[data_idx + self.num_datapoints,
                       cat_idx + self.num_datapoints + self.num_categories] = -1

                # x_upper_i - x_lower_{i+1} <= 0
                if cat_idx < self.num_categories - 1:
                    constr[2*self.num_datapoints + cat_idx,
                           self.num_datapoints + cat_idx + 1] = -1
                    constr[2*self.num_datapoints + cat_idx,
                           self.num_datapoints + self.num_categories + cat_idx] = 1
                data_idx += 1

        return constr

    def initialize_w(self):
        weights = np.diag(np.block(
                [np.ones(self.num_datapoints),
                 np.zeros(2*self.num_categories)])
        )
        return weights

    def get_cat_for_xi_idx(self, data_idx):
        for cat_idx, (_, indices) in enumerate(self.cat_ixs.items()):
            if data_idx in indices:
                return cat_idx

    def get_cat_indices(self):
        idx_tot = 0
        for x in self.xs:
            num_points = \
                np.sum(
                    [np.sum(self.xs[x].ixs[idx]) for idx in range(len(self.xs[x].ixs))]
                )
            self.cat_ixs[x] = list(range(idx_tot, idx_tot + num_points))
            idx_tot +=num_points


def qualitative_inner_problem_from_petab_problem(
        petab_problem: petab.Problem,
        amici_model: 'amici.Model',
        edatas: List['amici.ExpData']):
    # inner parameters
    inner_parameters = inner_parameters_from_parameter_df(
        petab_problem.parameter_df)

    x_ids = [x.id for x in inner_parameters]

    # used indices for all measurement specific parameters
    ixs = ixs_for_measurement_specific_parameters(
        petab_problem, amici_model, x_ids)
    # print(ixs)
    # transform experimental data
    edatas = [amici.numpy.ExpDataView(edata)['observedData']
              for edata in edatas]
    # print(edatas)
    # matrixify
    ix_matrices = ix_matrices_from_arrays(ixs, edatas)
    # print(ix_matrices)
    # assign matrices
    for par in inner_parameters:
        par.ixs = ix_matrices[par.id]

    return OptimalScalingProblem(inner_parameters, edatas)
