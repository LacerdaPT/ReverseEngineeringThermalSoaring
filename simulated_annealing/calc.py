import logging
import time
from copy import deepcopy
from numbers import Number
from types import LambdaType, FunctionType
from typing import Iterable

import numpy as np
from scipy import stats as st
from scipy.optimize import OptimizeResult
from scipy.optimize._dual_annealing import ObjectiveFunWrapper, LocalSearchWrapper


logger = logging.getLogger(__name__)


class GeneralizedObjectiveFunWrapper(ObjectiveFunWrapper):
    def __init__(self,  func, maxfun=1e7, *args, **kwargs):
        super().__init__(func, maxfun=maxfun, *args)

        self.kwargs = kwargs

    def fun(self, x):
        self.nfev += 1
        try:
            function_result = self.func(x, *self.args, **self.kwargs)
        except Exception as e:
            raise e
        return function_result


def my_annealing(func, bounds, kwargs=None, max_iter=1000, annealing_type='cauchy',
                 minimizer_kwargs=None, temperature=1, rescalings=None,
                 max_fiter=1e7, no_local_search=False, n_stuck=1e4, n_consecutive_exception_max=None,
                 start_from=0, new_best_callback=None, new_accepted_callback=None, x0=None, x_best=None, f_best=None,
                 i_best=None):

    if kwargs is None:
        kwargs = {}
    if minimizer_kwargs is None:
        minimizer_kwargs = {}
    if n_consecutive_exception_max is None:
        n_consecutive_exception_max = np.inf

    lu = list(zip(*bounds))
    lower = np.array(lu[0])
    upper = np.array(lu[1])
    n_dim = len(bounds)
    bound_range = upper - lower
    # Checking bounds are valid
    if (np.any(np.isinf(lower)) or np.any(np.isinf(upper)) or np.any(
            np.isnan(lower)) or np.any(np.isnan(upper))):
        raise ValueError('Some bounds values are inf values or nan values')
    # Checking that bounds are consistent
    if not np.all(lower < upper):
        raise ValueError('Bounds are not consistent min < max')
    # Checking that bounds are the same length
    if not len(lower) == len(upper):
        raise ValueError('Bounds do not have the same dimensions')

    # ================================================================================================================ #
    # =====================================         SET DISTRIBUTION           ======================================= #
    # ================================================================================================================ #
    if isinstance(annealing_type, str):
        if annealing_type == 'cauchy':
            step_dist = st.cauchy
            step_dist_kwargs = {'loc': 0,
                                'scale': lambda i, t: t}
        elif annealing_type == 'gaussian':
            step_dist = st.norm
            step_dist_kwargs = {'loc': 0,
                                'scale': lambda i, t: np.sqrt(2 * t)}
        else:
            step_dist = getattr(st, annealing_type)
            step_dist_kwargs = {'loc': 0,
                                'scale': lambda i, t: t}
    else:
        raise ValueError('annealing_type must be a str')

    # ================================================================================================================ #
    # =====================================         SET TEMPERATURE           ======================================== #
    # ================================================================================================================ #
    if isinstance(temperature, Number):

        if annealing_type == 'cauchy':
            temperature_array = temperature / np.arange(1, start_from + max_iter + 1) ** (1 / n_dim)
        elif annealing_type == 'gaussian':
            temperature_array = [np.log(2) * temperature / np.log(1 + i) for i in np.arange(1, start_from + max_iter + 1)]
        else:
            temperature_array = [temperature / (1 + i) for i in np.arange(0, start_from + max_iter)]

    elif isinstance(temperature, Iterable):
        temperature_array = deepcopy(temperature)
    else:
        raise ValueError('Temperature must be either a number or a array of size max_iter')

    assert len(temperature_array) == start_from + max_iter, 'The length of temperature must be the same as max_iter'

    # ================================================================================================================ #
    # ================================================================================================================ #

    if rescalings is None:
        rescalings = bound_range / temperature
    elif isinstance(rescalings, (int, float)):
        rescalings = np.full(shape=n_dim, fill_value=rescalings)
    else:
        rescalings = np.array(rescalings)

    # list to hold history of accepted values of x and f
    x_previous_array = []
    f_previous_array = []
    x_candidate_array = []
    f_candidate_array = []
    acceptance_prob_array = []
    accept_array = []
    new_global_minimum_array = []
    message = []
    # OptimizeResult object to be returned
    optimize_res = OptimizeResult()
    optimize_res.success = True
    optimize_res.status = 0

    # Wrapper for the objective function
    func_wrapper = GeneralizedObjectiveFunWrapper(func, max_fiter, **kwargs)
    # Wrapper fot the minimizer
    minimizer_wrapper = LocalSearchWrapper(
        bounds, func_wrapper, **minimizer_kwargs)

    # Set Initial Conditions
    if x0 is None:
        x_previous = np.array([np.random.uniform(*b) for b in bounds])
    else:
        x_previous = x0
    f_previous = func_wrapper.fun(x_previous)
    x_best = x_best if x_best is not None else x_previous
    f_best = f_best if f_best is not None else f_previous
    i_best = i_best if i_best is not None else -1

    # Flow control Variables
    total_running_time = 0
    n_consecutive_exception = 0
    iteration = 0
    i_stuck = 0
    start_time = time.time()

    for i, current_iteration in enumerate(range(start_from, max_iter + start_from)):
        logger.info(f'iteration {i} / {max_iter} - {100 * i / max_iter:.2g} % complete '
                    f'( {current_iteration} / {start_from + max_iter} - {100 * current_iteration / (start_from + max_iter):.2g} % )')
        logger.debug(f'{total_running_time / (i + 1) * ( max_iter - i) / 60.:.2g} minutes to go')

        # Compute temperature for this step
        current_temperature = temperature_array[current_iteration]

        if iteration >= max_fiter:
            message.append("Maximum number of iteration reached")
            need_to_stop = True
            break

        # get new candidate
        current_dist_kwargs = {}
        for arg, value in step_dist_kwargs.items():
            if isinstance(value, (FunctionType, LambdaType)):
                current_dist_kwargs[arg] = value(current_iteration, current_temperature)
            else:
                current_dist_kwargs[arg] = value
        current_step = step_dist(**current_dist_kwargs).rvs(size=n_dim)
        current_step = current_step * rescalings

        x_candidate = np.mod(x_previous + current_step - lower,
                             bound_range) + lower
        logger.debug(f'{x_previous=}')
        logger.debug(f'{current_step=}')
        logger.debug(f'{x_candidate=}')
        logger.debug(f'{f_previous=:.5g}')
        # Calculate new f
        try:
            f_candidate = func_wrapper.fun(x_candidate)
            n_consecutive_exception = 0
        except Exception as e:
            logger.error('function call raised the following exception')
            logger.exception(e)
            f_candidate = np.inf
            n_consecutive_exception += 1
            if n_consecutive_exception < n_consecutive_exception_max:
                logger.warning('continuing..')
            else:
                logger.warning('exiting..')
                break
        finally:
            logger.debug(f'{f_candidate=:.5g}')
            logger.debug(f'{n_consecutive_exception=}')

        logger.debug(f'Delta = {f_candidate - f_previous:.5g}')
        logger.debug(f'{current_temperature=:.5g}')
        random_var = np.random.uniform()
        accept_probability = np.exp(-(f_candidate - f_previous) / current_temperature)
        if f_candidate < f_previous:
            x_previous = x_candidate
            f_previous = f_candidate
            accepted = True
            i_stuck = 0

        elif random_var < accept_probability:
            x_previous = x_candidate
            f_previous = f_candidate
            accepted = True
        else:
            i_stuck += 1
            accepted = False

        logger.debug(f'{accept_probability=:.5g}')
        logger.debug(f'{accepted=:.5g}')
        
        if f_previous < f_best:
            if not no_local_search:
                print(f'starting local search around {x_previous=}, {f_best}')
                e_local, x_local = minimizer_wrapper.local_search(x_previous,
                                                                  f_previous)
                if e_local < f_previous:
                    x_previous = x_local
                    f_previous = e_local

            x_best = x_previous
            f_best = f_previous
            i_best = current_iteration
            new_global_minimum = True
        else:
            new_global_minimum = False

        if new_global_minimum and (new_best_callback is not None):
            new_best_callback(x_previous, f_previous, x_candidate, f_candidate, accept_probability, accepted, new_global_minimum)

        if accepted and (new_accepted_callback is not None):
            new_accepted_callback(x_previous, f_previous, x_candidate, f_candidate, accept_probability, accepted,
                                  new_global_minimum)

        logger.debug(f'{new_global_minimum=:.5g}')

        x_previous_array.append(x_previous)
        f_previous_array.append(f_previous)
        x_candidate_array.append(x_candidate)
        f_candidate_array.append(f_candidate)
        acceptance_prob_array.append(accept_probability)
        accept_array.append(accepted)
        new_global_minimum_array.append(new_global_minimum)

        if i_stuck > n_stuck:
            break
        iteration += 1

        total_running_time = time.time() - start_time

    temperature_array = temperature_array[start_from: start_from + iteration]
    x_previous_array = np.array(x_previous_array)
    x_candidate_array = np.array(x_candidate_array)
    f_previous_array = np.array(f_previous_array)[:, np.newaxis]
    f_candidate_array = np.array(f_candidate_array)[:, np.newaxis]
    acceptance_prob_array = np.array(acceptance_prob_array)[:, np.newaxis]
    accept_array = np.array(accept_array)[:, np.newaxis]
    new_global_minimum_array = np.array(new_global_minimum_array)[:, np.newaxis]
    temperature_array = np.array(temperature_array)[:, np.newaxis]

    history = np.hstack([x_previous_array,
                         f_previous_array,
                         x_candidate_array,
                         f_candidate_array,
                         acceptance_prob_array,
                         accept_array,
                         new_global_minimum_array,
                         temperature_array,
                         ])

    # Setting the OptimizeResult values
    optimize_res.x = np.array(x_best)
    optimize_res.fun = f_best
    optimize_res.i = i_best
    optimize_res.nit = iteration
    optimize_res.nfev = func_wrapper.nfev
    optimize_res.njev = func_wrapper.ngev
    optimize_res.nhev = func_wrapper.nhev
    optimize_res.message = message
    return optimize_res, history
