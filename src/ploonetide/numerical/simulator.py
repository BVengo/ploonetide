import numpy as np
import warnings

from scipy.integrate._ivp.base import OdeSolver
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

__all__ = ['Variable', 'Simulation']


# monkey patching the ode solvers with a progress bar

# save the old methods - we still need them
old_init = OdeSolver.__init__
old_step = OdeSolver.step

# define our own methods
def new_init(self, fun, t0, y0, t_bound, vectorized=True, support_complex=False):

    # define the progress bar
    bar_format = '{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} steps | {elapsed}<{remaining}'
    self.pbar = tqdm(desc='Progress: ', bar_format=bar_format, total=t_bound - t0, initial=t0)
    self.last_t = t0

    # call the old method - we still want to do the old things too!
    old_init(self, fun, t0, y0, t_bound, vectorized, support_complex)


def new_step(self):
    # call the old method
    old_step(self)

    # update the bar
    tst = self.t - self.last_t
    self.pbar.update(tst)
    self.last_t = self.t

    # close the bar if the end is reached
    if self.t >= self.t_bound:
        self.pbar.close()


# overwrite the old methods with our customized ones
OdeSolver.__init__ = new_init
OdeSolver.step = new_step


class Variable:
    """Define a new variable for integration

    Args:
        name (str): Name of a variable for integrating
        v_ini (float): Value of a variable (or initial condition)
    """

    def __init__(self, name, v_ini):

        self.name = name
        self.v_ini = v_ini

        pass

    def return_vec(self) -> np.array:

        return np.array([self.v_ini])


class Simulation:
    """Build a simulation.

    Args:
        variables (list): List of variables (or initial conditions)
    """

    def __init__(self, variables):
        self.variables = variables
        self.N_variables = len(self.variables)
        self.Ndim = len(self.variables)
        self.quant_vec = np.concatenate(np.array([var.return_vec()
                                                  for var in self.variables]))

    def set_diff_eq(self, calc_diff_eqs, **kwargs):
        """
        Method which assigns an external solver function as the diff-eq solver
        for the integrator. For N-body or gravitational setups, this is the
        function which calculates accelerations.

        Args:
            calc_diff_eqs: A function which returns a [y] vector for RK4
            **kwargs: Any additional inputs/hyperparameters the external function requires
        """
        self.diff_eq_kwargs = kwargs
        self.calc_diff_eqs = calc_diff_eqs

    def set_integration_method(self, method='RK45'):
        """Define integration method for the simulation.

        Args:
            method (str, optional): method to use ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
        """
        self.integration_method = method

    def run(self, t, dt, t0=0.0):
        """Run simulation for the given variables.

        Params:
            t (float): total time (in simulation units) to run the simulation. Can have units or not, just set has_units appropriately.
            dt (float): timestep (in simulation units) to advance the simulation. Same as above
            t0 (float, optional): set a non-zero start time to the simulation.
        """

        t_span = np.array([0.000001, t])
        tint = np.arange(t_span[0], t_span[1], dt)  # Vector for time

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # sols = odeint(self.calc_diff_eqs, self.quant_vec, tint,
            #               args=(self.diff_eq_kwargs,), mxstep=1000, mxordn=20)

            # nsteps = int((t - t0) / dt)
            # for i in tqdm(range(nsteps), desc='Progress: ', bar_format=fmt):
            self.bar_format = '{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} steps | {elapsed}<{remaining}'

            sols = solve_ivp(self.calc_diff_eqs, t_span, self.quant_vec, vectorized=True, rtol=1E-20, min_step=1e-6,
                             method=self.integration_method, args=(self.diff_eq_kwargs,), t_eval=tint)

            self.history = sols.t, sols.y
