

import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm

__all__ = ['Variable', 'Simulation']


class Variable:
    def __init__(self, name, v_ini):
        self.name = name
        self.v_ini = v_ini

        pass

    def return_vec(self):

        return np.array([self.v_ini])


class Simulation:
    def __init__(self, variables):
        self.variables = variables
        self.N_variables = len(self.variables)
        self.Ndim = len(self.variables)
        self.quant_vec = np.concatenate(np.array([var.return_vec()
                                                  for var in self.variables]))

    def set_diff_eq(self, calc_diff_eqs, **kwargs):
        '''
        Method which assigns an external solver function as the diff-eq solver
        for the integrator. For N-body or gravitational setups, this is the
        function which calculates accelerations.
        ---------------------------------
        Params:
            calc_diff_eqs: A function which returns a [y] vector for RK4
            **kwargs: Any additional inputs/hyperparameters the external function requires
        '''
        self.diff_eq_kwargs = kwargs
        self.calc_diff_eqs = calc_diff_eqs

    def set_integration_method(self, method="rk4"):
        '''
        Class method to set the integration method for the simulation
        '''

        self.integration_method = method

    def integrator(self, t, dt):
        '''
        Integrator. Calculates a new y vector
            --------------------------------
        Params:
            t: time. Only used if the DO depends on time (gravity doesn't).
            dt: timestep. Non adaptive in this case.
        '''

        tint = np.arange(0.000001, t, dt)  # Vector for time

        if self.integration_method == "lsoda":
            sols = odeint(self.calc_diff_eqs, self.quant_vec, tint,
                          args=(self.diff_eq_kwargs,),
                          mxords=20, mxordn=20, mxstep=5000)

            return tint, sols

        if self.integration_method == "rk4":
            k1 = dt * np.array(self.calc_diff_eqs(self.quant_vec, t,
                                                  self.diff_eq_kwargs))
            k2 = dt * np.array(self.calc_diff_eqs(self.quant_vec + 0.5 * k1,
                                                  t + 0.5 * dt,
                                                  self.diff_eq_kwargs))
            k3 = dt * np.array(self.calc_diff_eqs(self.quant_vec + 0.5 * k2,
                                                  t + 0.5 * dt,
                                                  self.diff_eq_kwargs))
            k4 = dt * np.array(self.calc_diff_eqs(self.quant_vec + k3,
                                                  t + dt,
                                                  self.diff_eq_kwargs))

            sols = self.quant_vec + (k1 + 2 * k2 + 2 * k3 + k4) / 6.

            return sols

    def run(self, t, dt, t0=0):
        '''
        Method which runs the simulation on a given set of bodies.
        ---------------------
        Params:
            t: total time (in simulation units) to run the simulation. Can have units or not, just set has_units appropriately.
            dt: timestep (in simulation units) to advance the simulation. Same as above
            t0 (optional): set a non-zero start time to the simulation.

        Returns:
            None, but leaves an attribute history accessed via
            'simulation.history' which contains all y vectors for the simulation.
            These are of shape (Nstep,Nbodies * 6), so the x and y positions of particle 1 are
            simulation.history[:,0], simulation.history[:,1], while the same for particle 2 are
            simulation.history[:,6], simulation.history[:,7]. Velocities are also extractable.
        '''

        if self.integration_method == "lsoda":
            self.history = self.integrator(t, dt)
            pass

        if self.integration_method == "rk4":
            history = [self.quant_vec]
            ts = [0.000001]
            nsteps = int((t - t0) / dt)
            fmt = "{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} steps | {elapsed}<{remaining}"
            for i in tqdm(range(nsteps), desc="Progress: ", bar_format=fmt):
                y_new = self.integrator(0, dt)
                history.append(y_new)
                self.quant_vec = y_new
                t = ts[-1] + dt
                ts.append(t)
            self.history = (np.array(ts), np.array(history))
            pass
