import tbilby
import bilby
import numpy as np
from tbilby.core.prior.HG import MarginalizedTruncatedHollowedGaussian, ConditionalTruncatedHollowedGaussian
from tbilby.core.prior.HG import condition_func_t1, condition_func_t2, condition_func_t3, condition_func_t4
from tbilby.core.prior.HG import condition_func_f1, condition_func_f2, condition_func_f3, condition_func_f4
from tbilby.core.prior.HG import ConditionalPriorDict, get_A1, get_A2, get_A3
from bilby.core.prior import ConditionalLogUniform, LogUniform
from gwpy.timeseries import TimeSeries

# First set up logging and some output directories and labels
logger = bilby.core.utils.logger
outdir = "GW150914_wavelet"
label = "GW150914"
sampling_frequency = 2048.0
trigger_time = 1126259462.391
detectors = ["H1", "L1"]
maximum_frequency = 896
minimum_frequency = 20
roll_off = 0.4  # Roll off duration of tukey window in seconds, default is 0.4s
duration = 4  # Analysis segment duration
post_trigger_duration = 2  # Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time

channel_dict = {'H1':'GWOSC',
                'L1':'GWOSC'}


class TransdimensionalConditionalLogUniform(tbilby.core.prior.TransdimensionalConditionalLogUniform):
   # one must define the transdimensional_condition_function function, so we know what to do with conditional variables...  
    # it is an abstract function, without it you cant instantiate this class 
    def transdimensional_condition_function(self,**required_variables):
        ''' setting the minimum according the the last peak value of the gaussian.
        Here you refer to the parameters you are 
        working with '''
        # mu is returned as an array 
        maximum = self.maximum
        if(len(self.amplitude)>0): # handle the first mu case
            maximum = self.amplitude[-1]
            below = maximum < self.minimum
            try:
                maximum[below] = self.maximum[0]
            except:
                maximum[below] = self.maximum
        setattr(self,'maximum',maximum)  # setting the atribute of the class
            
        return dict(maximum=maximum)

    
# Here we define out source model - this is the sine-Gaussian model in the
# frequency domain.
def sine_gaussian(frequency_array, amplitude, f0, Q, phi0, dt, e):
    r"""
    Our custom source model, this is just a Gaussian in frequency with
    variable global phase.

    .. math::

        \tilde{h}_{\plus}(f) = \frac{A \tau}{2\sqrt{\pi}}}
        e^{- \pi \tau (f - f_{0})^2 + i \phi_{0}} \\
        \tilde{h}_{\times}(f) = \tilde{h}_{\plus}(f) e^{i \pi / 2}


    Parameters
    ----------
    frequency_array: array-like
        The frequencies to evaluate the model at. This is required for all
        Bilby source models.
    amplitude: float
        An overall amplitude prefactor.
    f0: float
        The central frequency.
    tau: float
        The damping rate.
    phi0: float
        The reference phase.

    Returns
    -------
    dict:
        A dictionary containing "plus" and "cross" entries.

    """
    tau =  Q / (2*np.pi*f0) 
    arg = -((np.pi * tau * (frequency_array - f0)) ** 2) 
    plus = np.sqrt(np.pi) * amplitude * tau * np.exp(arg) * (np.exp(1j *phi0)+np.exp(-1j*phi0)*np.exp(-Q**2*frequency_array/f0))/ 2.0
    cross = e * plus * np.exp(1j * np.pi / 2)
    
    plus *= np.exp(-2j*frequency_array*np.pi*dt)
    cross *= np.exp(-2j*frequency_array*np.pi*dt)
    return {"plus": plus, "cross": cross}


def signal_model(frequency_array, n, amplitude0, amplitude1 , amplitude2, amplitude3, f0, f1, f2, f3, Q, Q1, Q2, Q3,
                 phi0, phi1, phi2, phi3, dt1, dt2, dt3, e, **kwargs):
    ndim = n
    model = {}
    model["plus"] = np.zeros(frequency_array.shape, dtype='complex128')
    model["cross"] = np.zeros(frequency_array.shape, dtype='complex128')
    amplitude = [amplitude0, amplitude1, amplitude2, amplitude3]
    f0 = [f0, f1, f2, f3]
    Q = [Q, Q1, Q2, Q3]
    phi0 = [phi0, phi1, phi2, phi3]
    dt = [0, dt1, dt2, dt3]
    for i in range(int(n)):
        model["plus"] += sine_gaussian(frequency_array, amplitude[i], f0[i], Q[i], phi0[i], dt[i], e)["plus"] 
        model["cross"] += sine_gaussian(frequency_array, amplitude[i], f0[i], Q[i], phi0[i], dt[i], e)["cross"] 
    
    return {"plus":model["plus"], "cross":model["cross"]}


# Now we pass our source function to the WaveformGenerator
waveform_arguments = dict(
    minimum_frequency=minimum_frequency,
    maximum_frequency=maximum_frequency,
)
waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=signal_model,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# We now use gwpy to obtain analysis and psd data and create the ifo_list
ifo_list = bilby.gw.detector.InterferometerList([])
for det in detectors:
    logger.info("Downloading analysis data for ifo {}".format(det))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    data = TimeSeries.fetch_open_data(det, start_time, end_time)
    ifo.strain_data.set_from_gwpy_timeseries(data)

    logger.info("Downloading psd data for ifo {}".format(det))
    psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time)
    psd_alpha = 2 * roll_off / duration
    psd = psd_data.psd(
        fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median"
    )
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value
    )
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency
    ifo_list.append(ifo)


logger.info("Saving data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)


# Here we define the priors
priors = bilby.core.prior.PriorDict(filename="n4.prior")

priors["geocent_time"] = bilby.core.prior.Uniform(
    trigger_time - 0.5, trigger_time + 0.5, name="geocent_time"
)
priors =tbilby.core.base.create_transdimensional_priors(transdimensional_prior_class=TransdimensionalConditionalLogUniform,\
                                                          param_name='amplitude',\
                                                          nmax= 4,\
                                                          nested_conditional_transdimensional_params=['amplitude'],\
                                                          conditional_transdimensional_params=[],\
                                                          conditional_params=[],\
                                                          prior_dict_to_add=priors,\
                                                          SaveConditionFunctionsToFile=False,\
                                                          minimum= 1e-23,maximum=1e-18)

priors['dt1']=MarginalizedTruncatedHollowedGaussian(condition_func_t1, alpha=2, beta=0.5, sigma_t=0.4, sigma_f=40, minimum_t=-1, maximum_t=1, minimum_f=20, maximum_f=250, n=1)
priors['dt2']=MarginalizedTruncatedHollowedGaussian(condition_func_t2, alpha=2, beta=0.5, sigma_t=0.4, sigma_f=40, minimum_t=-1, maximum_t=1, minimum_f=20, maximum_f=250, n=2)
priors['dt3']=MarginalizedTruncatedHollowedGaussian(condition_func_t3, alpha=2, beta=0.5, sigma_t=0.4, sigma_f=40, minimum_t=-1, maximum_t=1, minimum_f=20, maximum_f=250, n=3)

priors['f0']=bilby.core.prior.Uniform(minimum = 20, maximum = 250)
priors['f1']=ConditionalTruncatedHollowedGaussian(condition_func_f1, alpha=2, beta=0.5, sigma_t=0.4, sigma_f=40, minimum_t=-1, maximum_t=1, minimum_f=20, maximum_f=250, n=1)
priors['f2']=ConditionalTruncatedHollowedGaussian(condition_func_f2, alpha=2, beta=0.5, sigma_t=0.4, sigma_f=40, minimum_t=-1, maximum_t=1, minimum_f=20, maximum_f=250, n=2)
priors['f3']=ConditionalTruncatedHollowedGaussian(condition_func_f3, alpha=2, beta=0.5, sigma_t=0.4, sigma_f=40, minimum_t=-1, maximum_t=1, minimum_f=20, maximum_f=250, n=3)

priors['n'] = tbilby.core.prior.DiscreteUniform(0,4,'n_dimension')
priors=ConditionalPriorDict(priors)

likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifo_list, waveform_generator=waveform_generator
)

result = bilby.core.sampler.run_sampler(
    likelihood,
    priors,
    sampler="dynesty",
    sample="rwalk_dynesty",
    nlive = 6000,
    nact = 25,
    walks = 600,
    outdir=outdir,
    label=label,
    resume=True,
    npool=32,
)

