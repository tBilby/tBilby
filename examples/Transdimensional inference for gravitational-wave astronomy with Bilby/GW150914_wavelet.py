import tbilby
import bilby
import numpy as np
from scipy.interpolate import interp1d

from tbilby.core.prior.order_stats import TransdimensionalConditionalDescendingOrderStatPrior
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

# We now use gwpy to obtain analysis and psd data and create the ifo_list
ifo_list = bilby.gw.detector.InterferometerList([])
psd_list = []
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
    
    psd_list.append(psd)
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value
    )

    #psd_list.append(ifo.power_spectral_density_array)
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency
    ifo_list.append(ifo)


def antenna_conversion_SNR_to_amplitude(SNR, Q, f, e, ra, dec, geocent_time, psi, ifo_list=ifo_list):
    total = 0
    for detector in ifo_list:
        coefficient = 0
        coefficient += detector.antenna_response(ra, dec, geocent_time, psi, 'plus')**2
        coefficient += e**2 * detector.antenna_response(ra,dec,geocent_time, psi, 'cross')**2
        coefficient *= Q/(2*np.sqrt(2*np.pi)*f*detector.power_spectral_density.power_spectral_density_interpolated(f))
        total += coefficient
    return SNR/np.sqrt(total)

def signal_model(frequency_array, n, SNR0, SNR1, SNR2, SNR3, SNR4, SNR5, SNR6, SNR7, f0, f1, f2, f3, f4, f5, f6, f7, Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7,
                 phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7, dt1, dt2, dt3, dt4, dt5, dt6, dt7, geocent_time, psi, ra, dec, e, **kwargs):
    ndim = n
    model = {}
    model["plus"] = np.zeros(frequency_array.shape, dtype='complex128')
    model["cross"] = np.zeros(frequency_array.shape, dtype='complex128')
    SNR = [SNR0, SNR1, SNR2, SNR3, SNR4, SNR5, SNR6, SNR7]
    f = [f0, f1, f2, f3, f4, f5, f6, f7]
    Q = [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]
    phi= [phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7]
    dt = [0, dt1, dt2, dt3, dt4, dt5, dt6, dt7]
    for i in range(int(n)):
        A = antenna_conversion_SNR_to_amplitude(SNR[i], Q[i], f[i], e, ra, dec, geocent_time+dt[i], psi)
        sig = sine_gaussian(frequency_array, A, f[i], Q[i], phi[i], dt[i], e)
        model["plus"] += sig["plus"] 
        model["cross"] += sig["cross"] 
    
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

# Here we define the priors
class TransdimensionalConditionalDescendingOrderStatPriorSNR(TransdimensionalConditionalDescendingOrderStatPrior):
   
    def transdimensional_condition_function(self,**required_variables):
        
        #len(self.SNR.shape[1])
        if len(self.SNR)>0:
            self._prev_val=self.SNR[-1]
            self._this_order_num = self.SNR.shape[0]+1
        else:
            self.this_order_num = 1
            # to handle the first parameter when _pre_val is the pre set int
            if isinstance(self.n, np.ndarray):
                self._prev_val = self.minimum * np.ones(self.n.shape)
        try:
            self._tot_order_num=self.n.astype(int)
        except:
            self._tot_order_num=int(self.n)
        return dict(_prev_val=self._prev_val,_this_order_num=self._this_order_num, _tot_order_num=self._tot_order_num)

# prameter_prior
priors = bilby.core.prior.dict.ConditionalPriorDict()
priors = tbilby.core.base.create_transdimensional_priors(transdimensional_prior_class=TransdimensionalConditionalDescendingOrderStatPriorSNR,\
                                                          param_name='SNR',\
                                                          nmax= 8,\
                                                          nested_conditional_transdimensional_params=['SNR'],\
                                                          conditional_transdimensional_params=[],\
                                                          conditional_params=['n'],\
                                                          prior_dict_to_add=priors,\
                                                          SaveConditionFunctionsToFile=False,\
                                                          minimum= 0,maximum=30,prev_val=30,this_order_num=1)

priors["geocent_time"] = bilby.core.prior.Uniform(
    trigger_time - 0.3, trigger_time + 0.2, name="geocent_time")

priors['dt1']= bilby.core.prior.Uniform( -0.3, 0.2, name="dt1")
priors['dt2']= bilby.core.prior.Uniform( -0.3, 0.2, name="dt2")
priors['dt3']= bilby.core.prior.Uniform( -0.3, 0.2, name="dt3")
priors['dt4']= bilby.core.prior.Uniform( -0.3, 0.2, name="dt4")
priors['dt5']= bilby.core.prior.Uniform( -0.3, 0.2, name="dt5")
priors['dt6']= bilby.core.prior.Uniform( -0.3, 0.2, name="dt6")
priors['dt7']= bilby.core.prior.Uniform( -0.3, 0.2, name="dt7")

priors['f0']=bilby.core.prior.Uniform(minimum = 20, maximum = 512)
priors['f1']=bilby.core.prior.Uniform(minimum = 20, maximum = 512)
priors['f2']=bilby.core.prior.Uniform(minimum = 20, maximum = 512)
priors['f3']=bilby.core.prior.Uniform(minimum = 20, maximum = 512)
priors['f4']=bilby.core.prior.Uniform(minimum = 20, maximum = 512)
priors['f5']=bilby.core.prior.Uniform(minimum = 20, maximum = 512)
priors['f6']=bilby.core.prior.Uniform(minimum = 20, maximum = 512)
priors['f7']=bilby.core.prior.Uniform(minimum = 20, maximum = 512)

priors["e"] = bilby.core.prior.Uniform(name='e', minimum=-1, maximum=1)
priors["psi"] = bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
priors["phi0"] = bilby.core.prior.Uniform(name='phi0', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors["phi1"] = bilby.core.prior.Uniform(name='phi1', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors["phi2"] = bilby.core.prior.Uniform(name='phi2', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors["phi3"] = bilby.core.prior.Uniform(name='phi3', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors["phi4"] = bilby.core.prior.Uniform(name='phi4', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors["phi5"] = bilby.core.prior.Uniform(name='phi5', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors["phi6"] = bilby.core.prior.Uniform(name='phi6', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors["phi7"] = bilby.core.prior.Uniform(name='phi7', minimum=0, maximum=2 * np.pi, boundary='periodic')

priors["Q0"] = bilby.core.prior.Uniform(name='Q0', minimum=0.1, maximum=40)
priors["Q1"] = bilby.core.prior.Uniform(name='Q1', minimum=0.1, maximum=40)
priors["Q2"] = bilby.core.prior.Uniform(name='Q2', minimum=0.1, maximum=40)
priors["Q3"] = bilby.core.prior.Uniform(name='Q3', minimum=0.1, maximum=40)
priors["Q4"] = bilby.core.prior.Uniform(name='Q4', minimum=0.1, maximum=40)
priors["Q5"] = bilby.core.prior.Uniform(name='Q5', minimum=0.1, maximum=40)
priors["Q6"] = bilby.core.prior.Uniform(name='Q6', minimum=0.1, maximum=40)
priors["Q7"] = bilby.core.prior.Uniform(name='Q7', minimum=0.1, maximum=40)
priors["dec"] = bilby.core.prior.Cosine(name='dec')
priors["ra"] = bilby.core.prior.Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')

priors['n'] = tbilby.core.prior.DiscreteUniform(1,8,'n_dimension')

likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifo_list, waveform_generator=waveform_generator
)

result = bilby.core.sampler.run_sampler(
    likelihood,
    priors,
    sampler="dynesty",
    sample="rwalk",
    nlive = 2000,
    nact = 80,
    outdir=outdir,
    label=label,
    resume=True,
    npool=16,
)

