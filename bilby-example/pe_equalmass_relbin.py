import bilby
import numpy as np
from gwbench import basic_relations as bsr
from gwbench import injections
from astropy.cosmology import Planck18

# Parameters
z                  = 2.
DL                 = Planck18.luminosity_distance(z).value
m1                 = 300. * (1.+z)
m2                 = 299. * (1.+z)
Mc, eta            = bsr.Mc_eta_of_m1_m2(m1,m2)
q                  = bsr.q_of_eta(eta)
chi1               = 0.1
chi2               = 0.1

# Sampling angles
seed               = 120597
rngs               = [np.random.default_rng(seeed) for seeed in np.random.default_rng(seed).integers(100000,size=4)]
angles             = injections.angle_sampler(1,seed)
iota               = angles[0][0]
ra                 = angles[1][0]
dec                = angles[2][0]
psi                = angles[3][0]
theta1             = np.arcsin(rngs[0].uniform(-1,1))
theta2             = np.arcsin(rngs[1].uniform(-1,1))
phi1               = rngs[2].uniform(0,2*np.pi)
phi2               = rngs[3].uniform(0,2*np.pi)

# Specify the output directory and the name of the simulation.
outdir = "results_equalmass_relbin"
label = "equalmass"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility. 
bilby.core.utils.random.seed(88170235)

injection_parameters = dict(
    mass_1               = m1,
    mass_2               = m2,
    a_1                  = chi1, # The labels chi1,2 are for non-precessing waveforms 
    a_2                  = chi2,
    tilt_1               = theta1,
    tilt_2               = theta2,
    phi_12               = phi1,
    phi_jl               = phi2,
    luminosity_distance  = DL, # Distance in Mpc (like gwbench)
    theta_jn             = iota, # This is iota for non-precessing systems 
    psi                  = psi,
    ra                   = ra,
    dec                  = dec,
    phase                = 0., 
    geocent_time         = 0.,
    fiducial             = 1,
)

# Set the duration and sampling frequency of the data segment
sampling_frequency = 2048.0
minimum_frequency  = 3.
duration           = bilby.gw.detector.get_safe_signal_duration(m1,m2,0,0,0,0,flow=minimum_frequency)

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant ="IMRPhenomXPHM",
    reference_frequency  = 3.0,
    minimum_frequency    = minimum_frequency,
)

# Create the waveform_generator using a LAL BinaryBlackHole source function
duration = duration + 2
waveform_generator = bilby.gw.WaveformGenerator(
    duration                      = duration,
    sampling_frequency            = sampling_frequency,
    frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole_relative_binning,
    parameter_conversion          = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments            = waveform_arguments,
)

# Set up interferometers and inject signal.
ifos = bilby.gw.detector.InterferometerList(["CE40A", "CE20B", "ET"])
for ifo in ifos:
    ifo.minimum_frequency = minimum_frequency

ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency = sampling_frequency,
    duration           = duration,
    start_time         = injection_parameters["geocent_time"] - 2,
)

ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

# Set up a PriorDict, which inherits from dict. If we do nothing, then the default priors get used.
priors = bilby.gw.prior.BBHPriorDict()

del priors['mass_1'], priors['mass_2'], priors['chirp_mass']

priors['total_mass']          = bilby.core.prior.Uniform(minimum = (m1+m2) - 0.3 * (m1+m2), maximum = (m1+m2) + 0.3 * (m1+m2), name = 'total_mass')
priors['mass_ratio']          = bilby.gw.prior.Uniform(minimum=0.125, maximum=1, name='mass_ratio')
priors['luminosity_distance'] = bilby.gw.prior.Uniform(minimum=9000, maximum=21000, name='luminosity_distance')

time_delay = ifos[0].time_delay_from_geocenter(
    injection_parameters["ra"],
    injection_parameters["dec"],
    injection_parameters["geocent_time"],
)

priors["CE40A_time"] = bilby.core.prior.Uniform(
    minimum=injection_parameters["geocent_time"] + time_delay - 0.1,
    maximum=injection_parameters["geocent_time"] + time_delay + 0.1,
    name="CE40A_time",
    latex_label="$t_CE40A$",
    unit="$s$",
)

# Perform a check that the prior does not extend to a parameter space longer than the data
priors.validate_prior(duration, minimum_frequency)

# Set up the fiducial parameters for the relative binning likelihood to be the
# injected parameters. Note that because we sample in chirp mass and mass ratio
# but injected with mass_1 and mass_2, we need to convert the mass parameters
fiducial_parameters = injection_parameters.copy()
m1 = fiducial_parameters.pop("mass_1")
m2 = fiducial_parameters.pop("mass_2")
fiducial_parameters["total_mass"] = bilby.gw.conversion.component_masses_to_total_mass(m1,m2)
fiducial_parameters["mass_ratio"] = m2 / m1
fiducial_parameters['CE40A_time'] = injection_parameters["geocent_time"] + time_delay

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveform generator
likelihood = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
    interferometers          = ifos,
    waveform_generator       = waveform_generator,
    priors                   = priors,
    distance_marginalization = False,
    time_marginalization     = False,
    time_reference           = "CE40A",
    fiducial_parameters      = fiducial_parameters,
)

# Run sampler.  In this case, we're going to use the `nestle` sampler
result = bilby.run_sampler(
    likelihood           = likelihood,
    priors               = priors,
    sampler              = "dynesty",
    npoints              = 1000,
    npool                = 48,
    injection_parameters = injection_parameters,
    outdir               = outdir,
    label                = label,
)


