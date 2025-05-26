import torch

def sample_latent(mu, logvar, dist_type='gaussian', df=3.0):
    """
    sampling function
    :param mu: mean tensor
    :param logvar: log variance tensor
    :param dist_type: (str) probability distribution type (either 'gaussian' or 'student_t')
    :param df: (float) degrees of freedom
    :return:
    """
    if dist_type == 'gaussian':
        return sample_gaussian(mu, logvar)
    elif dist_type == 'student_t':
        return sample_student_t(mu, logvar, df=df)
    else:
        raise ValueError('Unsupported distribution type: {}'.format(dist_type))

def sample_gaussian(mu, logvar):
    """
    gaussian sampling
    :param mu: mean tensor
    :param logvar: log variance tensor
    :return:
    """
    std = torch.exp(0.5 * logvar)
    std = torch.clamp(std, min=1e-2, max=2.0)
    eps = torch.randn_like(std)

    return mu + eps * std

def sample_student_t(mu, logvar, df=3.0):
    """
    Sample from Student-t distribution using Gaussian + Chi-squared trick
    :param mu: mean tensor
    :param logvar: log variance tensor
    :param df: degrees of freedom
    :return: samples from Student-t distribution
    """
    std = torch.exp(0.5 * logvar)   # control std * t_sample
    std = torch.clamp(std, max=5.0) # prevent denominator from approaching 0

    eps = torch.randn_like(std)
    chi2 = torch.distributions.Chi2(df).sample(std.shape).to(std.device)
    chi2 = torch.clamp(chi2, min=1e-4) # avoid chi2 overflow

    t_sample = eps / torch.sqrt(chi2 / df)
    t_sample = torch.clamp(t_sample, min=-3.0, max=3.0) # suppress tail outlier

    return mu + std * t_sample