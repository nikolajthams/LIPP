import numpy as np

def likelihood(beta, x_direct, x_integrated, scale=1, dx=None, kappa=1, Omega=None, phi_inv=lambda x: x, dphi_inv=lambda x: 1):
    l = np.log(phi_inv(x_direct.dot(beta))).sum() if x_direct.size > 0 else 0
    l -= phi_inv(x_integrated.dot(beta)).sum()*dx
    if Omega is not None: l -= kappa*beta.dot(Omega).dot(beta)
    return scale*l

def dlikelihood(beta, x_direct, x_integrated, scale=1, dx=None, kappa=1, Omega=None, phi_inv=lambda x: x, dphi_inv=lambda x: 1):
    l = (x_direct.T*(dphi_inv(x_direct.dot(beta))/phi_inv(x_direct.dot(beta)))).sum(axis=1) if x_direct.size > 0 else 0
    l -= (dphi_inv(x_integrated.dot(beta))*x_integrated.T).sum(axis=1)*dx
    if Omega is not None: l -= 2*kappa*Omega.dot(beta)
    return scale*l

def quad_loss(beta, x_dir, x_int, scale=1, dx=None, kappa=1, Omega=None, phi_inv=lambda x: x, dphi_inv=lambda x: 1):
    # Calculate value of loss function
    q = 1/2*np.power(phi_inv(beta.dot(x_int.T)),2).sum()*dx - phi_inv(x_dir.dot(beta)).sum()
    if Omega is not None: q += kappa*beta.dot(Omega).dot(beta)
    return scale*q

def dquad_loss(beta, x_dir, x_int, scale=1, dx=None, kappa=1, Omega=None, phi_inv=lambda x: x, dphi_inv=lambda x: 1):
    # Calculate gradient
    q=((phi_inv(x_int.dot(beta))*dphi_inv(x_int.dot(beta)))*x_int.T).sum(axis=1)*dx - (dphi_inv(x_dir.dot(beta))*x_dir.T).sum(axis=1)
    if Omega is not None: q += 2*kappa*Omega.dot(beta)
    return scale*q
