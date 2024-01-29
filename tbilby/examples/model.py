import numpy as np
def model(x,n_gauss,mu5,mu0,mu4,sigma_g,mu3,mu6,mu2,mu1,mu7):
	mu=[mu0,mu1,mu2,mu3,mu4,mu5,mu6,mu7]
	
	result=np.zeros(x.shape)
	
	for n in np.arange(int(np.round(n_gauss))):
				result +=globals()['gauss'](x,sigma_g=sigma_g,mu=mu[n]) 


	return result