import numpy as np
import matplotlib.pylab as plt
import pandas as pd

####

def bgls(t, y, err, plow= 0.5, phigh= 100, ofac= 10):
    # Copyright (c) 2014-2015 Annelies Mortier, João Faria
    # with some of my tweaks
    # from BGLS: A Bayesian formalism for the generalised Lomb-Scargle periodogram
    # by A Mortier, J P Faria, C M Correia, A Santerne, and N C Santos
    eps= np.finfo(float).eps
    n_steps= int(ofac*len(t)*(1/plow - 1/phigh))
    f= np.linspace(1/phigh, 1/plow, n_steps)
    omegas= 2*np.pi*f
    err2= err*err
    w= 1/err2
    W= sum(w)
    bigY= sum(w*y)  # Eq. (10)
    p, constants, exponents = [], [], []

    for i, omega in enumerate(omegas):
        theta= 0.5*np.arctan2(sum(w*np.sin(2.*omega*t)), sum(w*np.cos(2.*omega*t)))
        x= omega*t - theta
        cosx= np.cos(x)
        sinx= np.sin(x)
        wcosx= w*cosx
        wsinx= w*sinx
        C= sum(wcosx)
        S= sum(wsinx)
        YCh= sum(y*wcosx)
        YSh= sum(y*wsinx)
        CCh= sum(wcosx*cosx)
        SSh= sum(wsinx*sinx)
        if np.abs(CCh) > eps and np.abs(SSh) > eps:
            K= (C*C*SSh + S*S*CCh - W*CCh*SSh)/(2*CCh*SSh)
            L= (bigY*CCh*SSh - C*YCh*SSh - S*YSh*CCh)/(CCh*SSh)
            M= (YCh*YCh*SSh + YSh*YSh*CCh)/(2*CCh*SSh)
            constants.append(1/np.sqrt(CCh*SSh*abs(K)))
        elif np.abs(CCh) <= eps:
            K= (S*S - W*SSh)/(2*SSh)
            L= (bigY*SSh - S*YSh)/(SSh)
            M= (YSh*YSh)/(2*SSh)
            constants.append(1/np.sqrt(SSh*abs(K)))
        elif np.abs(SSh) <= eps:
            K= (C*C - W*CCh)/(2*CCh)
            L= (bigY*CCh - C*YCh)/(CCh)
            M= (YCh*YCh)/(2*CCh)
            constants.append(1/np.sqrt(CCh*abs(K)))
        if K > 0:
            raise RuntimeError('K is positive. This should not happen.')
        else:
            exponents.append(M - L*L/(4*K))
    constants= np.array(constants)
    exponents= np.array(exponents)
    logp= np.log10(constants) + exponents*np.log10(np.exp(1))
    p= 10**(logp - np.max(logp))
    p= np.array(p)/max(p)  # normalize

    return 1/f, p

####

if False:
    # test
    t= np.arange(0, 300, 3)*0.1
    pertest= 15.2
    sig= 0.2
    y= 1.4 + np.sin(2*np.pi/pertest*t) + np.random.normal(0, sig, len(t))
    per, prob= bgls(t, y, sig*np.ones(len(t)), plow= 1, phigh= 30)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t, y, '.')
    plt.title('test: period= ' + str(pertest))
    plt.subplot(2,1,2)
    plt.plot(per, prob)
    plt.xlabel('period (hours)')
    plt.ylabel('probability')
    plt.show()




# Arin's data
df= pd.read_csv('Flavinexpostest3_subtracted.csv')
d= df.values
for i in range(1000, 1040, 2):
    y= d[i,2:]
    t= np.arange(len(y))*2.5/60
    # assume Poisson error
    err= np.ones(len(y))*np.sqrt(np.max(y))
    # Bayesian GLS periodogram
    per, prob= bgls(t, y, err, plow= 1, phigh= 5)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t, y)
    plt.title('time series ' + str(i))
    plt.subplot(2,1,2)
    plt.plot(per, prob)
    plt.xlabel('period (hours)')
    plt.ylabel('probability')
    plt.show()
