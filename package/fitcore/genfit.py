from ..pdf.genpdfSi import PDF_gen_Si
from ..pdf.genpdfPi import PDF_gen_Pi
from ..func.dataimport import gendata
from ..func.obs import getobs
import zfit
from math import pi,sqrt
import time
def genfitSi(data):
    print ('Begin fit')
    obs = getobs()
    Fl=zfit.Parameter("Fl",0.5,0.,1.)
    S3=zfit.Parameter("S3",0.,-1.,1.)
    S4=zfit.Parameter("S4",0.,-1.,1.)
    S5=zfit.Parameter("S5",0.,-1.,1.)
    AFB=zfit.Parameter("AFB",0.,-1.,1.)
    S7=zfit.Parameter("S7",0.,-1.,1.)
    S8=zfit.Parameter("S8",0.,-1.,1.)
    S9=zfit.Parameter("S9",0.,-1.,1.)
    pdf = PDF_gen_Si(obs=obs,Fl=Fl,S3=S3,S4=S4,S5=S5,AFB=AFB,S7=S7,S8=S8,S9=S9)
    print ('pdf.params',pdf.params)
    minimizer0 = zfit.minimize.Minuit(minimize_strategy=0)
    nll = zfit.loss.UnbinnedNLL(model=pdf, data=data)
    time_start = time.time()
    result = minimizer0.minimize(nll)
    time_end = time.time()
    time_c= time_end - time_start 
    print ('time used in fit minimize_strategy=0: ',time_c, 's')
    params = result.params
    result.hesse()
    print(params)
    minimizer2 = zfit.minimize.Minuit(minimize_strategy=2)
    time_start = time.time()
    result = minimizer2.minimize(nll)
    time_end = time.time()
    time_c= time_end - time_start 
    print ('time used in fit minimize_strategy=2: ',time_c, 's')
    params = result.params
    result.hesse()
    print(params)
    print ('fit finished')
    print ('afterfit pdf.params',pdf.params) 
    return pdf

def genfitPi(data):
    print ('Begin fit')
    obs = getobs()
    Fl=zfit.Parameter("Fl",0.5,0.,1.)
    P1=zfit.Parameter("P1",0.,-1.,1.)
    P2=zfit.Parameter("P2",0.,-0.5,0.5)
    P3=zfit.Parameter("P3",0.,-0.5,0.5)
    P4p=zfit.Parameter("P4p",0.,-1*sqrt(2),sqrt(2))
    P5p=zfit.Parameter("P5p",0.,-1*sqrt(2),sqrt(2))
    P6p=zfit.Parameter("P6p",0.,-1*sqrt(2),sqrt(2))
    P8p=zfit.Parameter("P8p",0.,-1*sqrt(2),sqrt(2))
    pdf = PDF_gen_Pi(obs=obs,Fl=Fl,P1=P1,P2=P2,P3=P3,P4p=P4p,P5p=P5p,P6p=P6p,P8p=P8p)
    print ('pdf.params',pdf.params)
    minimizer0 = zfit.minimize.Minuit(minimize_strategy=0)
    nll = zfit.loss.UnbinnedNLL(model=pdf, data=data)
    time_start = time.time()
    result = minimizer0.minimize(nll)
    time_end = time.time()
    time_c= time_end - time_start 
    print ('time used in fit minimize_strategy=0: ',time_c, 's')
    params = result.params
    result.hesse()
    print(params)
    minimizer2 = zfit.minimize.Minuit(minimize_strategy=2)
    time_start = time.time()
    result = minimizer2.minimize(nll)
    time_end = time.time()
    time_c= time_end - time_start 
    print ('time used in fit minimize_strategy=2: ',time_c, 's')
    params = result.params
    result.hesse()
    print(params)
    print ('fit finished')
    print ('afterfit pdf.params',pdf.params) 
    return pdf


