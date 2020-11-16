import zfit
import pandas as pd
from zfit import z
import tensorflow as tf
from tensorflow.math import cos,sin,sqrt
from math import pi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import uproot
from ROOT import TChain
from root_numpy import root2array,tree2array
import numpy as np
import time
import sys
import math
binborder = [1, 2, 4.3, 6, 8.68, 10.09, 12.86, 14.18, 16, 19]
##q2Bin border##
ncpu = 16
zfit.run.set_n_cpu(ncpu)
print ("cpu using:", ncpu)
##define cpu number using in zfit##

####define pdfs of Si base and Pi base####
class CustomPDFSi(zfit.pdf.ZPDF):
    ###Si base pdf###
    _PARAMS = ['FL','S3','S4','S5','AFB','S7','S8','S9']
    _N_OBS = 3

    def _unnormalized_pdf(self,x):
        x,y,z = x.unstack_x()
        #x:costk,y:costl,z:phi
        FL = self.params['FL']
        S3 = self.params['S3']
        S4 = self.params['S4']
        S5 = self.params['S5']
        AFB = self.params['AFB']
        S7 = self.params['S7']
        S8 = self.params['S8']
        S9 = self.params['S9']
        dcrate = (9/(pi*32)*(0.75*(1-FL)*(1-pow(x,2))
            +FL*pow(x,2)+0.25*(1-FL)*(1-pow(x,2))*(2*pow(y,2)-1)
            -FL*pow(x,2)*(2*pow(y,2)-1)+S3*(1-pow(x,2))*(1-pow(y,2))*cos(2*z)
            +S4*2*sqrt(1-pow(x,2))*x*2*sqrt(1-pow(y,2))*y*cos(z)
            +S5*2*sqrt(1-pow(x,2))*x*sqrt(1-pow(y,2))*cos(z)
            +4/3*AFB*(1-pow(x,2))*y+S7*2*sqrt(1-pow(x,2))*x*sqrt(1-pow(y,2))*sin(z)
            +S8*2*sqrt(1-pow(x,2))*x*2*sqrt(1-pow(y,2))*y*sin(z)
            +S9*(1-pow(x,2))*(1-pow(y,2))*sin(2*z)))
        return dcrate

class CustomPDFPi(zfit.pdf.ZPDF):
    ###Pi base pdf###
    _PARAMS = ['FL','P1','P2','P3','P4p','P5p','P6p','P8p']
    _N_OBS = 3

    def _unnormalized_pdf(self,x):
        x,y,z = x.unstack_x()
        #x:costk,y:costl,z:phi
        FL = self.params['FL']
        P1 = self.params['P1']
        P2 = self.params['P2']
        P3 = self.params['P3']
        P4p = self.params['P4p']
        P5p = self.params['P5p']
        P6p = self.params['P6p']
        P8p = self.params['P8p']
        FT = 1-FL
        dcrate = (9/(pi*32)*(0.75*FT*(1-x**2)+FL*x**2
            +0.25*FT*(1-x**2)*(2*y**2-1)-FL*x**2*(2*y**2-1)
            +0.5*P1*FT*(1-x**2)*(1-y**2)*cos(2*z)
            +sqrt(FT*FL)*(P4p*2*sqrt(1-x**2)*x*sqrt(1-y**2)*y*cos(z)
            +P5p*2*sqrt(1-x**2)*x*sqrt(1-y**2)*cos(z))
            -sqrt(FT*FL)*(P6p*2*sqrt(1-x**2)*x*sqrt(1-y**2)*sin(z)
            -P8p*sqrt(1-x**2)*x*2*sqrt(1-y**2)*y*sin(z))
            +2*P2*FT*(1-x**2)*y-P3*FT*(1-x**2)*(1-y**2)*sin(2*z)))
        return dcrate

###create zfit space###
costk=zfit.Space('cos_theta_k',(-1,1))
costl=zfit.Space('cos_theta_l',(-1,1))
phi=zfit.Space('phi_kst_mumu',(-pi,pi))
obs=costk*costl*phi
obs_list=[costk,costl,phi]
###prepare data sample###
def gendata(q2Bin):
    tree = TChain("ntuple")
    tree.Add('/afs/cern.ch/user/x/xuqin/data/2016/skims/GEN_NoFilter/GEN_BFilter_B0MuMuKstar_p*.root')
    #tree.Add('/Users/xuelong/cernbox/workdir/B0KstMuMu/zfit/gentree.root')
    array_data = tree2array(tree,branches=['cos_theta_k','cos_theta_l','phi_kst_mumu'],selection='genQ2>{} && genQ2<{} && genSignHasFSR<0.5'.format(binborder[q2Bin],binborder[q2Bin+1]))
    df_data = pd.DataFrame(array_data)
    data = zfit.Data.from_pandas(df_data)
    return data
###construct pdf###
def constructpdf(pdfbase):
    if pdfbase=='Si':
        FL=zfit.Parameter("FL",0.5,0.,1.)
        S3=zfit.Parameter("S3",0.,-1.,1.)
        S4=zfit.Parameter("S4",0.,-1.,1.)
        S5=zfit.Parameter("S5",0.,-1.,1.)
        AFB=zfit.Parameter("AFB",0.,-1.,1.)
        S7=zfit.Parameter("S7",0.,-1.,1.)
        S8=zfit.Parameter("S8",0.,-1.,1.)
        S9=zfit.Parameter("S9",0.,-1.,1.)
        pdf = CustomPDFSi(obs=obs,FL=FL,S3=S3,S4=S4,S5=S5,AFB=AFB,S7=S7,S8=S8,S9=S9)
        return pdf
    elif pdfbase=='Pi':
        FL=zfit.Parameter("FL",0.5,0.,1.)
        P1=zfit.Parameter("P1",0.,-1.,1.)
        P2=zfit.Parameter("P2",0.,-0.5,0.5)
        P3=zfit.Parameter("P3",0.,-0.5,0.5)
        P4p=zfit.Parameter("P4p",0.,-math.sqrt(2),math.sqrt(2))
        P5p=zfit.Parameter("P5p",0.,-math.sqrt(2),math.sqrt(2))
        P6p=zfit.Parameter("P6p",0.,-math.sqrt(2),math.sqrt(2))
        P8p=zfit.Parameter("P8p",0.,-math.sqrt(2),math.sqrt(2))
        pdf = CustomPDFPi(obs=obs,FL=FL,P1=P1,P2=P2,P3=P3,P4p=P4p,P5p=P5p,P6p=P6p,P8p=P8p)        
        return pdf
    else:
        print ('wrong pdfbase!!!')
        return None

def chooseminimizer(minimizer):
    if minimizer=='minuit':
        return zfit.minimize.Minuit()
    elif minimizer=='scipy':
        return zfit.minimize.Scipy()
    elif minimizer=='adam':
        return zfit.minimize.Adam()
    else:
        print ('wrong minimizer!!!')
        return None

def genfit(q2Bin,obs,pdfbase,minimizer):
    ##q2Bin: q2Bin of fit
    ##obs: Space of fit
    ##data: Data use in fit
    ##pdfbase: Si or Pi
    ##minimizer: minuit,scipy,adam
    print ('######################begin fit genMC sample#####################')
    print ('q2Bin is ', q2Bin)
    print ('pdf base is', pdfbase)
    print ('minimizer is ', minimizer)
    pdf = constructpdf(pdfbase)
    data = gendata(q2Bin)
    if pdf==None:
        print ('empty pdf!!!')
        return None
    nll = zfit.loss.UnbinnedNLL(model=pdf, data=data)
    minimizer_use = chooseminimizer(minimizer)
    if minimizer_use==None:
        print ('empty minimizer!!!')
        return None
    time_start = time.time()
    result = minimizer_use.minimize(nll)
    time_end = time.time()
    time_c= time_end - time_start 
    print('time cost in zfit:', time_c, 's')
    params = result.params
    print("Function minimum:", result.fmin)
    print("Converged:", result.converged)
    print("Full minimizer information:", result.info)
    print (params)
    param_errors, _ = result.errors(method='minuit_minos')
    for var, errors in param_errors.items():
        print('{}: ^{{+{}}}_{{-{}}}'.format(var.name, errors['upper'], errors['lower']))
    result.hesse()
    print(params)
    print('#######################finish fit genMC sample q2Bin{} {}#######################'.format(q2Bin,pdfbase))
    return pdf

def plot(q2Bin,obs,obs_list,pdf,n_bins,pdfbase):
    print('#######################begin plot fit result q2Bin{} {}#######################'.format(q2Bin,pdfbase))
    data = gendata(q2Bin)
    angular = ['costk','costl','phi']
    x_plot_costk = np.linspace(-1, 1, 1000)
    x_plot_costl = np.linspace(-1, 1, 1000)
    x_plot_phi = np.linspace(-pi, pi, 1000)

    x_plot = [x_plot_costk,x_plot_costl,x_plot_phi]
    lower,upper = obs.limits
    for i in range(3):
        x_plot_data = zfit.Data.from_numpy(array=x_plot[i], obs=obs_list[i])
        pdf_plot = pdf.partial_integrate(x=x_plot_data,
                limits=obs_list[(i+1)%3]*obs_list[(i+2)%3])
        y_plot = zfit.run(pdf_plot)
        data_np = data[:,i].numpy()
        n_sample = data_np.shape[0]
        print ('total entries is ', n_sample)
        plot_scaling = n_sample/n_bins * obs_list[i].area()
        color = 'black'
        linewidth = 2.5
        plt.figure()
        plt.title("Projection of {}".format(angular[i]))
        counts, bin_edges = np.histogram(data_np, n_bins,
                range=(lower[-1][i], upper[0][i]))
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
        err = np.sqrt(counts)
        counts_max = counts[np.argmax(counts)]
        plt.errorbar(bin_centres, counts, yerr=err, fmt='o', color='xkcd:black',label="GenMC sample")
        plt.plot(x_plot[i], y_plot * plot_scaling, label="Fit Result",linewidth=linewidth*2)
        plt.xlabel(angular[i])
        plt.ylim(ymin=0,ymax=1.2*counts_max)
        plt.legend()
        plt.savefig("pdf{}{}proBin{}.png".format(pdfbase,angular[i],q2Bin))
    print('#######################finish plot fit result q2Bin{} {}#######################'.format(q2Bin,pdfbase))

q2Binlist = [0,1,2,3,5,7]
pdfbaselist = ['Si','Pi']
minimizerlist = ['minuit','scipy','adam']
if len(sys.argv)<4:
    print ('Parameter missing!')
    print ('Please run this code by      python *.py q2Bin pdfbase minimizer')
    print ('q2Bin:0,1,2,3,5,7')
    print ('pdfbase:Si,Pi')
    print ('minimizer:minuit,scipy,adam')
        
elif int(sys.argv[1]) not in q2Binlist or sys.argv[2] not in pdfbaselist or sys.argv[3] not in minimizerlist:
    print ('Wrong parameters!')
    print ('Please run this code by      python *.py q2Bin pdfbase minimizer')
    print ('q2Bin:0,1,2,3,5,7')
    print ('pdfbase:Si,Pi')
    print ('minimizer:minuit,scipy,adam')

else:
    q2Bin = int(sys.argv[1])
    pdfbase = sys.argv[2]
    minimizer = sys.argv[3]
#    data = gendata(q2Bin)
    pdf = genfit(q2Bin,obs,pdfbase,minimizer)
    n_bins = 50
    plot(q2Bin,obs,obs_list,pdf,n_bins,pdfbase)
















