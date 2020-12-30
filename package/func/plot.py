from .obs import getobs,getobslist
import numpy as np
from math import pi
import zfit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import mplhep as hep
plt.style.use(hep.style.CMS)
import time
def genplot(pdf,data,pdfbase,q2Bin,n_bins):
    time_start = time.time()
    obs = getobs()
    obs_list = getobslist()
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
        plt.savefig("/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/zfit/final/plots/gentotal/pdf{}{}proBin{}.png".format(pdfbase,angular[i],q2Bin))
    time_end = time.time()
    time_c= time_end - time_start 
    print ('time used in plot fit result: ',time_c, 's')

