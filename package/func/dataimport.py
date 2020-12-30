###this code is used to convert RooDataSet to pandas.dataframe
import zfit
from ROOT import RooDataSet,TFile,RooWorkspace,RooRealVar,RooArgList,RooArgSet
import pandas as pd
from .obs import getobs
import time
genStat = [1700246,3190316,2416720,4413921,1,5184529,1,3094894,1]
dataStat = [994,2288,1979,3403,0,5381,0,3506,0]
d16Stat = [246,605,519,873,0,1301,0,908,0]

###import gen-level fit samples
def gendata(filename,q2Bin,nSample):
    time_start = time.time()
    obs = getobs()
    file = TFile.Open(filename)
    if not file:
        print ('File not found: ', filename)
        return
    wspname = 'ws_b{}p0'.format(q2Bin)
    wsp = file.Get(wspname)
    if not wsp:
        print ('Workspace not found in file: ', filename)
        return
    ctK = wsp.var('ctK')
    ctL = wsp.var('ctL')
    phi = wsp.var('phi')
    rand = wsp.var("rand")
    if (not ctK) or (not ctL) or (not phi):
        print ('Variables not found in file: ', filename)
        return
    vars = RooArgList(ctK,ctL,phi)
    datasetname = 'data_genDen_ev_b{}'.format(q2Bin)
    fullData = wsp.data(datasetname)
    if not fullData:
        print ('DataSet ', datasetname, ' not found in file: ', filename)
        return
    data = RooDataSet()
    if nSample==0:
        data = fullData
    elif nSample<=10000:
        data = fullData.reduce(RooArgSet(vars),'rand > {:1.6f} && rand < {:1.6f}'.format((nSample-1)*dataStat[q2Bin]/genStat[q2Bin],nSample*dataStat[q2Bin]/genStat[q2Bin]))
    else:
        data = fullData.reduce(RooArgSet(vars),'rand > {:.6f} && rand < {:.6f}'.format((nSample%10000-1)*d16Stat[q2Bin]/genStat[q2Bin],(nSample%10000)*d16Stat[q2Bin]/genStat[q2Bin]))
    print ('Fit ', data.numEntries(), ' events')
    ###convert the final dataset to pd.dataframe
    datalist = []
    for i in range(0,data.numEntries()):
        ctKvalue = data.get(i).getRealValue('ctK')
        ctLvalue = data.get(i).getRealValue('ctL')
        phivalue = data.get(i).getRealValue('phi')
        datalist.append([ctKvalue,ctLvalue,phivalue])
    #print ('lenth of datalist ', len(datalist))
    finaldata_df = pd.DataFrame(datalist,columns=['costk','costl','phi'])
    finaldata = zfit.Data.from_pandas(finaldata_df,obs=obs)
    print ('generate zfit data from RooDataSet successfully')
    time_end = time.time()
    time_c= time_end - time_start 
    print ('time used in fit data import: ',time_c, 's')
    return finaldata









