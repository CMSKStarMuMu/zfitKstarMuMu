import zfit
from package.fitcore.genfit import genfitSi,genfitPi
from package.func.dataimport import gendata
from package.func.plot import genplot
import time
import sys

if len(sys.argv) !=3:
    print ('Please run this code as: python *.py q2Bin pdfbase')
    print ('q2Bin: 0,1,2,3,5,7')
    print ('pdfbase: Si,Pi')
    exit()
binnum = int(sys.argv[1])
pdfbase = sys.argv[2]
if binnum not in [0,1,2,3,5,7]:
    print ('Please input right q2Bin num')
    print ('q2Bin: 0,1,2,3,5,7')
    exit()
if pdfbase not in ['Si','Pi']:
    print ('Please input right pdfbase')
    print ('pdfbase: Si,Pi')
    exit()
print ('We are going to fit GEN-LEVEL bin ',binnum, ' in pdfbase ', pdfbase)
filename = '/afs/cern.ch/user/x/xuqin/data/effKDE/2016/lmnr/newphi/effDataset_b{}_2016.root'.format(binnum)
data = gendata(filename=filename,q2Bin=binnum,nSample=0)
if pdfbase=='Si':
    pdf = genfitSi(data=data)
elif pdfbase=='Pi':
    pdf = genfitPi(data=data)    
genplot(pdf=pdf,data=data,pdfbase=pdfbase,q2Bin=binnum,n_bins=50)
print ('We have finished fitting GEN-LEVEL bin ',binnum, ' in pdfbase ', pdfbase)

