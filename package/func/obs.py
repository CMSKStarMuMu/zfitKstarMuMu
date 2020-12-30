import zfit
from math import pi

costk=zfit.Space('costk',(-1,1))
costl=zfit.Space('costl',(-1,1))
phi=zfit.Space('phi',(-pi,pi))
def getobs():
    obs=costk*costl*phi
    return obs

def getobslist():
    obslist = [costk,costl,phi]
    return obslist