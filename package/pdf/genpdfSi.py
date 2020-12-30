import zfit
import pandas as pd
from zfit import z
import tensorflow as tf
from tensorflow.math import cos,sin,sqrt
from math import pi, fabs
import numpy as np
import sys
import math
from sympy import assoc_legendre, symbols, lambdify
###This code is used to define Si base pdf of Gen-level###

def assoc_legendre_tf(n,m,x_tf):
    x_np = x_tf.numpy()
    x = symbols('x')
    func = assoc_legendre(n,m,x)
    value = lambdify(x,func,"numpy")
    result = value(x_np)
 #   tensor = tf.convert_to_tensor(value(x_np))
    return result


def PDF_gen_Si_func(costk, costl , phi, Fl, S3, S4, S5, AFB, S7, S8, S9):
    ##sint2k: square of sin(thetak); sin2tk: sin(2thetak); sintk: sin(thetak); 
    cost2k = costk * costk
    sint2k = 1 - cost2k
    sintk =  sqrt(sint2k)
    sin2tk = 2 * sintk * costk


    cost2l = costl * costl
    sint2l = 1 - cost2l
    sintl = sqrt(sint2l)
    sin2tl = 2 * sintl * costl
    cos2tl = 2 * cost2l - 1 

    sinphi = sin(phi)
    cosphi = cos(phi)
    cos2phi = cos(2*phi)
    sin2phi = sin(2*phi)

    dcrate = (0.75 * (1 - Fl) * sint2k + Fl * cost2k
    + 0.25 * (1 - Fl) * sint2k * cos2tl - Fl * cost2k * cos2tl
    + S3 * sint2k * sint2l * cos2phi + S4 * sin2tk * sin2tl * cosphi
    + S5 * sin2tk * sintl * cosphi + 4/3 * AFB * sint2k * costl
    + S7 * sin2tk * sintl * sinphi + S8 * sin2tk * sin2tl * sinphi
    + S9 * sint2k * sint2l * sin2phi)

    return 9 / (32 * 3.14159265) * dcrate

class PDF_gen_Si(zfit.pdf.ZPDF):
    _PARAMS = ['Fl','S3','S4','S5','AFB','S7','S8','S9']
    _N_OBS = 3
    def _unnormalized_pdf(self,x):
        costk,costl,phi = x.unstack_x()
        Fl = self.params['Fl']
        S3 = self.params['S3']
        S4 = self.params['S4']
        S5 = self.params['S5']
        AFB = self.params['AFB']
        S7 = self.params['S7']
        S8 = self.params['S8']
        S9 = self.params['S9']
        return PDF_gen_Si_func(costk, costl , phi, Fl, S3, S4, S5, AFB, S7, S8, S9)

###Define analytical integral###
#full space integral
def integralSi_full( x, limits, norm_range , params , model):
    return 9. / (32 * 3.14159265) * (4.0 / 9.0) * (2 * pi * 2 * 2)

lower_costk_costl_phi = ((-1,-1,-pi),)
upper_costk_costl_phi = ((1,1,pi),)
limits_costk_costl_phi = zfit.Space (axes =(0, 1 , 2), limits =(lower_costk_costl_phi, upper_costk_costl_phi))
PDF_gen_Si.register_analytic_integral (func = integralSi_full, limits = limits_costk_costl_phi)

#integral over costl and phi
def integralSi_costl_phi(x, limits, norm_range , params , model):
    costk = x.unstack_x()
    Fl = params['Fl']
    K00 = assoc_legendre_tf(0,0,costk)
    K20 = assoc_legendre_tf(2,0,costk)
    return 9. / (32 * 3.14159265) * (4.0 / 9.0* K00 + (4.0* Fl / 3.0 - 4.0/ 9.0) * K20 ) * (2 * pi * 2)

lower_costl_phi = ((-1,-pi),)
upper_costl_phi = ((1,pi),)
limits_costl_phi = zfit.Space (axes =(1 , 2), limits =(lower_costl_phi, upper_costl_phi))
PDF_gen_Si.register_analytic_integral (func = integralSi_costl_phi, limits = limits_costl_phi)

#integral over costk and phi
def integralSi_costk_phi(x, limits, norm_range , params , model):
    costl = x.unstack_x()
    Fl = params['Fl']
    AFB = params['AFB']
    L00 = assoc_legendre_tf(0,0,costl)
    L20 = assoc_legendre_tf(2,0,costl)
    L10 = assoc_legendre_tf(1,0,costl)
    return 9./(32 * 3.14159265) * (4.0/9.0*L00 + (2.0/9.0-2.0*Fl/3.0)*L20 + (4.0/3.0)*(2.0/3.0)*AFB*L10 ) * (2*pi*2)

lower_costk_phi = ((-1,-pi),)
upper_costk_phi = ((1,pi),)
limits_costk_phi = zfit.Space (axes =(0 , 2), limits =(lower_costk_phi, upper_costk_phi))
PDF_gen_Si.register_analytic_integral (func = integralSi_costk_phi, limits = limits_costk_phi)

#integral over costk and costl
def integralSi_costk_costl(x, limits, norm_range , params , model):
    phi = x.unstack_x()
    S3 = params['S3']
    S9 = params['S9']
    return 9./(32 * 3.14159265) * ( 4.0/9.0 + 1/18.0*2*S3*cos(2*phi)*2*2 + 1/9.0*S9*sin(2*phi)*2*2 ) * (2 * 2)

lower_costk_costl = ((-1,-1),)
upper_costk_costl = ((1,1),)
limits_costk_costl = zfit.Space (axes =(0 , 1), limits =(lower_costk_costl, upper_costk_costl))
PDF_gen_Si.register_analytic_integral (func = integralSi_costk_costl, limits = limits_costk_costl)

#integral over phi
def integralSi_phi(x, limits, norm_range, params, model):
    costk, costl = x.unstack_x()
    Fl = params['Fl']
    AFB = params['AFB']
    L00 = assoc_legendre_tf(0,0,costl)
    L10 = assoc_legendre_tf(1,0,costl)
    L20 = assoc_legendre_tf(2,0,costl)
    K00 = assoc_legendre_tf(0,0,costk)
    K20 = assoc_legendre_tf(2,0,costk)
    return  9./(32 * 3.14159265) * ( (4.0/9.0)*L00*K00 + (4.0*Fl/3.0-4.0/9.0)*L00*K20 + (2.0/9.0-2.0*Fl/3.0)*L20*K00 + (-2.0/9.0-2.0*Fl/3.0)*L20*K20 + (4.0/3.0)*(2.0/3.0)*AFB*L10*K00 + (-4.0/3.0)*(2.0/3.0)*AFB*L10*K20 ) * (2*pi);

lower_phi = ((-pi),)
upper_phi = ((pi),)
limits_phi = zfit.Space (axes =(2), limits =(lower_phi, upper_phi))
PDF_gen_Si.register_analytic_integral (func = integralSi_phi, limits = limits_phi)


#integral over costl
def integralSi_costl(x, limits, norm_range, params, model):
    costk, phi = x.unstack_x()
    Fl = params['Fl']
    S3 = params['S3']
    S9 = params['S9']
    S5 = params['S5']
    S7 = params['S7']    
    K00 = assoc_legendre_tf(0,0,costk)
    K20 = assoc_legendre_tf(2,0,costk)
    K21 = assoc_legendre_tf(2,1,costk)
    K22 = assoc_legendre_tf(2,2,costk)
    return 9./(32 * 3.14159265) * ( (4.0/9.0)*K00 + (4.0*Fl/3.0-4.0/9.0)*K20 + 1/18.0*2*S3*K22*cos(2*phi)*2 + 1/9.0*S9*K22*sin(2*phi)*2 + (2.0/3.0)*S5*K21*cos(phi)*pi/4 + (-2.0/3.0)*S7*K21*sin(phi)*pi/4 ) * (2)

lower_costl = ((-1),)
upper_costl = ((1),)
limits_costl = zfit.Space (axes =(1), limits =(lower_costl, upper_costl))
PDF_gen_Si.register_analytic_integral (func = integralSi_costl, limits = limits_costl)

#integral over costk
def integralSi_costk(x, limits, norm_range, params, model):
    costl, phi = x.unstack_x()
    Fl = params['Fl']
    S3 = params['S3']
    S9 = params['S9']
    AFB = params['AFB']
    L00 = assoc_legendre_tf(0,0,costl)
    L10 = assoc_legendre_tf(1,0,costl)
    L20 = assoc_legendre_tf(2,0,costl)
    L22 = assoc_legendre_tf(2,2,costl)
    return 9./(32 * 3.14159265) * ( (4.0/9.0)*L00 + (2.0/9.0-2.0*Fl/3.0)*L20 + (4.0/3.0)*(3.0/2.0)*AFB*L10 + 1/18.0*2*S3*L22*cos(2*phi)*2 + 1/9.0*S9*L22*sin(2*phi)*2) * (2)

lower_costk = ((-1),)
upper_costk = ((1),)
limits_costk = zfit.Space (axes =(0), limits =(lower_costk, upper_costk))
PDF_gen_Si.register_analytic_integral (func = integralSi_costk, limits = limits_costk) 











