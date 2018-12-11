
import importlib
# importlib.reload(test)
import numpy as np
import photutils
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils import make_source_mask
#from photutils import find_peaks
#from astropy.visualization import simple_norm

def removeBackground(ima):
    mask= make_source_mask(ima, snr=2, npixels=5, dilate_size=7)
    mean, median, std= sigma_clipped_stats(ima, sigma=3.0, mask=mask)
    
    return ima-median         
         
 
def findStarsRelativePosition():
    
    ima_A = fits.getdata('/home/gcarla/tera1/201512/December_reduced_iskren/NGC2419_H.fits')
    ima_B = fits.getdata('/home/gcarla/tera1/201610/reduction/NGC2419_201610_19_23_24_26_H_reg.fits')
    
    ima_A_cut1 = ima_A[1460:1490,890:920]
    ima_A_cut2 = ima_A[1295:1325,1315:1345]
    ima_A_cut3 = ima_A[1415:1445,1960:1990]
    ima_A_cut4 = ima_A[1960:1990,1735:1765]
    ima_A_cut5 = ima_A[1750:1780,1765:1795]
   
    ima_B_cut1 = ima_B[785:815,495:525]
    ima_B_cut2 = ima_B[535:565,875:905]
    ima_B_cut3 = ima_B[525:555,1530:1560]
    ima_B_cut4 = ima_B[1105:1135,1415:1445]
    ima_B_cut5 = ima_B[900:930,1405:1435]