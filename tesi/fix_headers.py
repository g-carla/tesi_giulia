'''
Created on 1 dic 2018

@author: giulia
'''

import os
import glob
from astropy.io import fits


class FixHeaders():

    def __init__(self, pathname):
        self._pathname= pathname
        self._matchingFiles= glob.glob(os.path.join(self._pathname,'*.fits'))


    def headerUpdateKey(self, key, value):
        for filename in self._matchingFiles:
            print('update %s: %s= %s' % (filename, key, str(value)))
            fits.setval(filename, key, value)


    def headerDeleteKey(self, key):
        for filename in self._matchingFiles:
            print('delete %s: %s' % (filename, key))
            fits.delval(filename, key)


    def headerRenameKey2(self, key, newkey):
        for filename in self._matchingFiles:
            print('rename %s: %s= %s' % (filename, key, newkey))
            try:
                value= fits.getval(filename, key)
                fits.setval(filename, newkey, value)
                fits.delval(filename, key)
            except Exception as e:
                print(str(e))
                pass


    def headerRenameKey(self, oldKey, newKey):
        for filename in self._matchingFiles:
            try:
                hdulist= fits.open(filename, 'update')
                hdr= hdulist[0].header
                
                v= hdr[oldKey]
                del hdr[oldKey]
                hdr['HIERARCH '+newKey]=v
                print('Modyfing %s: replaced %s with %s' % (
                            filename, oldKey, newKey))
                hdulist.close(output_verify='fix')
            except KeyError:
                pass
            except Exception as e:
                print(filename)
                print('headerRename failed (%s)' % str(e))
                pass
            


    def headerGetKey(self, key):
        ret={}
        for filename in self._matchingFiles:
            ret[filename]= fits.getval(filename, key)
        return ret
