__version__= "$Id$"

import glob
import pyfits


class SnapshotTweaker():

    def __init__(self, pathname, logger):
        self._pathname= pathname
        self._logger= logger
        self._matchingFiles= glob.glob(self._pathname)


    def headerUpdateKey(self, key, value):
        for filename in self._matchingFiles:
            self._logger.notice('update %s: %s= %s' % (filename, key, str(value)))
            pyfits.setval(filename, key, value)


    def headerDeleteKey(self, key):
        for filename in self._matchingFiles:
            self._logger.notice('delete %s: %s' % (filename, key))
            pyfits.delval(filename, key)


    def headerRenameKey2(self, key, newkey):
        for filename in self._matchingFiles:
            self._logger.notice('rename %s: %s= %s' % (filename, key, newkey))
            try:
                value= pyfits.getval(filename, key)
                pyfits.setval(filename, newkey, value)
                pyfits.delval(filename, key)
            except:
                pass


    def headerRenameKey(self, oldKey, newKey):
        for filename in self._matchingFiles:
            self._logger.notice('rename %s: %s= %s' %
                                (filename, oldKey, newKey))
            try:
                hdulist= pyfits.open(filename, 'update')
                hdr= hdulist[0].header
                for k, v in hdr.items():
                    newK= k.replace(oldKey, newKey)
                    if newK in hdr:
                        self._logger.warn('%s already exists' % newK)
                        continue
                    if newK != k:
                        del hdr[k]
                        hdr.update('HIERARCH '+newK, v)
                hdulist.close(output_verify='fix')
            except Exception, e:
                self._logger.warn('headerRename failed (%s)') % str(e)
                pass


    def headerGetKey(self, key):
        ret={}
        for filename in self._matchingFiles:
            ret[filename]= pyfits.getval(filename, key)
        return ret
