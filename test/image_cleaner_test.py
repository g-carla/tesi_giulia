'''
Created on 16 ott 2018

@author: gcarla
'''
import unittest
from tesi.image_cleaner import ImageCleaner


class ImageCleanerTest(unittest.TestCase):



    def testDummyFunctionReturnsTheRightAnswer(self):
        anImageCleaner= ImageCleaner()
        answer= anImageCleaner.aDummyFunction()
        self.assertEqual(
            42, answer,
            "Wrong Answer!") 



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()