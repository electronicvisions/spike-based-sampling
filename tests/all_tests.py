#!/usr/bin/env python2
# encoding: utf-8

import unittest

from test_basics import TestBasics
from test_rtr_models import TestRTRModels
from test_noice import TestNN
from test_nest import TestNest


all_tests = unittest.TestSuite([
    unittest.TestLoader().loadTestsFromTestCase(TestBasics),
    unittest.TestLoader().loadTestsFromTestCase(TestNest),
    unittest.TestLoader().loadTestsFromTestCase(TestRTRModels),
    unittest.TestLoader().loadTestsFromTestCase(TestNN),
])

