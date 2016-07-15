#!/usr/bin/env python2
# encoding: utf-8

import unittest

from test_basics import TestBasics
from test_rtr_models import TestRTRModels


suite_basics = unittest.TestLoader().loadTestsFromTestCase(TestBasics)
suite_rtr = unittest.TestLoader().loadTestsFromTestCase(TestRTRModels)

all_tests = unittest.TestSuite([suite_basics, suite_rtr])


