import requests
import os
import unittest
from unittest.mock import patch, MagicMock

import logging

logging = logging.getLogger(__name__).setLevel(logging.DEBUG)

MODULE = "agents.deep_q_learning"


class CertifyTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super(CertifyTest, cls).setUpClass()
        env = MagicMock()


    def test_apk_not_found(self):
        response: dict = certify_md5(apps.UNEXISTING.md5)
        self.assertEqual(response["status"], settings.SCAN_STATUS_FAILED)
        self.assertEqual(response["error"], "Apk file not Found")

    def test_apk_is_split(self):
        response: dict = certify_md5(apps.SPLIT_APK.md5)
        self.assertEqual(response["status"], settings.SCAN_STATUS_SKIPPED)
        self.assertEqual(response["reason"], "App Split")