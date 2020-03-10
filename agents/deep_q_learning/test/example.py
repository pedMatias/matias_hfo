import requests
import os
import unittest
from unittest.mock import patch

import app_signature_checker.settings as settings
from app_signature_checker.managers.app import App
from app_signature_checker.certify.tasks import certify_md5
from app_signature_checker.utils import download_apk
from app_signature_checker.certify.test import md5s_to_test as apps

import logging

logging = logging.getLogger(__name__).setLevel(logging.DEBUG)

MODULE = "app_signature_checker"

URL_POST_CERTIFY = 'http://localhost:5000/certify/md5/job'


def set_up() -> None:
    download_apk(apps.UNIVERSAL_APK.md5)
    download_apk(apps.BASE_APK.md5)
    download_apk(apps.SPLIT_APK.md5,
                 link="http://pool.apk.aptoide.com/apps/me-saket-dank-debug"
                      "-10-47005221-87f95183509a4b49c332a7511d320d45-config"
                      "-x86-64.apk")
    download_apk(apps.ALREADY_CERTIFIED.md5)
    download_apk(apps.GREY_SIGNATURE.md5)
    download_apk(apps.NOT_IN_GOOGLE.md5)
    download_apk(apps.APK_WITH_MISSING_DATA.md5)
    download_apk(apps.MATURE.md5)
    download_apk(apps.PAID.md5)


class CertifyTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super(CertifyTest, cls).setUpClass()

    def test_apk_not_found(self):
        response: dict = certify_md5(apps.UNEXISTING.md5)
        self.assertEqual(response["status"], settings.SCAN_STATUS_FAILED)
        self.assertEqual(response["error"], "Apk file not Found")

    def test_apk_is_split(self):
        response: dict = certify_md5(apps.SPLIT_APK.md5)
        self.assertEqual(response["status"], settings.SCAN_STATUS_SKIPPED)
        self.assertEqual(response["reason"], "App Split")

    @unittest.skip("Do not make sense anymore")
    def test_md5_not_found_in_aptoide(self):
        md5 = apps.NOT_IN_APTOIDE.md5

        def set_up():
            path = ''
            for path_part in [settings.APK_ROOT_MOUNT_POINT, md5[0], md5[1]]:
                path = os.path.join(path, path_part)
                if not os.path.isdir(path):
                    os.mkdir(path)
            file = open(os.path.join(path, md5 + '.apk'), "w+")
            file.close()

        def tear_down():
            apk = os.path.join(settings.APK_ROOT_MOUNT_POINT, md5[0],
                               md5[1], md5 + '.apk')
            os.remove(apk)

        set_up()
        response: dict = certify_md5(md5)
        self.assertEqual(response["status"], settings.SCAN_STATUS_FAILED)
        tear_down()

    @patch.object(App, "dev_signature_certified")
    @patch.object(App, "dev_signature_unknown")
    @patch(MODULE + ".managers.scrapers_handler.app_in_google_and_not_paid")
    def test_apk_with_missing_information(self, in_google_and_not_paid, unknown,
                                          certified):
        certified.return_value = False
        unknown.return_value = True
        in_google_and_not_paid.return_value = False, None
        # test:
        response: dict = certify_md5(apps.APK_WITH_MISSING_DATA.md5)
        self.assertEqual(response["status"], settings.SCAN_STATUS_COMPLETED)
        self.assertEqual(response["certified"], False)
        self.assertEqual(response["reason"], "App not in Google Play or Paid")

    def test_signature_not_unknown(self):
        response: dict = certify_md5(apps.GREY_SIGNATURE.md5)
        self.assertEqual(response["status"], settings.SCAN_STATUS_SKIPPED)
        self.assertIn("Signature status is", response["reason"])

    @patch.object(App, "dev_signature_unknown")
    @patch(MODULE + ".managers.change_md5_status.approve_md5")
    @patch(MODULE + ".managers.change_md5_status.trust_md5")
    def test_signature_already_certified(self, mock_unknown, mock_trust,
                                         mock_approve):
        mock_unknown.return_value = True
        mock_trust.return_value = True
        mock_approve.return_value = True
        response: dict = certify_md5(apps.ALREADY_CERTIFIED.md5)
        self.assertEqual(response["status"], settings.SCAN_STATUS_COMPLETED)
        self.assertEqual(response["rank_changed_to_trusted"], True)
        self.assertEqual(response["certified"], False)
        self.assertEqual(response["reason"], "Signature already validated;")

    @patch.object(App, "dev_signature_unknown")
    @patch.object(App, "dev_signature_certified")
    def test_mature_md5(self, mock_certified, mock_unknown):
        mock_certified.return_value = True
        mock_unknown.return_value = True
        # test:
        response: dict = certify_md5(apps.MATURE.md5)
        self.assertEqual(response["status"], settings.SCAN_STATUS_COMPLETED)
        self.assertEqual(response["certified"], False)
        self.assertIn("App is Mature", response["reason"])

    @patch.object(App, "dev_signature_certified")
    @patch.object(App, "dev_signature_unknown")
    def test_paid_md5(self, mock_unknown, mock_certified):
        mock_certified.return_value = False
        mock_unknown.return_value = True
        # test:
        response: dict = certify_md5(apps.PAID.md5)
        self.assertEqual(response["status"], settings.SCAN_STATUS_COMPLETED)
        self.assertEqual(response["certified"], False)
        self.assertEqual(response["rank_changed_to_trusted"], False)
        self.assertIn("App not in Google Play or Paid", response["reason"])

    @patch.object(App, "dev_signature_certified")
    @patch.object(App, "dev_signature_unknown")
    def test_app_not_in_google(self, mock_unknown, mock_certified):
        mock_certified.return_value = False
        mock_unknown.return_value = True
        response: dict = certify_md5(apps.NOT_IN_GOOGLE.md5)
        self.assertEqual(response["status"], settings.SCAN_STATUS_COMPLETED)
        self.assertEqual(response["certified"], False)
        self.assertEqual(response["reason"], "App not in Google Play or Paid")

    @patch.object(App, "dev_signature_certified")
    @patch.object(App, "dev_signature_unknown")
    @patch(MODULE + ".managers.scrapers_handler.app_in_google_and_not_paid")
    def test_md5_with_few_downloads(self, mock_in_google, mock_unknown,
                                    mock_certified):
        mock_certified.return_value = False
        mock_unknown.return_value = True
        mock_in_google.return_value = True, 10
        # test:
        response: dict = certify_md5(apps.UNIVERSAL_APK.md5)
        self.assertEqual(response["status"], settings.SCAN_STATUS_COMPLETED)
        self.assertEqual(response["certified"], False)
        self.assertIn("Number of downloads bellow Google Play Cap", response[
            "reason"])

    @patch.object(App, "dev_signature_certified")
    @patch.object(App, "dev_signature_unknown")
    @patch(MODULE + ".managers.scrapers_handler.app_in_google_and_not_paid")
    @patch(MODULE + ".managers.direct_handler.get_google_play_dev_signature")
    @patch(MODULE + ".managers.change_md5_status.approve_md5")
    @patch(MODULE + ".managers.change_md5_status.trust_md5")
    @patch.object(App, "certify_dev_signature")
    def test_base_apk(self, mock_certify, mock_trust, mock_approve,
                      mock_google_signature, mock_in_google, mock_unknown,
                      mock_certified):
        mock_certified.return_value = False
        mock_unknown.return_value = True
        mock_in_google.return_value = True, 10000
        mock_google_signature.return_value = apps.BASE_APK.dev_signature
        mock_trust.return_value = True
        mock_approve.return_value = True
        mock_certify.return_value = True
        # test:
        response: dict = certify_md5(apps.BASE_APK.md5)
        self.assertEqual(response["status"], settings.SCAN_STATUS_COMPLETED)
        self.assertEqual(response["certified"], True)

    @patch.object(App, "dev_signature_certified")
    @patch.object(App, "dev_signature_unknown")
    @patch(MODULE + ".managers.scrapers_handler.app_in_google_and_not_paid")
    @patch(MODULE + ".managers.direct_handler.get_google_play_dev_signature")
    @patch(MODULE + ".managers.change_md5_status.approve_md5")
    @patch(MODULE + ".managers.change_md5_status.trust_md5")
    @patch.object(App, "certify_dev_signature")
    def test_universal_apk(self, mock_certify, mock_trust, mock_approve,
                           mock_google_signature, mock_in_google, mock_unknown,
                           mock_certified):
        mock_certified.return_value = False
        mock_unknown.return_value = True
        mock_in_google.return_value = True, 10000
        mock_google_signature.return_value = apps.UNIVERSAL_APK.dev_signature
        mock_trust.return_value = True
        mock_approve.return_value = True
        mock_certify.return_value = True
        # test:
        response: dict = certify_md5(apps.UNIVERSAL_APK.md5)
        self.assertEqual(response["status"], settings.SCAN_STATUS_COMPLETED)
        self.assertEqual(response["certified"], True)

    @patch.object(App, "dev_signature_certified")
    @patch.object(App, "dev_signature_unknown")
    @patch(MODULE + ".managers.change_md5_status.approve_md5")
    @patch(MODULE + ".managers.change_md5_status.trust_md5")
    @patch.object(App, "certify_dev_signature")
    def test_google(self, mock_certify, mock_trust, mock_approve,
                    mock_unknown, mock_certified):
        mock_certified.return_value = False
        mock_unknown.return_value = True
        mock_trust.return_value = True
        mock_approve.return_value = True
        mock_certify.return_value = True
        # test:
        response: dict = certify_md5(apps.ALREADY_CERTIFIED.md5)
        self.assertEqual(response["status"], settings.SCAN_STATUS_COMPLETED)
        self.assertEqual(response["certified"], True)


if __name__ == '__main__':
    unittest.main()
