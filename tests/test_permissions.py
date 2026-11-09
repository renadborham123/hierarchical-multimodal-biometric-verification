import unittest

from permissions import get_user_role, is_command_allowed


class PermissionTests(unittest.TestCase):
    def test_admin_unlock(self):
        self.assertTrue(is_command_allowed("user_01", "UNLOCK"))

    def test_employee_unlock_denied(self):
        self.assertFalse(is_command_allowed("user_02", "UNLOCK"))

    def test_guest_report_denied(self):
        self.assertFalse(is_command_allowed("user_04", "REPORT"))

    def test_employee_status(self):
        self.assertTrue(is_command_allowed("user_02", "STATUS"))

    def test_unknown_user_role_and_command(self):
        self.assertIsNone(get_user_role("unknown"))
        self.assertFalse(is_command_allowed("unknown", "STATUS"))
        self.assertFalse(is_command_allowed("user_01", "DELETE"))
