import time
import unittest

from krknctl_lightspeed.command_parser import load_meta_commands, build_commands


class CommandParser(unittest.TestCase):
    def test_load_meta_commands(self):
        commands = load_meta_commands("meta_commands_.json")
        pass

    def test_build_commands(self):
        build_commands("meta_commands_.json", "krknctl-input")
