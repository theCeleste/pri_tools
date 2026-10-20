import unittest

from privacy_attack.prefix_attack import build_parser, validate_args


class PrefixAttackArgumentTests(unittest.TestCase):
    def parse(self, *target_args):
        parser = build_parser()
        args = parser.parse_args([*target_args, "--output-dir", "/tmp/prefix-output"])
        validate_args(args, parser)
        return args

    def test_adapter_and_steering_can_be_combined(self):
        args = self.parse("--adapter", "/tmp/adapter", "--steering", "/tmp/steering")
        self.assertEqual(args.adapter, "/tmp/adapter")
        self.assertEqual(args.steering, "/tmp/steering")

    def test_adapter_only_remains_supported(self):
        args = self.parse("--adapter", "/tmp/adapter")
        self.assertIsNone(args.steering)

    def test_target_is_required(self):
        parser = build_parser()
        args = parser.parse_args(["--output-dir", "/tmp/prefix-output"])
        with self.assertRaises(SystemExit):
            validate_args(args, parser)


if __name__ == "__main__":
    unittest.main()
