"""Hardware selection and fallback checks without packages, GPUs or network."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools import install_torch as setup


class TorchSelectionTests(unittest.TestCase):
    @patch.object(setup, 'ensure_torch')
    def test_helper_refuses_to_modify_a_global_python(self, ensure):
        with patch.object(setup.sys, 'argv', ['install_torch.py']), \
                patch.object(setup.sys, 'prefix', 'global-python'), \
                patch.object(setup.sys, 'base_prefix', 'global-python'):
            self.assertEqual(setup.main(), 1)
        ensure.assert_not_called()

    def test_hardware_matrix(self):
        cases = [
            ([], 'cpu', '2.11.0'),
            ([('N/A', 'N/A')], 'cpu', '2.11.0'),
            ([('470.0', '6.1')], 'cpu', '2.11.0'),
            ([('616.92', '3.5')], 'cpu', '2.11.0'),
            ([('616.92', '6.1')], 'cu118', '2.7.1'),
            ([('550.54', '8.6')], 'cu118', '2.7.1'),
            ([('566.36', '8.6')], 'cu126', '2.11.0'),
            ([('616.92', '8.6')], 'cu130', '2.11.0'),
            ([('566.36', '12.0')], 'cpu', '2.11.0'),
            ([('576.80', '12.0')], 'cu128', '2.11.0'),
            ([('616.92', '12.0')], 'cu130', '2.11.0'),
            ([('616.92', '6.1'), ('616.92', '12.0')], 'cpu', '2.11.0'),
        ]
        for rows, channel, version in cases:
            with self.subTest(gpus=rows):
                result = setup.select_build(rows)
                self.assertEqual(result.channel, channel)
                self.assertEqual(result.torch, version)

    @patch.object(setup.subprocess, 'run')
    def test_negative_gpu_query_exit_means_unavailable(self, run):
        run.return_value = SimpleNamespace(returncode=-1073741502, stdout='616.92, 12.0\n')
        self.assertEqual(setup.detect_gpus(), [])

    @patch.object(setup.subprocess, 'run')
    def test_gpu_query_parses_all_cards(self, run):
        run.return_value = SimpleNamespace(returncode=0, stdout='616.92, 8.6\n616.92, 12.0\n')
        self.assertEqual(setup.detect_gpus(), [('616.92', '8.6'), ('616.92', '12.0')])

    @patch.object(setup.subprocess, 'run')
    def test_install_pins_matching_packages_and_rejects_negative_exit(self, run):
        run.return_value = SimpleNamespace(returncode=-1073741502)
        self.assertFalse(setup.install(setup.Build('cu118', '2.7.1', '0.22.1')))
        command = run.call_args.args[0]
        self.assertIn('torch==2.7.1', command)
        self.assertIn('torchvision==0.22.1', command)
        self.assertIn('https://download.pytorch.org/whl/cu118', command)

    @patch.object(setup.subprocess, 'run')
    def test_probe_checks_convolution_and_vision_operators(self, run):
        run.return_value = SimpleNamespace(returncode=0, stdout='verified', stderr='')
        self.assertTrue(setup.probe('cuda'))
        command = run.call_args.args[0]
        self.assertIn('conv2d', command[-1])
        self.assertIn('torchvision.ops.nms', command[-1])

    @patch.object(setup, 'detect_gpus', return_value=[('616.92', '12.0')])
    @patch.object(setup, 'probe', return_value=True)
    @patch.object(setup, 'install')
    def test_valid_existing_cuda_is_reused(self, install, probe, _):
        self.assertEqual(setup.ensure_torch(), 'cuda')
        install.assert_not_called()

    @patch.object(setup, 'detect_gpus', return_value=[('616.92', '12.0')])
    @patch.object(setup, 'probe', side_effect=[False, False, True])
    @patch.object(setup, 'install', return_value=True)
    def test_cuda_runtime_failure_falls_back_to_verified_cpu(self, install, probe, _):
        self.assertEqual(setup.ensure_torch(), 'cpu')
        self.assertEqual([call.args[0].channel for call in install.call_args_list], ['cu130', 'cpu'])
        self.assertEqual(probe.call_args.kwargs, {'require_cpu_build': True})

    @patch.object(setup, 'detect_gpus', return_value=[('616.92', '12.0')])
    @patch.object(setup, 'probe', side_effect=[False, True])
    @patch.object(setup, 'install', side_effect=[False, True])
    def test_cuda_install_failure_falls_back_to_cpu(self, install, probe, _):
        self.assertEqual(setup.ensure_torch(), 'cpu')

    @patch.object(setup, 'detect_gpus', return_value=[('616.92', '12.0')])
    @patch.object(setup, 'probe', return_value=False)
    @patch.object(setup, 'install', return_value=False)
    def test_explicit_index_failure_is_not_overridden(self, install, probe, _):
        with self.assertRaises(RuntimeError):
            setup.ensure_torch(index_override='https://download.pytorch.org/whl/cu128')
        self.assertEqual(install.call_count, 1)

    @patch.object(setup, 'detect_gpus')
    @patch.object(setup, 'probe', side_effect=[False, True])
    @patch.object(setup, 'install', return_value=True)
    def test_cpu_mode_does_not_require_a_gpu(self, install, probe, detect):
        self.assertEqual(setup.ensure_torch('cpu'), 'cpu')
        detect.assert_not_called()
        self.assertEqual(install.call_args.args[0].channel, 'cpu')
        self.assertEqual(probe.call_args.kwargs, {'require_cpu_build': True})

    @patch.object(setup, 'probe', return_value=False)
    @patch.object(setup, 'install', return_value=False)
    def test_failed_cpu_install_cannot_report_success(self, install, probe):
        with self.assertRaises(RuntimeError):
            setup.ensure_torch('cpu')

    @patch.object(setup, 'detect_gpus', return_value=[])
    @patch.object(setup, 'probe', return_value=True)
    @patch.object(setup, 'install')
    def test_automatic_cpu_requires_cpu_build_for_app_device_selection(self, install, probe, _):
        self.assertEqual(setup.ensure_torch(), 'cpu')
        probe.assert_called_once_with('cpu', require_cpu_build=True)
        install.assert_not_called()


if __name__ == '__main__':
    unittest.main()
