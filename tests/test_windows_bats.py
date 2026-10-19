"""Exercise real cmd.exe control flow using offline executable stubs.

No packages or models are installed. The stub reproduces signed Windows exit
codes, including Python Install Manager's missing-runtime failure.
"""
from pathlib import Path
import ast
import os
import re
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
INSTALL = '\u5b89\u88c5\u73af\u5883.bat'
DOWNLOAD = '\u4e0b\u8f7d\u6a21\u578b.bat'
STUB_SOURCE = r'''
using System;
using System.IO;
using System.Reflection;
using System.Text;
class Stub {
    static int Main(string[] args) {
        string exe = Assembly.GetExecutingAssembly().Location;
        string name = Path.GetFileName(exe);
        string line = string.Join(" | ", args);
        File.AppendAllText(Environment.GetEnvironmentVariable("TEST_BAT_LOG"), name + " : " + line + "\n", Encoding.UTF8);
        string missing = Environment.GetEnvironmentVariable("TEST_MISSING_VERSIONS") ?? "";
        if (name == "py.exe" && args.Length > 0 && missing.Contains(args[0])) return unchecked((int)0xA0000006);
        if (Environment.GetEnvironmentVariable("TEST_NO_PYTHON") == "1" && (name == "py.exe" || name == "python.exe") && !exe.Contains(".venv")) return unchecked((int)0xA0000006);
        if (name == "nvidia-smi.exe") return Environment.GetEnvironmentVariable("TEST_NO_GPU") == "1" ? 1 : 0;
        string fail = Environment.GetEnvironmentVariable("TEST_FAIL_CONTAINS");
        if (!String.IsNullOrEmpty(fail) && line.Contains(fail)) return unchecked((int)0xC0000142);
        string venv = Path.Combine(Environment.CurrentDirectory, ".venv");
        if (line.Contains("Invalid project venv") && File.Exists(Path.Combine(venv, "invalid.marker"))) return 19;
        if (line.Contains("src.rename(dst)")) Directory.Move(venv, venv + ".backup_test");
        if (line.Contains("-m | venv")) {
            string scripts = Path.Combine(venv, "Scripts");
            Directory.CreateDirectory(scripts);
            File.Copy(exe, Path.Combine(scripts, "python.exe"), true);
        }
        if (line.Contains("Existing PyTorch works") && Environment.GetEnvironmentVariable("TEST_REUSE_TORCH") != "1") return unchecked((int)0xC0000142);
        return 0;
    }
}
'''


class BatchFormatTests(unittest.TestCase):
    def test_ascii_crlf_and_inline_python(self):
        for name in (INSTALL, DOWNLOAD, 'start.bat'):
            with self.subTest(script=name):
                raw = (REPO / name).read_bytes()
                text = raw.decode('ascii')
                self.assertEqual(raw.count(b'\n'), raw.count(b'\r\n'))
                for code in re.findall(r'-c "([^"]*)"', text):
                    code = code.replace('%USE_CUDA%', '1').replace('%MODEL_NAMES%', 'inpaint')
                    ast.parse(code)


@unittest.skipUnless(os.name == 'nt', 'Requires Windows cmd.exe')
class WindowsBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = Path(os.environ['SystemRoot']) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
        if not compiler.is_file():
            raise unittest.SkipTest('Offline stub compiler is unavailable')
        cls.build = tempfile.TemporaryDirectory(prefix='imageedit_bat_test_')
        cls.addClassCleanup(cls.build.cleanup)
        source = Path(cls.build.name) / 'stub.cs'
        source.write_text(STUB_SOURCE, encoding='utf-8')
        cls.stub = Path(cls.build.name) / 'stub.exe'
        subprocess.run([str(compiler), '/nologo', '/target:exe', '/out:' + str(cls.stub), str(source)], check=True, capture_output=True)

    def run_script(self, script=INSTALL, args=(), existing=False, invalid=False, explicit=False, **settings):
        case = tempfile.TemporaryDirectory(prefix='bat_\u4e2d\u6587 & ! ')
        self.addCleanup(case.cleanup)
        root = Path(case.name) / 'project with spaces'
        root.mkdir()
        for name in (INSTALL, DOWNLOAD, 'requirements.txt', 'bootstrap.py', 'download_models.py'):
            shutil.copy2(REPO / name, root / name)
        (root / 'tools').mkdir()
        for name in ('download_advanced_models.py', 'download_lama.py', 'install_torch.py'):
            shutil.copy2(REPO / 'tools' / name, root / 'tools' / name)
        tools = Path(case.name) / 'mock tools'
        tools.mkdir()
        for name in ('py.exe', 'python.exe', 'nvidia-smi.exe'):
            shutil.copy2(self.stub, tools / name)
        if existing or invalid:
            (root / '.venv/Scripts').mkdir(parents=True)
            shutil.copy2(self.stub, root / '.venv/Scripts/python.exe')
        if invalid:
            (root / '.venv/invalid.marker').write_text('preserve me')
        log = Path(case.name) / 'commands.log'
        env = {k: v for k, v in os.environ.items() if not k.startswith('TEST_')}
        for key in ('TORCH_INDEX_URL', 'HF_ENDPOINT', 'IMAGEEDIT_PYTHON'):
            env.pop(key, None)
        env.update(PATH=str(tools) + os.pathsep + str(Path(os.environ['SystemRoot']) / 'System32'), IMAGEEDIT_NO_PAUSE='1', TEST_BAT_LOG=str(log), PYTHONIOENCODING='utf-8')
        env.update(settings)
        if explicit:
            env['IMAGEEDIT_PYTHON'] = str(tools / 'python.exe')
        command = 'cmd.exe /d /s /c ""' + str(root / script) + '"' + ''.join(' ' + a for a in args) + '"'
        result = subprocess.run(command, cwd=case.name, env=env, capture_output=True, timeout=30)
        output = (result.stdout + result.stderr).decode('utf-8', errors='replace')
        calls = log.read_text(encoding='utf-8-sig') if log.exists() else ''
        self.assertNotIn('is not recognized', output)
        return result.returncode, output, calls, root

    def test_negative_missing_versions_fall_back_to_310(self):
        code, output, calls, _ = self.run_script(TEST_MISSING_VERSIONS='-3.12;-3.11')
        self.assertEqual(code, 0, output)
        self.assertIn('Using installed Python 3.10.', output)
        self.assertIn('-3.10 | -m | venv', calls)
        self.assertNotIn('-3.12 | -m | venv', calls)
        self.assertIn('tools\\install_torch.py | auto', calls)

    def test_path_python_fallback(self):
        code, output, calls, _ = self.run_script(TEST_MISSING_VERSIONS='-3.12;-3.11;-3.10;-3.13')
        self.assertEqual(code, 0, output)
        self.assertIn('python.exe : -m | venv', calls)

    def test_explicit_python_overrides_launcher(self):
        code, output, calls, _ = self.run_script(explicit=True, TEST_MISSING_VERSIONS='-3.12;-3.11;-3.10;-3.13')
        self.assertEqual(code, 0, output)
        self.assertIn('python.exe : -m | venv', calls)
        self.assertNotIn('py.exe :', calls)

    def test_broken_environment_is_preserved_before_recreation(self):
        code, output, calls, root = self.run_script(invalid=True, TEST_MISSING_VERSIONS='-3.12;-3.11')
        self.assertEqual(code, 0, output)
        self.assertTrue((root / '.venv.backup_test/invalid.marker').exists())
        self.assertFalse((root / '.venv/invalid.marker').exists())
        self.assertLess(calls.index('src.rename(dst)'), calls.index('-m | venv'))

    def test_no_python_preserves_broken_environment(self):
        code, output, calls, root = self.run_script(invalid=True, TEST_NO_PYTHON='1')
        self.assertEqual(code, 1, output)
        self.assertTrue((root / '.venv/invalid.marker').exists())
        self.assertNotIn('src.rename(dst)', calls)
        self.assertNotIn('-m | pip', calls)

    def test_reuse_working_environment_and_torch(self):
        code, output, calls, _ = self.run_script(existing=True, TEST_REUSE_TORCH='1')
        self.assertEqual(code, 0, output)
        self.assertNotIn('-m | venv', calls)
        self.assertNotIn('--force-reinstall', calls)

    def test_cpu_skips_gpu_probe(self):
        code, output, calls, _ = self.run_script(args=('cpu',))
        self.assertEqual(code, 0, output)
        self.assertIn('tools\\install_torch.py | cpu', calls)
        self.assertNotIn('nvidia-smi.exe', calls)

    def test_negative_errors_stop_installation(self):
        for marker in ('-m | venv', 'tools\\install_torch.py', '-r | requirements.txt', '-m | pip | check', 'import app'):
            with self.subTest(failure=marker):
                code, output, calls, _ = self.run_script(TEST_FAIL_CONTAINS=marker)
                self.assertEqual(code, 1, output)
                self.assertNotIn('SUCCESS.', output)
                if marker != 'import app':
                    self.assertNotIn('import app', calls)

    def test_download_all_and_minimal(self):
        for args, expected in (((), 'clipseg | inpaint | ip2p | clip'), (('minimal',), 'inpaint\n')):
            with self.subTest(args=args):
                code, output, calls, _ = self.run_script(DOWNLOAD, args, existing=True)
                self.assertEqual(code, 0, output)
                self.assertIn('download_models.py | ' + expected, calls)

    def test_negative_errors_stop_download(self):
        for marker in ('import huggingface_hub', 'download_models.py', 'tools\\download_advanced_models.py', 'tools\\download_lama.py'):
            with self.subTest(failure=marker):
                code, output, calls, _ = self.run_script(DOWNLOAD, existing=True, TEST_FAIL_CONTAINS=marker)
                self.assertEqual(code, 1, output)
                self.assertNotIn('SUCCESS.', output)
                self.assertNotIn('Model directory checks passed', calls)

    def test_download_requires_environment(self):
        code, output, calls, _ = self.run_script(DOWNLOAD)
        self.assertEqual(code, 1, output)
        self.assertEqual(calls, '')


if __name__ == '__main__':
    unittest.main()
