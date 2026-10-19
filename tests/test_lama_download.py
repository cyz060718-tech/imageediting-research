"""Downloader failure and offline-transfer checks with no real model or network."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import httpx
from tools import download_lama as download

DATA = b'verified checkpoint fixture'
MD5 = hashlib.md5(DATA).hexdigest()
URL = 'https://example.test/big-lama.pt'


class LaMaDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / 'models/lama/big-lama.pt'

    def client(self, handler):
        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
        self.addCleanup(client.close)
        return client

    def test_verified_local_copy_can_be_imported_without_network(self):
        source = self.root / 'big-lama.pt'
        source.write_bytes(DATA)
        download.prepare(self.target, URL, MD5, source=source)
        self.assertEqual(self.target.read_bytes(), DATA)
        self.assertEqual(source.read_bytes(), DATA)
        self.assertEqual(json.loads(self.target.with_name('provenance.json').read_text())['md5'], MD5)

    def test_bad_offline_source_is_not_published(self):
        source = self.root / 'big-lama.pt'
        source.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            download.prepare(self.target, URL, MD5, source=source)
        self.assertFalse(self.target.exists())

    def test_http_download_follows_redirect_and_checks_digest(self):
        def handler(request):
            if request.url.path == '/big-lama.pt':
                return httpx.Response(302, headers={'Location': '/asset'})
            return httpx.Response(200, content=DATA)
        download.prepare(self.target, URL, MD5, client=self.client(handler))
        self.assertEqual(self.target.read_bytes(), DATA)
        self.assertFalse(self.target.with_suffix('.download').exists())

    def test_network_failure_retries_and_succeeds(self):
        calls = []
        def handler(request):
            calls.append(request.url)
            if len(calls) == 1:
                raise httpx.ConnectError('temporary TLS connection failure', request=request)
            return httpx.Response(200, content=DATA)
        self.target.parent.mkdir(parents=True)
        with patch.object(download.time, 'sleep'):
            download.download_checked(URL, self.target.with_suffix('.download'), MD5,
                                      self.client(handler), sleep=lambda _: None)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.target.with_suffix('.download').read_bytes(), DATA)

    def test_failed_checksum_never_overwrites_target(self):
        client = self.client(lambda request: httpx.Response(200, content=b'not the checkpoint'))
        self.target.parent.mkdir(parents=True)
        with self.assertRaises(RuntimeError):
            download.download_checked(URL, self.target.with_suffix('.download'), MD5,
                                      client, attempts=2, sleep=lambda _: None)
        self.assertFalse(self.target.exists())

    def test_valid_existing_model_is_reused(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(DATA)
        client = self.client(lambda request: self.fail('Valid local models must not be downloaded again'))
        download.prepare(self.target, URL, MD5, client=client)
        self.assertEqual(self.target.read_bytes(), DATA)

    def test_bad_existing_model_is_preserved(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b'bad existing file')
        with self.assertRaises(ValueError):
            download.prepare(self.target, URL, MD5)
        self.assertEqual(self.target.read_bytes(), b'bad existing file')


if __name__ == '__main__':
    unittest.main()
