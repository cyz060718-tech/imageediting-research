"""Start the real web app and check localhost HTTP without downloading models."""
from pathlib import Path
import os
import sys
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Keep the smoke check local even on machines with an HTTP proxy configured.
for key in ('NO_PROXY', 'no_proxy'):
    existing = os.environ.get(key, '')
    os.environ[key] = ','.join(filter(None, [existing, 'localhost', '127.0.0.1', '::1']))
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')


def main():
    import app
    demo = app.build_demo().queue(default_concurrency_limit=1)
    try:
        demo.launch(server_name='127.0.0.1', inbrowser=False, share=False,
                    prevent_thread_lock=True, quiet=True)
        with build_opener(ProxyHandler({})).open(demo.local_url, timeout=30) as response:
            assert response.status == 200, response.status
            assert len(response.read()) > 100
        print('Web app startup and localhost HTTP check passed:', demo.local_url)
    finally:
        demo.close()


if __name__ == '__main__':
    main()
