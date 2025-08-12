==> Cloning from https://github.com/ShivamCh09/Capstone-News-Classifier-Project
==> Checking out commit 4035f95182652f108aaafffb4d7ce3de7fd7d1a3 in branch main
==> Using Python version 3.13.4 (default)
==> Docs on specifying a Python version: https://render.com/docs/python-version
==> Using Poetry version 2.1.3 (default)
==> Docs on specifying a Poetry version: https://render.com/docs/poetry-version
==> Running build command 'pip install -r requirements.txt'...
Collecting flask (from -r requirements.txt (line 1))
  Downloading flask-3.1.1-py3-none-any.whl.metadata (3.0 kB)
Collecting scikit-learn (from -r requirements.txt (line 2))
  Downloading scikit_learn-1.7.1-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
Collecting pandas (from -r requirements.txt (line 3))
  Downloading pandas-2.3.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (91 kB)
Collecting joblib (from -r requirements.txt (line 4))
  Downloading joblib-1.5.1-py3-none-any.whl.metadata (5.6 kB)
Collecting gunicorn (from -r requirements.txt (line 5))
  Downloading gunicorn-23.0.0-py3-none-any.whl.metadata (4.4 kB)
Collecting blinker>=1.9.0 (from flask->-r requirements.txt (line 1))
  Downloading blinker-1.9.0-py3-none-any.whl.metadata (1.6 kB)
Collecting click>=8.1.3 (from flask->-r requirements.txt (line 1))
  Downloading click-8.2.1-py3-none-any.whl.metadata (2.5 kB)
Collecting itsdangerous>=2.2.0 (from flask->-r requirements.txt (line 1))
  Downloading itsdangerous-2.2.0-py3-none-any.whl.metadata (1.9 kB)
Collecting jinja2>=3.1.2 (from flask->-r requirements.txt (line 1))
  Downloading jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting markupsafe>=2.1.1 (from flask->-r requirements.txt (line 1))
  Downloading MarkupSafe-3.0.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.0 kB)
Collecting werkzeug>=3.1.0 (from flask->-r requirements.txt (line 1))
  Downloading werkzeug-3.1.3-py3-none-any.whl.metadata (3.7 kB)
Collecting numpy>=1.22.0 (from scikit-learn->-r requirements.txt (line 2))
  Downloading numpy-2.3.2-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (62 kB)
Collecting scipy>=1.8.0 (from scikit-learn->-r requirements.txt (line 2))
  Downloading scipy-1.16.1-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (61 kB)
Collecting threadpoolctl>=3.1.0 (from scikit-learn->-r requirements.txt (line 2))
  Downloading threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Collecting python-dateutil>=2.8.2 (from pandas->-r requirements.txt (line 3))
  Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting pytz>=2020.1 (from pandas->-r requirements.txt (line 3))
  Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
Collecting tzdata>=2022.7 (from pandas->-r requirements.txt (line 3))
  Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting packaging (from gunicorn->-r requirements.txt (line 5))
  Downloading packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
Collecting six>=1.5 (from python-dateutil>=2.8.2->pandas->-r requirements.txt (line 3))
  Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Downloading flask-3.1.1-py3-none-any.whl (103 kB)
Downloading scikit_learn-1.7.1-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (9.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.4/9.4 MB 20.0 MB/s eta 0:00:00
Downloading pandas-2.3.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.1/12.1 MB 152.9 MB/s eta 0:00:00
Downloading joblib-1.5.1-py3-none-any.whl (307 kB)
Downloading gunicorn-23.0.0-py3-none-any.whl (85 kB)
Downloading blinker-1.9.0-py3-none-any.whl (8.5 kB)
Downloading click-8.2.1-py3-none-any.whl (102 kB)
Downloading itsdangerous-2.2.0-py3-none-any.whl (16 kB)
Downloading jinja2-3.1.6-py3-none-any.whl (134 kB)
Downloading MarkupSafe-3.0.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (23 kB)
Downloading numpy-2.3.2-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (16.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.6/16.6 MB 90.4 MB/s eta 0:00:00
Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Downloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
Downloading scipy-1.16.1-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (35.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.2/35.2 MB 143.2 MB/s eta 0:00:00
Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
Downloading threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Downloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
Downloading werkzeug-3.1.3-py3-none-any.whl (224 kB)
Downloading packaging-25.0-py3-none-any.whl (66 kB)
Installing collected packages: pytz, tzdata, threadpoolctl, six, packaging, numpy, markupsafe, joblib, itsdangerous, click, blinker, werkzeug, scipy, python-dateutil, jinja2, gunicorn, scikit-learn, pandas, flask
Successfully installed blinker-1.9.0 click-8.2.1 flask-3.1.1 gunicorn-23.0.0 itsdangerous-2.2.0 jinja2-3.1.6 joblib-1.5.1 markupsafe-3.0.2 numpy-2.3.2 packaging-25.0 pandas-2.3.1 python-dateutil-2.9.0.post0 pytz-2025.2 scikit-learn-1.7.1 scipy-1.16.1 six-1.17.0 threadpoolctl-3.6.0 tzdata-2025.2 werkzeug-3.1.3
[notice] A new release of pip is available: 25.1.1 -> 25.2
[notice] To update, run: pip install --upgrade pip
==> Uploading build...
==> Uploaded in 7.0s. Compression took 2.8s
==> Build successful 🎉
==> Deploying...
==> Running 'gunicorn api:app'
Traceback (most recent call last):
  File "/opt/render/project/src/.venv/bin/gunicorn", line 8, in <module>
    sys.exit(run())
             ~~~^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/app/wsgiapp.py", line 66, in run
    WSGIApplication("%(prog)s [OPTIONS] [APP_MODULE]", prog=prog).run()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/app/base.py", line 235, in run
    super().run()
    ~~~~~~~~~~~^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/app/base.py", line 71, in run
    Arbiter(self).run()
    ~~~~~~~^^^^^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/arbiter.py", line 57, in __init__
    self.setup(app)
    ~~~~~~~~~~^^^^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/arbiter.py", line 117, in setup
    self.app.wsgi()
    ~~~~~~~~~~~~~^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/app/base.py", line 66, in wsgi
    self.callable = self.load()
                    ~~~~~~~~~^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/app/wsgiapp.py", line 57, in load
    return self.load_wsgiapp()
           ~~~~~~~~~~~~~~~~~^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/app/wsgiapp.py", line 47, in load_wsgiapp
    return util.import_app(self.app_uri)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/util.py", line 370, in import_app
    mod = importlib.import_module(module)
  File "/usr/local/lib/python3.13/importlib/__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 1026, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/opt/render/project/src/api.py", line 13, in <module>
    import gradio as gr
ModuleNotFoundError: No module named 'gradio'
==> Exited with status 1
==> Common ways to troubleshoot your deploy: https://render.com/docs/troubleshooting-deploys
==> Running 'gunicorn api:app'
Traceback (most recent call last):
  File "/opt/render/project/src/.venv/bin/gunicorn", line 8, in <module>
    sys.exit(run())
             ~~~^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/app/wsgiapp.py", line 66, in run
    WSGIApplication("%(prog)s [OPTIONS] [APP_MODULE]", prog=prog).run()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/app/base.py", line 235, in run
    super().run()
    ~~~~~~~~~~~^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/app/base.py", line 71, in run
    Arbiter(self).run()
    ~~~~~~~^^^^^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/arbiter.py", line 57, in __init__
    self.setup(app)
    ~~~~~~~~~~^^^^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/arbiter.py", line 117, in setup
    self.app.wsgi()
    ~~~~~~~~~~~~~^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/app/base.py", line 66, in wsgi
    self.callable = self.load()
                    ~~~~~~~~~^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/app/wsgiapp.py", line 57, in load
    return self.load_wsgiapp()
           ~~~~~~~~~~~~~~~~~^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/app/wsgiapp.py", line 47, in load_wsgiapp
    return util.import_app(self.app_uri)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/opt/render/project/src/.venv/lib/python3.13/site-packages/gunicorn/util.py", line 370, in import_app
    mod = importlib.import_module(module)
  File "/usr/local/lib/python3.13/importlib/__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 1026, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/opt/render/project/src/api.py", line 13, in <module>
    import gradio as gr
ModuleNotFoundError: No module named 'gradio'