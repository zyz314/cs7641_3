# Installation

Install Poetry and add it to `PATH`:

```
curl -sSL https://install.python-poetry.org | python -
echo 'export PATH="/home/$USER/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Test installation of poetry
poetry --version
```

Install project dependencies and build the project:
```
poetry install
poetry build
```

# Troubleshooting

## Certifi SSL certificate expiration
The `ucimlrepo` package requires the `certifi` package to be updated in the host OS, not the virtual env installed by Poetry. On certain versions of Python, one may run into an issue when fetching the dataset (see the example error message below).

<details open>
    <summary> Example of SSL certificate expiration issue </summary>

    Traceback (most recent call last):
    File "/usr/lib/python3.10/urllib/request.py", line 1348, in do_open
        h.request(req.get_method(), req.selector, req.data, headers,
    File "/usr/lib/python3.10/http/client.py", line 1283, in request
        self._send_request(method, url, body, headers, encode_chunked)
    File "/usr/lib/python3.10/http/client.py", line 1329, in _send_request
        self.endheaders(body, encode_chunked=encode_chunked)
    File "/usr/lib/python3.10/http/client.py", line 1278, in endheaders
        self._send_output(message_body, encode_chunked=encode_chunked)
    File "/usr/lib/python3.10/http/client.py", line 1038, in _send_output
        self.send(msg)
    File "/usr/lib/python3.10/http/client.py", line 976, in send
        self.connect()
    File "/usr/lib/python3.10/http/client.py", line 1448, in connect
        super().connect()
    File "/usr/lib/python3.10/http/client.py", line 942, in connect
        self.sock = self._create_connection(
    File "/usr/lib/python3.10/socket.py", line 845, in create_connection
        raise err
    File "/usr/lib/python3.10/socket.py", line 833, in create_connection
        sock.connect(sa)
    TimeoutError: [Errno 110] Connection timed out

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
    File "/mnt/c/Users/quang/Desktop/DeepLearning/cs7641/assignment1/.venv/lib/python3.10/site-packages/ucimlrepo/fetch.py", line 68, in fetch_ucirepo
        response = urllib.request.urlopen(api_url, context=ssl.create_default_context(cafile=certifi.where()))
    File "/usr/lib/python3.10/urllib/request.py", line 216, in urlopen
        return opener.open(url, data, timeout)
    File "/usr/lib/python3.10/urllib/request.py", line 519, in open
        response = self._open(req, data)
    File "/usr/lib/python3.10/urllib/request.py", line 536, in _open
        result = self._call_chain(self.handle_open, protocol, protocol +
    File "/usr/lib/python3.10/urllib/request.py", line 496, in _call_chain
        result = func(*args)
    File "/usr/lib/python3.10/urllib/request.py", line 1391, in https_open
        return self.do_open(http.client.HTTPSConnection, req,
    File "/usr/lib/python3.10/urllib/request.py", line 1351, in do_open
        raise URLError(err)
    urllib.error.URLError: <urlopen error [Errno 110] Connection timed out>

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
    File "/mnt/c/Users/quang/Desktop/DeepLearning/cs7641/assignment1/main.py", line 4, in <module>
        adult = fetch_ucirepo(id=2) 
    File "/mnt/c/Users/quang/Desktop/DeepLearning/cs7641/assignment1/.venv/lib/python3.10/site-packages/ucimlrepo/fetch.py", line 71, in fetch_ucirepo
        raise ConnectionError('Error connecting to server')
    ConnectionError: Error connecting to server
</details>

If such issue occurs, try the following command in the host OS:
```
pip install certifi --upgrade
```