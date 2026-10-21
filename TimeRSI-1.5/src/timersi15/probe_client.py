"""Same decision/probe JSON interface, minimal startup and a window-local service."""
import argparse
import os
import json
from pathlib import Path
import socket
import time
import uuid


def request(path, payload):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(path))
        with connection.makefile('rwb') as stream:
            stream.write(json.dumps(payload, allow_nan=False).encode()+b'\n')
            stream.flush()
            response = json.loads(stream.readline())
    if 'error' in response:
        raise RuntimeError(response['error'])
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decision', type=Path)
    parser.add_argument('--weights')
    parser.add_argument('--group')
    parser.add_argument('--input', type=Path, default=Path('INPUT.json'))
    parser.add_argument('--socket', type=Path, default=Path('.composition.sock'))
    args = parser.parse_args()
    packet = json.loads(args.input.read_text())
    if args.decision:
        decision = json.loads(args.decision.read_text())
    elif args.weights:
        decision = dict(default=dict(weights={packet['global_winner']: 1.}), groups={})
        spec = dict(weights=json.loads(args.weights))
        if args.group:
            decision['groups'][args.group] = spec
        else:
            decision['default'] = spec
    else:
        parser.error('Provide --decision FILE or --weights JSON')
    directory_fd = os.open(args.input.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        socket_path = Path(f'/proc/self/fd/{directory_fd}') / args.socket.name
        result = request(socket_path, {'decision': decision})['result']
    finally:
        os.close(directory_fd)
    result['at'] = time.time()
    directory = args.input.parent/'composition-probes'
    directory.mkdir(exist_ok=True)
    path = directory/(uuid.uuid4().hex+'.json')
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({**result, 'receipt': str(path)}))


if __name__ == '__main__':
    main()
