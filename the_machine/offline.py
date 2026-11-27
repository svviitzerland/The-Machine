"""Keeps the Machine off the internet while it runs.

Only this computer and the local network (where its head lives) can be reached.
Anything else is refused, and the first attempt from each place in the code is
reported with where it came from, so a library that phones home shows itself.
"""
from __future__ import annotations

import ipaddress
import socket
import sys
import traceback

_seen: set[str] = set()


def _local(address) -> bool:
    host = address[0] if isinstance(address, tuple) else address
    if not isinstance(host, str):
        return True                                 # unix sockets and the like
    try:
        ip = ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        return host == "localhost" or host.endswith(".local")
    if ip.version == 6:
        # Some networks hand out "private" fd00:: addresses that lead straight to the internet,
        # so for IPv6 only this machine and the link itself count. The head uses IPv4.
        return ip.is_loopback or ip.is_link_local
    return ip.is_loopback or ip.is_private or ip.is_link_local


def guard() -> None:
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def check(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not _local(address):
            where = "".join(traceback.format_stack(limit=8)[:-2])
            if where not in _seen:
                _seen.add(where)
                print(f"[offline] blocked a connection to {address[0]}:{address[1]} from:\n{where}", file=sys.stderr)
            raise OSError(f"The Machine runs offline; {address[0]} is not on this computer or network")

    def guarded_connect(sock, address):
        check(sock, address)
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        check(sock, address)
        return connect_ex(sock, address)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
