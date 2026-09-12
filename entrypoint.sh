#!/bin/bash
# 容器入口：以 root 启动 sshd（监听 2301），随后降权为 dev 用户常驻
set -e

# host 网络下容器 hostname 不在 /etc/hosts，补上以免 sudo 报解析告警
grep -q "$(hostname)" /etc/hosts || echo "127.0.1.1 $(hostname)" >> /etc/hosts

mkdir -p /run/sshd
/usr/sbin/sshd

exec gosu dev "$@"
