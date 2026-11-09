#!/bin/bash
# GCE startup script: runs as root on every boot.
md=http://metadata.google.internal/computeMetadata/v1
curl -sfH Metadata-Flavor:Google \
  $md/instance/attributes/serve-sh > /opt/serve.sh
chmod +x /opt/serve.sh
# flashinfer JIT-compiles its kernels; the image has no g++
command -v g++ >/dev/null || apt-get install -y -qq \
  -o DPkg::Lock::Timeout=600 g++
# retries until the first-boot driver install has finished
systemd-run --unit=vllm --setenv=HOME=/root \
  -p Restart=on-failure -p RestartSec=30 /opt/serve.sh
# power off after 60 min with GPU < 5% and no ssh session
systemd-run --unit=idle-off bash -c '
idle=0
while sleep 60; do
  u=$(nvidia-smi --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "${u:-0}" -lt 5 ] && ! who | grep -q pts; then
    idle=$((idle+1)); else idle=0; fi
  [ $idle -ge 60 ] && poweroff
done'
