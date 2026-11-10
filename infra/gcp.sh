#!/usr/bin/env bash
# One L4 VM serving vLLM, in your current gcloud project.
#   infra/gcp.sh up spot|ondemand  create it (~10 min first boot)
#   infra/gcp.sh tunnel            vLLM API on localhost:8000
#   infra/gcp.sh ssh [--command CMD] | stop | start | down
# No external IP (Columbia's org policy forbids them): SSH via
# IAP, internet via Cloud NAT. Powers off after 60 idle min.
set -euo pipefail
cd "$(dirname "$0")"
VM=aog-l4
ZONES="us-central1-a us-central1-b us-central1-c
       us-east1-b us-east1-c us-east1-d"

zone() {
  gcloud compute instances list --filter="name=$VM" \
    --format="value(zone.basename())"
}

nat() {
  gcloud compute routers describe aog-$1 --region=$1 \
    >/dev/null 2>&1 && return
  gcloud compute routers create aog-$1 --region=$1 \
    --network=default
  gcloud compute routers nats create aog-$1 --region=$1 \
    --router=aog-$1 --auto-allocate-nat-external-ips \
    --nat-all-subnet-ip-ranges
}

up() {
  local sched
  case $1 in
    spot) sched="--provisioning-model=SPOT
                 --instance-termination-action=STOP" ;;
    ondemand) sched=--provisioning-model=STANDARD ;;
    *) echo "usage: up spot|ondemand"; exit 1 ;;
  esac
  gcloud services enable compute.googleapis.com
  gcloud compute firewall-rules describe aog-iap \
    >/dev/null 2>&1 ||
    gcloud compute firewall-rules create aog-iap \
      --network=default --allow=tcp:22,tcp:8000 \
      --source-ranges=35.235.240.0/20
  for z in $ZONES; do
    nat "${z%-*}"
    # shellcheck disable=SC2086
    gcloud compute instances create $VM --zone="$z" \
      --machine-type=g2-standard-4 --no-address \
      --accelerator=type=nvidia-l4,count=1 $sched \
      --maintenance-policy=TERMINATE \
      --image-project=deeplearning-platform-release \
      --image-family=common-cu129-ubuntu-2204-nvidia-580 \
      --boot-disk-size=100GB \
      --metadata=install-nvidia-driver=True \
      --metadata-from-file=startup-script=vm-startup.sh,serve-sh=serve.sh \
      && return
  done
  echo "no L4 capacity in: $ZONES"; exit 1
}

down() {
  local z; z=$(zone)
  [ -z "$z" ] || gcloud compute instances delete $VM \
    --zone="$z" --quiet
  gcloud compute routers list --filter="name~^aog-" \
    --format="value(name,region.basename())" |
    while read -r name region; do
      gcloud compute routers delete "$name" \
        --region="$region" --quiet
    done
}

case ${1:-} in
  up) up "${2:-}" ;;
  ssh) shift; gcloud compute ssh $VM --zone="$(zone)" \
         --tunnel-through-iap "$@" ;;
  tunnel) gcloud compute start-iap-tunnel $VM 8000 \
            --local-host-port=localhost:8000 --zone="$(zone)" ;;
  stop|start) gcloud compute instances "$1" $VM \
                --zone="$(zone)" ;;
  down) down ;;
  *) sed -n 2,7p gcp.sh; exit 1 ;;
esac
