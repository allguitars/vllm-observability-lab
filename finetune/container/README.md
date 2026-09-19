# aiDAPTIVLink2 fine-tuning container

`docker-compose.yml` is the Compose equivalent of the aiDAPTIVLink2 fine-tuning
training-slide command. It does not build or pull the Phison image, and it does not
start a container by itself.

It follows the MIG device-reservation pattern in the supplied aiDAPTIVLink 3.0
inference Compose example, while retaining the aiDAPTIVLink 2.0 fine-tuning
slide's image, mounts, IPC, privilege, and ulimit settings.

## Before use on the H200 host

1. Confirm the local image name with `docker image ls`.
2. Copy `compose.env.example` to `compose.env` and replace the three host paths if
   they differ from the training-slide example.
3. Run `nvidia-smi -L` and set `MIG_DEVICE_ID` to the intended `MIG-...` UUID.
   Do not copy the UUID shown in the reference screenshot.
4. Ensure `/dev/mapper`, `/var/lock`, the model directory, and the NVMe mount are
   present on that host.
5. If Toolkit dependencies must be installed inside the container, create a venv or
   install them outside `/workspace/finetune/toolkit`, because that mount is read-only.

## Start and enter the container

Run from this directory:

```bash
cp compose.env.example compose.env
docker compose --env-file compose.env up -d
docker compose --env-file compose.env exec phison-finetune bash
```

To stop and remove only this container:

```bash
docker compose --env-file compose.env down
```

The service is deliberately `privileged` and mounts host device-management paths,
matching the vendor slide. Run it only on the intended H200 host and review each
host path in `compose.env` before `up`. After startup, run `nvidia-smi -L` inside
the container and retain the output with the Toolkit evidence. It confirms which
device is exposed to the container, but does not by itself validate aiDAPTIVLink
2.0 fine-tuning on MIG.
