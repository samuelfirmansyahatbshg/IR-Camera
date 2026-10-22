# IR Camera

MLX90640 Raspberry Pi thermal camera web streamer. The application publishes its OpenCV inferno stream on host port 3000. A separately managed Nginx container exposes it below `/IR-Camera/`.

## Hardware

| MLX90640 | Raspberry Pi |
|---|---|
| VCC | 3.3 V |
| GND | GND |
| SDA | GPIO2 / SDA1 |
| SCL | GPIO3 / SCL1 |

The default sensor address is `0x33`. Enable I2C on the Raspberry Pi and confirm that bus 1 exists:

```sh
ls -l /dev/i2c-1
```

## Runtime

- `ir-camera-app`: prebuilt `ir-camera:arm64` image on host port 3000
- `ir-camera-nginx`: separately managed Nginx container on host port 80
- MLX90640 access: `/dev/i2c-1` passed to `ir-camera-app`
- Public URL: `http://<raspberry-pi-address>/IR-Camera/`

Camera files and downloaded image archives stay in `~/IR-Camera/`. The Nginx configuration remains outside this repository at `~/nginx/default.conf`.

The Raspberry Pi never builds the image and never installs Python, Debian, or Nginx packages. `compose.yaml` sets `pull_policy: never` and uses the fixed image tag `ir-camera:arm64`.

## CI pipeline

`.github/workflows/build.yml` runs for pushes to `main`, version tags matching `v*`, and manual dispatches. It:

1. Installs the test dependencies and runs `pytest`.
2. Enables QEMU and Docker Buildx.
3. Builds the Dockerfile for `linux/arm64`.
4. Verifies the image reports Linux ARM64 and serves `/healthz`.
5. Saves and gzip-compresses the complete image as `ir-camera-arm64.tar.gz`.
6. Verifies Docker can load the compressed archive directly.
7. Creates a commit-specific GitHub Release for every successful `main` build and attaches the archive.
8. Attaches `v*` tag builds to their matching version releases.

The runner must have Docker access and outbound access to container registries, Debian package repositories, PyPI, and the official GitHub Actions used by the workflow. No proxy credentials or deployment credentials are stored in the repository. CI never connects to the Raspberry Pi.

## Download and transfer from Windows PowerShell

Every successful `main` build becomes a GitHub Release. Download the newest ARM64 image directly from the latest release:

```powershell
$ReleaseUrl = "https://github.com/samuelfirmansyahatbshg/IR-Camera/releases/latest/download/ir-camera-arm64.tar.gz"
$ImagePath = "$HOME\Downloads\ir-camera-arm64.tar.gz"
Invoke-WebRequest -Uri $ReleaseUrl -OutFile $ImagePath
Get-Item $ImagePath
```

Release assets are downloaded directly as `.tar.gz` files. There is no Actions artifact ZIP to extract.

Transfer the Docker image to the Raspberry Pi:

```powershell
scp -i "$HOME\.ssh\pi_key" "$HOME\Downloads\ir-camera-arm64.tar.gz" firmansyah@nbnlle0004.bsh.corp.bshg.com:~/IR-Camera/
```

The separately managed Nginx configuration can still be transferred independently from the sibling local directory:

```powershell
ssh user@server 'mkdir -p ~/nginx'
scp .\nginx\default.conf user@server:~/nginx/default.conf
ssh user@server 'docker restart ir-camera-nginx'
```

Restarting the existing Nginx container reads the updated bind-mounted configuration. These commands do not install Nginx or pull an Nginx image.

## Deploy on the Raspberry Pi

Before replacing a working image, save a local rollback archive:

```bash
mkdir -p ~/IR-Camera/rollback
docker image save ir-camera:arm64 | gzip > ~/IR-Camera/rollback/ir-camera-arm64-before-update.tar.gz
```

Load the CI image and force-recreate the camera container from Windows PowerShell:

```powershell
ssh -i "$HOME\.ssh\pi_key" firmansyah@nbnlle0004.bsh.corp.bshg.com 'set -e; docker load -i ~/IR-Camera/ir-camera-arm64.tar.gz && cd ~/IR-Camera && docker compose up -d --no-build --pull never --force-recreate && docker compose ps'
```

`compose.yaml` already references `ir-camera:arm64`, so an additional `ir-camera-app:latest` tag is not required.

Inspect the camera container after deployment:

```powershell
ssh -i "$HOME\.ssh\pi_key" firmansyah@nbnlle0004.bsh.corp.bshg.com 'docker logs --tail 50 ir-camera-app'
```

The CI workflow verifies direct loading of the gzip-compressed Docker archive. If an older Docker Engine rejects compressed input, stream decompression without installing anything:

```bash
gzip -dc ~/IR-Camera/ir-camera-arm64.tar.gz | docker load
```

Verify both the application and external Nginx route:

```bash
curl --fail http://localhost:3000/healthz
curl --fail http://localhost/IR-Camera/healthz
```

The camera opens when the first browser requests `/IR-Camera/video_feed`. Container logs report I2C or sensor initialization failures.

## Roll back

Keep previous successful GitHub Release images under `~/IR-Camera/rollback/`, with the release tag or version in each filename. Loading an older release image restores its included `ir-camera:arm64` tag:

```bash
docker load -i ~/IR-Camera/rollback/ir-camera-arm64-<release-tag>.tar.gz
cd ~/IR-Camera
docker compose up -d --no-build --pull never --force-recreate
curl --fail http://localhost:3000/healthz
curl --fail http://localhost/IR-Camera/healthz
```

To restore the archive saved immediately before an update:

```bash
docker load -i ~/IR-Camera/rollback/ir-camera-arm64-before-update.tar.gz
cd ~/IR-Camera
docker compose up -d --no-build --pull never --force-recreate
```

## Operations

```bash
cd ~/IR-Camera
docker compose ps
docker compose logs -f app
docker compose restart
docker compose down
```

## Local tests

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
