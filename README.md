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

## Frame capture behavior

The web process uses one shared MLX90640 capture thread for all browser connections. Viewers receive the same encoded MJPEG frame instead of starting independent sensor read loops. I2C initialization and frame reads remain protected by the camera instance's lock; the production Gunicorn command intentionally keeps one worker and one camera instance.

The sensor starts at 2 Hz while frame reliability is being diagnosed. Each capture cycle makes at most three `getFrame` attempts, waits 100 ms between transient failures, and accepts `RuntimeError`, `ValueError`, and `OSError` as recoverable frame failures. After an exhausted cycle, the capture loop waits 500 ms before trying a new bounded cycle. Persistent failures therefore remain visible in logs without terminating the MJPEG response or creating a busy retry loop.

The sensor's native 32 x 24 temperature field is interpolated to the 640 x 480 browser frame before the temperature label and hot-point ring are drawn. Annotations therefore remain readable and do not obscure the thermal image.

The browser polls `/camera_status` every second. If the MLX90640 reports an initialization or frame-read error, the page displays a red `CAMERA OFFLINE` banner and full-frame overlay with the latest diagnostic. The side panel shows the measured capture rate in frames per second and the number of connected viewers. `/healthz` remains an application liveness check and can stay healthy while the physical sensor is unavailable.

Gunicorn runs one worker with a fixed thread pool and every MJPEG response holds its thread until the viewer disconnects. `MAX_STREAM_VIEWERS` caps concurrent `/video_feed` subscribers so the page, the status poll and the health check always keep free threads; additional viewers receive `503` with an explanatory message instead of hanging the site.

The confirmed repository defects were:

- `RuntimeError: Too many retries` escaped the camera adapter because it caught only `ValueError`, terminating that viewer's MJPEG generator.
- Every viewer previously ran its own read loop. The existing lock prevented simultaneous calls, but multiple viewers still multiplied sequential sensor reads.
- The initial 8 Hz refresh setting gave the bus and sensor less recovery margin.
- The temperature label and hot-point marker were drawn onto the native 32 x 24 frame and then enlarged 20 times, producing the giant white `Ho` shapes and star seen in the stream.
- Unbounded `/video_feed` subscribers consumed every Gunicorn thread, so the page, the status poll and the health check stalled while viewers were connected.

The underlying reason the MLX90640 library reports `Too many retries` is not established without hardware evidence. An ACK at address `0x33` confirms that a device responds, not that full frame transfers succeed. Remaining possibilities include I2C signal integrity, power stability, clock behavior, bus frequency compatibility, or the sensor itself.

If failures continue, run these read-only diagnostics manually on the Raspberry Pi:

```bash
docker logs --tail 100 ir-camera-app
docker inspect --format '{{json .HostConfig.Devices}}' ir-camera-app
docker exec ir-camera-app ls -l /dev/i2c-1
curl --noproxy "*" --fail http://localhost:3000/healthz
curl --noproxy "*" --fail http://localhost/IR-Camera/healthz
```

These commands only inspect logs, container configuration, the mapped device node, and HTTP health. They do not change I2C settings, permissions, containers, or host configuration.


## CI pipeline

`.github/workflows/build.yml` runs for pushes to `main`, version tags matching `v*`, and manual dispatches. It:

1. Installs the test dependencies and runs `pytest`.
2. Enables QEMU and Docker Buildx.
3. Builds the Dockerfile for `linux/arm64`.
4. Verifies the image reports Linux ARM64 and serves `/healthz`.
5. Saves and gzip-compresses the complete image as `ir-camera-arm64.tar.gz`.
6. Verifies Docker can load the compressed archive directly.
7. Creates a commit-specific GitHub Release for every successful `main` build and attaches both the image archive and matching `compose.yaml`.
8. Attaches `v*` tag builds and their matching `compose.yaml` to version releases.

The runner must have Docker access and outbound access to container registries, Debian package repositories, PyPI, and the official GitHub Actions used by the workflow. No proxy credentials or deployment credentials are stored in the repository. CI never connects to the Raspberry Pi.

## Download and transfer from Windows PowerShell

Every successful `main` build becomes a GitHub Release. Download both the ARM64 image and the matching Compose file from the same latest release:

```powershell
$ReleaseBase = "https://github.com/samuelfirmansyahatbshg/IR-Camera/releases/latest/download"
$ImagePath = "$HOME\Downloads\ir-camera-arm64.tar.gz"
$ComposePath = "$HOME\Downloads\compose.yaml"
Invoke-WebRequest -Uri "$ReleaseBase/ir-camera-arm64.tar.gz" -OutFile $ImagePath
Invoke-WebRequest -Uri "$ReleaseBase/compose.yaml" -OutFile $ComposePath
Get-Item $ImagePath, $ComposePath
```

Release assets are downloaded directly. There is no Actions artifact ZIP to extract.

Transfer both files to the Raspberry Pi. Updating `compose.yaml` is required because older deployments referenced `ir-camera-app:latest` instead of the CI image tag `ir-camera:arm64`.

```powershell
scp -i "$HOME\.ssh\pi_key" "$HOME\Downloads\ir-camera-arm64.tar.gz" "$HOME\Downloads\compose.yaml" firmansyah@nbnlle0004.bsh.corp.bshg.com:~/IR-Camera/
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

Load the CI image, validate the transferred Compose file, recreate the camera container, and wait for direct application health:

```powershell
ssh -i "$HOME\.ssh\pi_key" firmansyah@nbnlle0004.bsh.corp.bshg.com 'set -e; docker load -i ~/IR-Camera/ir-camera-arm64.tar.gz; cd ~/IR-Camera; docker compose config --quiet; docker compose up -d --no-build --pull never --force-recreate; for attempt in $(seq 1 30); do if curl --noproxy "*" --fail --silent --show-error http://localhost:3000/healthz; then break; fi; if [ "$attempt" -eq 30 ]; then docker logs --tail 100 ir-camera-app; exit 1; fi; sleep 2; done; docker compose ps'
```

The transferred `compose.yaml` references `ir-camera:arm64`, matching the tag stored in the release archive. No additional image tag is required. `--noproxy "*"` prevents a host proxy from redirecting the local health request to a proxy such as `localhost:3128`.

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
curl --noproxy "*" --fail http://localhost:3000/healthz
curl --noproxy "*" --fail http://localhost/IR-Camera/healthz
```

The camera opens when the first browser requests `/IR-Camera/video_feed`. Container logs report I2C or sensor initialization failures.

## Roll back

Keep previous successful GitHub Release images under `~/IR-Camera/rollback/`, with the release tag or version in each filename. Loading an older release image restores its included `ir-camera:arm64` tag:

```bash
docker load -i ~/IR-Camera/rollback/ir-camera-arm64-<release-tag>.tar.gz
cd ~/IR-Camera
docker compose up -d --no-build --pull never --force-recreate
curl --noproxy "*" --fail http://localhost:3000/healthz
curl --noproxy "*" --fail http://localhost/IR-Camera/healthz
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
