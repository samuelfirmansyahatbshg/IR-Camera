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

- `ir-camera-app`: prebuilt `ir-camera:arm64` image, internal port 3000 only
- `ir-camera-nginx`: separately managed Nginx container, the only service publishing a host port (80)
- MLX90640 access: `/dev/i2c-1` passed to `ir-camera-app`
- Public URL: `http://<raspberry-pi-address>/IR-Camera/`

Camera files and downloaded image archives stay in `~/IR-Camera/`. The Nginx configuration remains outside this repository at `~/nginx/default.conf`. The containers share the external `web-proxy` Docker network; camera Compose service `app` has the unique network alias `ir-camera`, so Nginx uses `ir-camera:3000` instead of a host port. Nginx remains a standalone container, not part of the camera Compose project.

The Raspberry Pi never builds the image and never installs Python, Debian, or Nginx packages. `compose.yaml` sets `pull_policy: never` and uses the fixed image tag `ir-camera:arm64`.

## Frame capture behavior

The web process uses one shared MLX90640 capture thread for all browser connections. Viewers receive the same encoded MJPEG frame instead of starting independent sensor read loops. I2C initialization and frame reads remain protected by the camera instance's lock; the production Gunicorn command intentionally keeps one worker and one camera instance.

The sensor uses a 4 Hz subpage refresh setting, yielding about 2 complete frames per second because each frame needs two subpages. On the Pi, a real sensor and JPEG-encoding benchmark measured 1.97 full frames/s with no retries at 4 Hz; settings of 8, 16, 32 and 64 Hz produced no successful captures and repeated `Too many retries` errors. Each capture cycle makes at most three `getFrame` attempts, waits 100 ms between transient failures, and accepts `RuntimeError`, `ValueError`, and `OSError` as recoverable frame failures. After an exhausted cycle, the capture loop waits 500 ms before trying a new bounded cycle. Persistent failures therefore remain visible in logs without terminating the MJPEG response or creating a busy retry loop. The Linux Blinka backend warns that Python cannot set I2C frequency, so the requested `800_000` value does not prove the host bus speed; no host I2C configuration was changed.

The configured refresh setting is checked against the sensor's control register before each capture under the existing camera lock. If the hardware setting changes, the adapter logs the old and observed register codes and restores 4 Hz before reading the frame. This prevents a cached sensor instance from silently remaining at 2 Hz (about 1 full frame/s). Register read/write failures use the same bounded capture retries and offline reporting as frame-read failures; the app does not change host I2C settings.

The MJPEG stream carries the sensor's native 32 x 24 frame: the server only maps temperatures to the inferno palette and nothing is drawn on top of the data. The browser scales the image to the display size, so no server-side interpolation or annotation can distort what the sensor measured.

The browser polls `/camera_status` every second. If the MLX90640 reports an initialization or frame-read error, the page displays a red `CAMERA OFFLINE` banner and full-frame overlay with the latest diagnostic. The side panel shows the measured capture rate, the frame average temperature (`average_celsius` in the status payload) and the temperature under the cursor; viewer counts stay in the payload for the server-side viewer cap but are no longer displayed. A color scale rendered by `/temperature_scale.png` sits beside the feed, mapping the inferno palette from `MAX_TEMPERATURE_C` (300 C, top) down to `MIN_TEMPERATURE_C` (-40 C, bottom), which is the MLX90640's specified object temperature range. The status payload reports the same bounds as `scale_min_celsius` and `scale_max_celsius`, and the page renders its tick labels from them, so changing the two constants moves the image and the labels together. Covering the full range means everyday scenes sit in a narrow part of the palette and look flat; narrow the constants when a run only spans a small band. `/healthz` remains an application liveness check and can stay healthy while the physical sensor is unavailable.

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
docker exec ir-camera-app python -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:3000/healthz", timeout=2).read().decode())'
curl --noproxy "*" --fail http://localhost/IR-Camera/healthz
```

These commands only inspect logs, container configuration, the mapped device node, and HTTP health. They do not change I2C settings, permissions, containers, or host configuration.


## Measuring temperatures in the page

Hovering the feed shows the temperature of the pixel under the cursor, both as a tooltip on the image and in the `Cursor` tile. The markers below the feed work like a handheld thermal camera:

- `Point` click places `P1`, `P2`, ... on one sensor cell.
- `Rectangle` drag places `R1`, `R2`, ... and covers every cell inside the region.
- `Circle` drag places `C1`, `C2`, ... from the centre outwards and covers every cell whose centre falls inside the radius.
- `Clear` removes all markers.
- `Save current` names the markers now on screen and stores them; `Saved layouts` then offers `Load` to place that layout again and `Delete` to remove it. Saving under an existing name replaces it. Loading replaces the markers on screen. Layouts live in the browser's local storage, so they stay on the machine that saved them.

The `C` / `F` toggle above the metrics panel switches every displayed temperature between Celsius and Fahrenheit. The choice is remembered in the browser's local storage.

The table lists each marker's minimum, maximum and average temperature and refreshes twice a second. Marker geometry is stored in sensor cell coordinates, so the overlay and the statistics stay correct at any window size. The page reads the grid from `GET /latest_frame.json`, which returns `{"width": 32, "height": 24, "temperatures": [[...]]}` with `temperatures` set to `null` before the first frame.

The unit toggle only affects what the page displays. The sensor, `/camera_status`, `/latest_frame.json` and every recording download stay in Celsius so saved data always carries one unit.

## Collecting data from the page

The page's `Start collecting` / `Stop collecting` buttons record the live stream in process memory; `Download` then returns the session to the browser. The format radio chooses the payload:

- `Images (JPEG)` returns `ir-camera-<session>.zip` with `frame-000001.jpg` upward plus a `metadata.json` carrying `session_id`, `started_at`, `stopped_at`, `frame_count` and the per-frame `timeframes`.
- `Numbers (JSON)` returns `ir-camera-<session>.json` with `timeframes` and `frames`, where `frames[i]` is the 24 x 32 Celsius grid captured at `timeframes[i]`.
- `Both (JPEG + JSON)` returns one `ir-camera-<session>.zip` containing the JPEGs, `metadata.json` and `frames.json`. The JSON includes the same temperature grids as Numbers mode, with one timestamp and one grid for every JPEG in the archive.

The recorder polls the shared capture loop through `FrameBroadcaster.latest_snapshot`, so it never opens a second sensor read path and never counts against `MAX_STREAM_VIEWERS`. Recording while the sensor is offline is allowed: the session stays open with zero frames and starts collecting as soon as frames return. `MAX_RECORDING_FRAMES` (2000, about 16 minutes at 2 full frames/s) bounds memory and ends the recording on its own. One recording exists at a time; starting a new one discards the previous frames, and a container restart loses an undownloaded session.

The supporting routes are `POST /recording/start` (`{"format": "images" | "json" | "both"}`), `POST /recording/stop`, `GET /recording/status` and `GET /recording/download`. Requests that do not match the recorder state return `409` with an `error` message. The `Resolution` tile beside the feed shows the native 32 x 24 sensor resolution; exports are not upscaled.

## Collecting data from the command line

`tools/irctl.py` drives the same recording routes as the page. It uses only the standard library, so it runs on any machine with Python 3.12 and on the Raspberry Pi itself. Every request bypasses proxies, which the corporate proxy would otherwise hijack.

```bash
python tools/irctl.py status
python tools/irctl.py start --format both
python tools/irctl.py stop
python tools/irctl.py save -o "$HOME/Downloads"
```

`record` runs the whole cycle and writes the file when it finishes:

```bash
python tools/irctl.py record --duration 60 --format both -o "$HOME/Downloads"
```

`save` and `record` accept a file or a directory for `-o`; a directory keeps the server's `ir-camera-<session>.<ext>` name, and the default is that name in the working directory. The camera URL defaults to `http://nbnlle0004.bsh.corp.bshg.com/IR-Camera/` and can be changed with `--url` or the `IR_CAMERA_URL` environment variable. Requests the recorder refuses, such as stopping when nothing is running, print the server's message and exit with status `1`.

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
ssh user@server 'docker exec ir-camera-nginx nginx -t && docker exec ir-camera-nginx nginx -s reload'
```

Validate and reload the existing Nginx container after copying its bind-mounted configuration. These commands do not install Nginx or pull an image. The configuration requires Nginx 1.27.3 or newer for open-source dynamic upstream DNS resolution; the deployed container was verified with 1.31.8. It uses Docker DNS (`127.0.0.11`) and a shared upstream zone to discover replacement camera containers without a reload. The trailing slash on `proxy_pass`, forwarded headers and unbuffered MJPEG settings are preserved.

## Deploy on the Raspberry Pi

### One-time shared network setup

Create the shared network before starting camera Compose, because it declares `web-proxy` as external:

```bash
docker network inspect web-proxy || docker network create web-proxy
```

For an existing standalone Nginx container, attach it once:

```bash
docker network connect web-proxy ir-camera-nginx
```

The attachment survives Nginx restarts. When creating or replacing Nginx, include the network in its launch command; do not use `host.docker.internal` or `--add-host`:

```bash
docker run -d --pull never \
  --name ir-camera-nginx \
  --restart unless-stopped \
  --network web-proxy \
  --publish 80:80 \
  --mount type=bind,src="$HOME/nginx/default.conf",dst=/etc/nginx/conf.d/default.conf,readonly \
  nginx:alpine
```

Camera Compose joins the same network with alias `ir-camera` and exposes port 3000 internally without a host `ports` mapping. Other Compose projects can join this external network with their own unique aliases and also use internal port 3000; only add proxy routes for applications actually deployed.

### Deploy a camera release

Run on the Pi. Before replacing an image, save the current image and Compose configuration:

```bash
cd ~/IR-Camera
mkdir -p rollback
docker image save ir-camera:arm64 | gzip > rollback/ir-camera-arm64-before-update.tar.gz
cp compose.yaml rollback/compose-before-update.yaml
```

After transferring the release image and matching Compose file, load the image and recreate the camera. No Nginx reload is needed when only the camera changes:

```bash
cd ~/IR-Camera
docker load -i ir-camera-arm64.tar.gz
docker compose config --quiet
docker compose up -d --no-build --pull never --force-recreate
for attempt in $(seq 1 30); do
  if docker exec ir-camera-app python -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:3000/healthz", timeout=2).read().decode())' \
    && curl --noproxy "*" --fail --silent --show-error http://localhost/IR-Camera/healthz; then
    break
  fi
  if [ "$attempt" -eq 30 ]; then
    docker logs --tail 100 ir-camera-app
    exit 1
  fi
  sleep 2
done
docker compose ps
```

The transferred `compose.yaml` references `ir-camera:arm64`, matching the image archive. Direct host requests to port 3000 no longer work by design. Use container-local health or the `/IR-Camera/healthz` proxy route. `--noproxy "*"` prevents a host proxy from intercepting localhost requests.

Inspect the camera container and Docker service-name health after deployment:

```bash
docker logs --tail 50 ir-camera-app
docker exec ir-camera-nginx wget -qO- http://ir-camera:3000/healthz
docker port ir-camera-app
```

The final command should show no published ports. The CI workflow still uses a temporary localhost port mapping on its isolated runner to smoke-test the image; that is not the production Compose configuration.

The CI workflow verifies loading the compressed Docker archive directly. If an older Docker Engine rejects compressed input, stream decompression without installing anything:

```bash
gzip -dc ~/IR-Camera/ir-camera-arm64.tar.gz | docker load
```

The camera opens when the first browser requests `/IR-Camera/video_feed`. Container logs report I2C or sensor initialization failures.

## Roll back

Keep previous successful GitHub Release images under `~/IR-Camera/rollback/`, with the release tag or version in each filename. Loading an older release image restores its included `ir-camera:arm64` tag:

```bash
docker load -i ~/IR-Camera/rollback/ir-camera-arm64-<release-tag>.tar.gz
cd ~/IR-Camera
docker compose up -d --no-build --pull never --force-recreate
docker exec ir-camera-app python -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:3000/healthz", timeout=2).read().decode())'
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
