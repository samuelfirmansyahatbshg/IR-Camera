# IR Camera

MLX90640 Raspberry Pi thermal camera web streamer. The browser displays the supplied OpenCV inferno stream at 640 x 480 through Nginx.

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

## Containers

- `ir-camera-nginx`: host port 80
- `ir-camera-app`: Gunicorn and Flask on private container port 3000
- MLX90640 access: `/dev/i2c-1` passed to `ir-camera-app`

All project files stay in `~/IR-Camera/`. Docker manages its image and container storage system-wide.

## First deployment

From the computer containing this repository, replace `user@server` with the Raspberry Pi SSH destination:

```sh
ssh user@server 'mkdir -p ~/IR-Camera'
scp -r ./IR-Camera/. user@server:~/IR-Camera/
ssh user@server 'cd ~/IR-Camera && docker compose up -d --build --remove-orphans'
```

Open `http://<raspberry-pi-address>/`.

## Deploy an updated Python application

```sh
scp ./IR-Camera/ir_camera.py user@server:~/IR-Camera/ir_camera.py
ssh user@server 'cd ~/IR-Camera && docker compose up -d --build app'
```

Deploy the entire directory when `Dockerfile`, `requirements.txt`, `compose.yaml`, or the Nginx configuration changes:

```sh
scp -r ./IR-Camera/. user@server:~/IR-Camera/
ssh user@server 'cd ~/IR-Camera && docker compose up -d --build --remove-orphans'
```

## Operations

```sh
cd ~/IR-Camera
docker compose ps
docker compose logs -f app
docker compose restart
docker compose down
```

The liveness endpoint is available through Nginx:

```sh
curl --fail http://localhost/healthz
```

The camera is opened when the first browser requests `/video_feed`. Container logs report I2C or sensor initialization failures.

## Local tests

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
