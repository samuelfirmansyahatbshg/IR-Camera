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

- `ir-camera-app`: Gunicorn and Flask on host port 3000
- `ir-camera-nginx`: separately managed Nginx container on host port 80
- MLX90640 access: `/dev/i2c-1` passed to `ir-camera-app`

Camera source and Docker build files stay in `~/IR-Camera/`. The Nginx configuration stays outside this repository at `~/nginx/default.conf`.

## First deployment

Keep the local `IR-Camera/` and `nginx/` directories beside each other. Replace `user@server` with the Raspberry Pi SSH destination:

```sh
ssh user@server 'mkdir -p ~/IR-Camera ~/nginx'
scp -r ./IR-Camera/. user@server:~/IR-Camera/
scp ./nginx/default.conf user@server:~/nginx/default.conf
ssh user@server 'cd ~/IR-Camera && docker compose up -d --build --remove-orphans'
```

Start the separately managed Nginx container:

```sh
ssh user@server 'docker rm -f ir-camera-nginx 2>/dev/null || true; docker run -d --name ir-camera-nginx -p 80:80 --restart unless-stopped --add-host=host.docker.internal:host-gateway -v "$HOME/nginx/default.conf:/etc/nginx/conf.d/default.conf:ro" nginx:alpine'
```

Open `http://<raspberry-pi-address>/IR-Camera/`.

## Deploy an updated Python application

```sh
scp ./IR-Camera/ir_camera.py user@server:~/IR-Camera/ir_camera.py
ssh user@server 'cd ~/IR-Camera && docker compose up -d --build'
```

Deploy the entire camera project when `Dockerfile`, `requirements.txt`, or `compose.yaml` changes:

```sh
scp -r ./IR-Camera/. user@server:~/IR-Camera/
ssh user@server 'cd ~/IR-Camera && docker compose up -d --build --remove-orphans'
```

Deploy an updated external Nginx configuration independently:

```sh
scp ./nginx/default.conf user@server:~/nginx/default.conf
ssh user@server 'docker restart ir-camera-nginx'
```

## Operations

```sh
cd ~/IR-Camera
docker compose ps
docker compose logs -f app
docker compose restart
docker compose down
```

Check the application directly or through Nginx:

```sh
curl --fail http://localhost:3000/healthz
curl --fail http://localhost/IR-Camera/healthz
```

The camera is opened when the first browser requests `/IR-Camera/video_feed`. Container logs report I2C or sensor initialization failures.

## Local tests

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
