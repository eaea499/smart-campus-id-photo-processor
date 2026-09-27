# 在线证件照 API 部署

服务只处理内存中的上传内容，默认监听 `127.0.0.1:8090`，由 Nginx 提供公网入口。

## 服务器准备

```bash
cd /opt
tar -xzf /root/smart-campus-id-photo-api.tar.gz
cd /opt/smart-campus-id-photo-processor
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r web_api/requirements.txt
```

原项目中的 `IDPhotoProcessor/`、模型目录和 `yolov8n.pt` 需要和 `web_api/` 一起部署。服务器还需安装与原项目一致的 OpenCV、NumPy、Ultralytics、Paddle 运行依赖；建议使用已经验证过的 Python 环境。

## systemd

```bash
cp web_api/smart-campus-id-photo.service /etc/systemd/system/smart-campus-id-photo.service
systemctl daemon-reload
systemctl enable --now smart-campus-id-photo
curl http://127.0.0.1:8090/health
```

## Nginx

将 `deploy/id-photo-api.nginx.conf` 中的两个 `location` 放进 `eaea499.cn` 的 HTTPS server 块，然后执行：

```bash
nginx -t && systemctl reload nginx
```
