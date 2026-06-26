# CloudML Image Push Runbook

本文记录把镜像从火山 registry 推到 CloudML registry 的固定流程。

源 registry:

```bash
robot-cn-beijing.cr.volces.com/jzh/part-prompt-seg:<TAG>
```

CloudML 目标 registry:

```bash
micr.cloud.mioffice.cn/jzh/part-prompt-seg:<TAG>
```

## 先确认登录态

两边都需要先 `docker login`。不要把密码写进脚本或文档。

```bash
docker login robot-cn-beijing.cr.volces.com
docker login micr.cloud.mioffice.cn
```

登录后凭据在 `~/.docker/config.json`，后面的脚本会读取这个文件。

## 小镜像: Docker pull/tag/push

本机 Docker 数据盘足够时，可以直接走 Docker：

```bash
SRC=robot-cn-beijing.cr.volces.com/jzh/part-prompt-seg:0621-1
DST=micr.cloud.mioffice.cn/jzh/part-prompt-seg:0621-1

docker pull "$SRC"
docker tag "$SRC" "$DST"
docker push "$DST"
```

推完看 Docker 返回的 digest。源和目标 digest 一致才算完成。

```bash
docker manifest inspect "$SRC" >/tmp/src.json
docker manifest inspect "$DST" >/tmp/dst.json
cmp -s /tmp/src.json /tmp/dst.json && echo "manifest identical"
```

注意：registry 页面显示的大小可能是压缩层大小，本地 `docker images` 显示的是解压后的
image size，两者不一定一致。最终以 manifest digest 和 layer digest 列表为准。

## 大镜像: 不要直接 docker pull

`0621-2` 这类大镜像压缩层就接近 36 GB。本机 Docker root 只有 40 GB 时，
`docker pull` 会在解压/导入 layer 时失败：

```text
no space left on device
```

这时用 registry-to-registry 复制，不把完整镜像解压到本机 Docker：

```bash
python scripts/ops/copy_registry_image.py \
  --src robot-cn-beijing.cr.volces.com/jzh/part-prompt-seg:0621-2 \
  --dst micr.cloud.mioffice.cn/jzh/part-prompt-seg:0621-2
```

脚本会：

- 读取源 manifest；
- 检查目标仓库已有 blob，已有则跳过；
- 只复制缺失 blob；
- 最后 PUT 原始 manifest 到目标 tag；
- 重新 GET 目标 manifest，确认 digest、layer 数和 manifest bytes。

## CloudML Harbor 的坑

`micr.cloud.mioffice.cn` 后面是 Harbor。普通 `docker push` 没问题，但手写 registry API
有两个容易踩的点：

- `POST /v2/<repo>/blobs/uploads/` 不能只用 basic auth，否则可能返回
  `CSRF token invalid`；
- 对 `micr` 的 HEAD/GET 请求会设置 `sid` cookie，后续上传请求如果带着这个 cookie，
  也可能触发 CSRF。

`scripts/ops/copy_registry_image.py` 已处理这两个问题：

- 对 CloudML 主动从 `https://micr-internal.cloud.mioffice.cn/service/token` 申请
  Harbor token；
- 目标 registry 的上传请求使用无 cookie 的新 session。

## 本次已验证的例子

### 0621-1

```text
robot-cn-beijing.cr.volces.com/jzh/part-prompt-seg:0621-1
micr.cloud.mioffice.cn/jzh/part-prompt-seg:0621-1

manifest digest:
sha256:098f730fcaf60cc6404ec73d50c31f963c29ac6c9c5379a91a15098bf8d70e3f

config digest:
sha256:562f76844370aa99a0d41d8aa677b3af0236ec557f12e4e7480576d87ed31338

layers:
28

compressed size:
10.510 GB
```

### 0621-2

```text
robot-cn-beijing.cr.volces.com/jzh/part-prompt-seg:0621-2
micr.cloud.mioffice.cn/jzh/part-prompt-seg:0621-2

manifest digest:
sha256:885c7a3dfa353de4e677f5768f3e6ace45a0ab8d3d408febe7a9e2b3095ad94a

config digest:
sha256:188ae8950f29a34ad0f51f9206bbbb02f94517bb40fd876a66cf8a175e82c495

layers:
41

compressed size:
35.958 GB

manifest bytes equal:
True
```

## 常用校验命令

只看远端 manifest，不依赖本地 image：

```bash
python - <<'PY'
import json, subprocess

images = [
    "robot-cn-beijing.cr.volces.com/jzh/part-prompt-seg:0621-2",
    "micr.cloud.mioffice.cn/jzh/part-prompt-seg:0621-2",
]

for image in images:
    manifest = subprocess.check_output(["docker", "manifest", "inspect", image], text=True)
    data = json.loads(manifest)
    layers = data.get("layers", [])
    print(image)
    print("  config:", data.get("config", {}).get("digest"))
    print("  layers:", len(layers))
    print("  compressed_GB:", round(sum(x.get("size", 0) for x in layers) / 1e9, 3))
PY
```

如果页面大小和命令结果不一致，以 `Docker-Content-Digest`、config digest、layer digest
列表为准。
