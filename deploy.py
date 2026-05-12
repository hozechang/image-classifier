"""
一键部署：分类器网站到 GitHub Pages
"""
import urllib.request, json, os, subprocess, sys

TOKEN = os.environ.get("GITHUB_TOKEN", "")
USER = "hozechang"
REPO = "image-classifier"
DIR = r"E:\share\play\classifier-app"

if not TOKEN:
    print("需要 GITHUB_TOKEN 环境变量")
    sys.exit(1)

# 1. Create repo
print("1/3 创建仓库...")
url = "https://api.github.com/user/repos"
data = json.dumps({"name": REPO, "description": "Image classifier with ONNX Runtime Web", "private": False, "has_pages": True}).encode()
req = urllib.request.Request(url, data=data, method="POST")
req.add_header("Authorization", f"token {TOKEN}")
req.add_header("Accept", "application/vnd.github+json")
req.add_header("User-Agent", "Python")
try:
    with urllib.request.urlopen(req) as r:
        print(f"  仓库已创建: https://github.com/{USER}/{REPO}")
except urllib.error.HTTPError as e:
    if e.code == 422:
        print(f"  仓库已存在，直接推送")
    else:
        print(f"  错误: {e.read().decode()[:300]}")
        sys.exit(1)

# 2. Git push
print("2/3 推送代码...")
os.chdir(DIR)

# Init if needed
if not os.path.exists(os.path.join(DIR, ".git")):
    subprocess.run(["git", "init"], check=True)
    subprocess.run(["git", "checkout", "-b", "main"], check=True)

remote_url = f"https://{USER}:{TOKEN}@github.com/{USER}/{REPO}.git"
result = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True)
if result.returncode != 0:
    subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
else:
    subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True)

subprocess.run(["git", "add", "-A"], check=True)
subprocess.run(["git", "commit", "-m", "Deploy image classifier"], check=True)
subprocess.run(["git", "push", "-u", "origin", "main", "--force"], check=True)
print("  推送成功!")

# 3. Enable Pages
print("3/3 启用 GitHub Pages...")
url2 = f"https://api.github.com/repos/{USER}/{REPO}/pages"
data2 = json.dumps({"source": {"branch": "main", "path": "/"}}).encode()
req2 = urllib.request.Request(url2, data=data2, method="POST")
req2.add_header("Authorization", f"token {TOKEN}")
req2.add_header("Accept", "application/vnd.github+json")
req2.add_header("User-Agent", "Python")
try:
    with urllib.request.urlopen(req2) as r:
        print(f"  Pages 已启用")
except urllib.error.HTTPError as e:
    if e.code == 409:
        # Already exists, update
        req3 = urllib.request.Request(url2, data=data2, method="PUT")
        req3.add_header("Authorization", f"token {TOKEN}")
        req3.add_header("Accept", "application/vnd.github+json")
        req3.add_header("User-Agent", "Python")
        with urllib.request.urlopen(req3) as r:
            print(f"  Pages 已更新")

print(f"\n部署完成!")
print(f"网站地址: https://{USER}.github.io/{REPO}/")
print(f"(等待 1-2 分钟生效)")
