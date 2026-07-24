#!/usr/bin/env python3
"""查询中转站 API Key 的额度使用情况。"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

BASE_URL = "http://121.15.155.221:8000"
DEFAULT_API_KEY = os.environ.get("USAGE_API_KEY", "sk-NLuv0FY4ujKoJn6YafrDiXgZPwQB5Iqq2Vx0NuRLwgxJNRc9")


def post_json(path, payload, timeout=10):
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fmt_ts(ts):
    if not ts:
        return "N/A"
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%d %H:%M:%S %Z")
    )


def main():
    parser = argparse.ArgumentParser(description="查询额度使用情况")
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API Key，也可通过环境变量 USAGE_API_KEY 传入",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("未提供 API Key（--api-key 或环境变量 USAGE_API_KEY）", file=sys.stderr)
        sys.exit(1)

    try:
        result = post_json("/api/usage", {"apiKey": args.api_key})
    except urllib.error.URLError as e:
        print(f"请求失败: {e}", file=sys.stderr)
        sys.exit(1)

    if not result.get("ok"):
        print("查询失败:", result, file=sys.stderr)
        sys.exit(1)

    usage = result["usage"]
    raw = result.get("raw", {}).get("data", {})
    quota_usd = usage.get("quotaUsd", {})

    print(f"令牌名称     : {raw.get('name', 'N/A')}")
    print(
        f"总额度       : {usage['totalGranted']:,} tokens "
        f"(${quota_usd.get('totalGranted', 0):.2f})"
    )
    print(
        f"已使用       : {usage['totalUsed']:,} tokens "
        f"(${quota_usd.get('totalUsed', 0):.2f})"
    )
    print(
        f"可用余额     : {usage['totalAvailable']:,} tokens "
        f"(${quota_usd.get('totalAvailable', 0):.2f})"
    )
    print(f"是否无限额度 : {'是' if usage.get('unlimitedQuota') else '否'}")
    print(f"到期时间     : {fmt_ts(usage.get('expiresAt'))}")


if __name__ == "__main__":
    main()
