# -*- coding: utf-8 -*-
"""Получение лога kernel через Kaggle GraphQL API (с XSRF)."""
import json
import os

import requests

creds = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
auth = (creds["username"], creds["key"])

s = requests.Session()
s.auth = auth
s.get("https://www.kaggle.com", timeout=30)
xsrf = s.cookies.get("XSRF-TOKEN", "")

q = {
    "operationName": "KernelOutputPage",
    "variables": {"kernelSlug": "leeth-md8-full-ft", "ownerSlug": "hunzlee"},
    "query": "query KernelOutputPage($kernelSlug: String!, $ownerSlug: String!) { kernelOutput(kernelSlug: $kernelSlug, ownerSlug: $ownerSlug) { status log } }",
}
r = s.post(
    "https://www.kaggle.com/api/i/graphql.GetGraphQLRequest",
    json=q,
    headers={"X-XSRF-TOKEN": xsrf},
    timeout=60,
)
print("status:", r.status_code)
if r.status_code == 200:
    ko = (r.json().get("data") or {}).get("kernelOutput") or {}
    print("kernel status:", ko.get("status"))
    print("=== LOG TAIL ===")
    print((ko.get("log") or "")[-2500:])
else:
    print(r.text[:500])