#!/usr/bin/env python
from __future__ import annotations

import getpass
from pathlib import Path


FIELDS = [
    ("KRX_ID", "KRX Data Marketplace 아이디", False),
    ("KRX_PW", "KRX Data Marketplace 비밀번호", True),
    ("DATA_GO_KR_SERVICE_KEY", "공공데이터포털 일반 인증키(대차 API)", True),
    ("OPENDART_API_KEY", "OpenDART 인증키", True),
    ("ECOS_API_KEY", "한국은행 ECOS 인증키", True),
]


def parse_existing(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main() -> None:
    project = Path(__file__).resolve().parent
    path = project / ".env.data.local"
    existing = parse_existing(path)
    print("입력값은 이 PC의 .env.data.local에만 저장되며 화면/로그에 비밀번호를 출력하지 않습니다.")
    print("채팅이나 Git에 계정 비밀번호를 붙여넣지 마세요.\n")
    values = dict(existing)
    for key, label, secret in FIELDS:
        current = existing.get(key, "")
        prompt = f"{label} [{ '설정됨' if current else '미설정' }, Enter=유지/건너뜀]: "
        value = getpass.getpass(prompt) if secret else input(prompt)
        value = value.strip()
        if value:
            values[key] = value
    lines = ["# 로컬 전용. Git/공유 금지."]
    for key, _, _ in FIELDS:
        lines.append(f"{key}={quote(values.get(key, ''))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장 완료: {path}")
    print("다음 실행: RUN_REQUIRED_DATA_DOWNLOAD.bat")


if __name__ == "__main__":
    main()
