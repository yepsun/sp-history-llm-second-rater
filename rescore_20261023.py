#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2026-10 补跑稳定性分析（应审稿人 P4 要求）。

严格复刻 output/ai_score_test.py（2026-07-09 主评分/重跑脚本）的
prompt、请求参数、解析与截断逻辑，仅新增：
  - 每次调用记录 API 回显的 model 版本与 token 用量
  - 两轮独立评分（runA / runB），断点续跑
  - 结果写入 rerun_20261023/<run>/，不触碰任何旧文件

用法:
  python3 rescore_20261023.py test            # 单条试跑
  python3 rescore_20261023.py run A           # 第一轮
  python3 rescore_20261023.py run B           # 第二轮
  python3 rescore_20261023.py report          # 汇总元数据
"""
import os, re, io, sys, json, time, hashlib, threading, subprocess
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from openpyxl import load_workbook, Workbook

BASE_DIR = Path("/Users/Yepsun/Mywork/Vscodeprojects/programs/STT/output")
RECORDS_DIR = BASE_DIR / "students_records"
HUMAN_DIR = BASE_DIR / "human_scores"
OUT_ROOT = BASE_DIR / "rerun_20261023"
META_PATH = OUT_ROOT / "meta.json"

# ---- 密钥：STT/.env 优先 ----
def load_key():
    env_file = BASE_DIR.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                k = line.split("=", 1)[1].strip()
                if k:
                    return k
    return os.environ.get("DEEPSEEK_API_KEY", "")

API_KEY = load_key()

# ---- 与 2026-07 原脚本逐字一致：prompt 模板直接从原文件提取 ----
def _extract_template():
    src = (BASE_DIR / "ai_score_test.py").read_text(encoding="utf-8")
    m = re.search(r'SCORE_PROMPT_TEMPLATE = """(.*?)"""', src, re.S)
    if not m:
        raise RuntimeError("cannot extract SCORE_PROMPT_TEMPLATE from ai_score_test.py")
    return m.group(1)

SCORE_PROMPT_TEMPLATE = _extract_template()

PAYLOAD_FIXED = {
    "model": "deepseek-v4-pro",
    "temperature": 0,
    "max_tokens": 8192,
    "stream": False,
    "thinking": {"type": "disabled"},
}
SYSTEM_MSG = "你是一名专业的医学教育评估专家，擅长对医学生SP问诊记录进行评分。"

_lock = threading.Lock()

def call_deepseek(prompt, retries=4):
    payload = dict(PAYLOAD_FIXED)
    payload["messages"] = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(retries):
        try:
            curl_cmd = [
                "curl", "-s", "-X", "POST",
                "https://api.deepseek.com/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {API_KEY}",
                "-d", json.dumps(payload, ensure_ascii=False),
            ]
            result = subprocess.run(curl_cmd, capture_output=True, text=True,
                                    encoding="utf-8", timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"curl failed: {result.stderr[:200]}")
            response = json.loads(result.stdout)
            if "choices" in response and response["choices"]:
                meta = {
                    "model_echo": response.get("model"),
                    "usage": response.get("usage"),
                    "created": response.get("created"),
                }
                return response["choices"][0]["message"]["content"], meta
            if "error" in response:
                raise RuntimeError(f"API error: {str(response['error'])[:300]}")
            raise RuntimeError(f"unexpected response: {str(response)[:300]}")
        except Exception as e:
            with _lock:
                print(f"    [retry {attempt+1}/{retries}] {e}", flush=True)
            if attempt < retries - 1:
                time.sleep(min(60, 2 ** attempt * 3))
    return None, None

# ---- 以下解析/建表逻辑与 2026-07 原脚本逐字一致 ----
def build_criteria_text(ws):
    lines = []
    for r in range(1, ws.max_row + 1):
        seq = ws.cell(row=r, column=1).value
        content = ws.cell(row=r, column=2).value
        score = ws.cell(row=r, column=3).value
        if seq is None or content is None or score is None:
            continue
        seq_str = str(seq).strip()
        content_str = str(content).strip()
        score_str = str(score).strip()
        if seq_str == "序列" or seq_str == "合计：" or not content_str:
            continue
        lines.append(f"评估内容{seq_str}：{content_str}（分值：{score_str}）")
    return "\n".join(lines)

def parse_ai_result(text):
    results = {}
    blocks = re.split(r'\n(?=评估内容\d+：)', text.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = re.match(r'评估内容(\d+)[：:]\s*【\s*([^】]+?)\s*】', block)
        if not m:
            continue
        seq = m.group(1)
        inner = m.group(2).strip()
        score = None
        actual_ms = re.findall(r'实际得分[：:]\s*(\d+(?:\.5)?)', block)
        if actual_ms:
            score = actual_ms[-1]
        elif re.match(r'^\d+(?:\.5)?$', inner):
            score = inner
        else:
            m2 = re.search(r'(?:得分|分值)[：:]\s*(\d+(?:\.5)?)', inner)
            if m2:
                score = m2.group(1)
            else:
                m3 = re.search(r'【\s*(?:实际得分|得分|分值)\s*】\s*[：:]*\s*(\d+(?:\.5)?)', block)
                if m3:
                    score = m3.group(1)
        if score is None:
            continue
        basis = ""
        reason = ""
        bm = re.search(r'依据[：:]\s*(.+?)(?=\n推理[：:]|$)', block, re.DOTALL)
        rm = re.search(r'推理[：:]\s*(.+?)(?=\n评估内容\d+[：:]|$)', block, re.DOTALL)
        if bm:
            basis = bm.group(1).strip()
        if rm:
            reason = rm.group(1).strip()
        if reason and float(score) > 0 and any(p in reason for p in ['不得分', '得0分', '给0分']):
            score = "0"
        results[seq] = {"score": score, "basis": basis, "reason": reason}
    return results

def build_output_sheet(ws_out, ws_human, ai_results):
    headers = ["序列", "评估内容及要求", "分值", "人工得分", "AI得分", "AI依据", "AI推理"]
    for c, h in enumerate(headers, 1):
        ws_out.cell(row=1, column=c, value=h)
    row_idx = 2
    for r in range(1, ws_human.max_row + 1):
        seq = ws_human.cell(row=r, column=1).value
        content = ws_human.cell(row=r, column=2).value
        score = ws_human.cell(row=r, column=3).value
        human_score = ws_human.cell(row=r, column=4).value
        if seq is None or content is None or score is None:
            continue
        seq_str = str(seq).strip()
        if seq_str == "序列" or seq_str == "合计：":
            continue
        if not str(content).strip():
            continue
        ai = ai_results.get(seq_str, {"score": "", "basis": "", "reason": ""})
        try:
            human_score_num = float(human_score) if human_score is not None and str(human_score).strip() != "" else None
        except (ValueError, TypeError):
            human_score_num = None
        try:
            ai_score_num = float(ai["score"]) if ai["score"] is not None and str(ai["score"]).strip() != "" else None
        except (ValueError, TypeError):
            ai_score_num = None
        if ai_score_num is not None:
            max_score = float(score)
            if ai_score_num > max_score:
                ai["reason"] = (ai["reason"] + f"（注：模型原始输出分数 {ai_score_num:g} 超出该条目满分 {max_score:g}，已按满分截断）").strip()
                ai_score_num = max_score
            elif ai_score_num < 0:
                ai["reason"] = (ai["reason"] + f"（注：模型原始输出分数 {ai_score_num:g} 为负，已按 0 分截断）").strip()
                ai_score_num = 0.0
        ws_out.cell(row=row_idx, column=1, value=int(seq_str))
        ws_out.cell(row=row_idx, column=2, value=content)
        ws_out.cell(row=row_idx, column=3, value=float(score))
        ws_out.cell(row=row_idx, column=4, value=human_score_num)
        ws_out.cell(row=row_idx, column=5, value=ai_score_num)
        ws_out.cell(row=row_idx, column=6, value=ai["basis"])
        ws_out.cell(row=row_idx, column=7, value=ai["reason"])
        row_idx += 1
    total_row = row_idx
    ws_out.cell(row=total_row, column=1, value="合计：")
    ws_out.cell(row=total_row, column=4, value=f"=SUM(D2:D{total_row - 1})")
    ws_out.cell(row=total_row, column=5, value=f"=SUM(E2:E{total_row - 1})")

def find_transcript(sp_label, student):
    for p in sorted(RECORDS_DIR.glob(f"**/{student}{sp_label}_*.txt")):
        return p
    return None

def list_encounters():
    """[(student, sp_label, sheet, human_file, out_name)]：全部 276 场"""
    encs = []
    for hf in sorted(HUMAN_DIR.glob("*.xlsx")):
        student = hf.stem
        wb = load_workbook(hf, read_only=False)
        for sheet in wb.sheetnames:
            m = re.search(r'SP(\d+)', sheet)
            if not m:
                continue
            sp = f"SP{m.group(1).zfill(2)}"
            encs.append((student, sp, sheet, hf, f"{student}{sp}.xlsx"))
    return encs

def score_one(enc, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    student, sp, sheet, hf, out_name = enc
    ws_human = load_workbook(hf)[sheet]
    transcript_path = find_transcript(sp, student)
    if not transcript_path:
        return ("no_transcript", out_name, None)
    prompt = SCORE_PROMPT_TEMPLATE.format(
        criteria_text=build_criteria_text(ws_human),
        transcript=transcript_path.read_text(encoding="utf-8"))
    content, meta = call_deepseek(prompt)
    if content is None:
        return ("api_fail", out_name, meta)
    ai_results = parse_ai_result(content)
    wb_out = Workbook()
    ws_out = wb_out.active
    ws_out.title = sheet
    build_output_sheet(ws_out, ws_human, ai_results)
    wb_out.save(out_dir / out_name)
    return ("ok", out_name, {"n_items": len(ai_results), **(meta or {})})

def append_meta(run, records):
    OUT_ROOT.mkdir(exist_ok=True)
    with _lock:
        meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
        entry = {
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "rescore_20261023.py (replica of ai_score_test.py, 2026-07-09)",
            "endpoint": "https://api.deepseek.com/v1/chat/completions",
            "payload": PAYLOAD_FIXED,
            "records": records,
        }
        meta[run] = entry
        META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=1))

def cmd_test():
    encs = list_encounters()
    enc = next(e for e in encs if e[0] == "丁愚昔" and e[1] == "SP04")
    print(f"TEST: {enc[0]} {enc[1]} ({enc[2]})")
    status, name, info = score_one(enc, OUT_ROOT / "test")
    print("status:", status, "| info:", json.dumps(info, ensure_ascii=False)[:400])

def cmd_run(run):
    out_dir = OUT_ROOT / run
    out_dir.mkdir(parents=True, exist_ok=True)
    encs = list_encounters()
    todo = [e for e in encs if not (out_dir / e[4]).exists()]
    print(f"run {run}: {len(encs)} encounters total, {len(todo)} to do", flush=True)
    t0 = time.time()
    done = fail = miss = 0
    models, usage_tot = set(), {"prompt": 0, "completion": 0}
    def work(e):
        return score_one(e, out_dir)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(work, e): e for e in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            status, name, info = fut.result()
            if status == "ok":
                done += 1
                if info and info.get("model_echo"):
                    models.add(info["model_echo"])
                if info and info.get("usage"):
                    usage_tot["prompt"] += info["usage"].get("prompt_tokens", 0)
                    usage_tot["completion"] += info["usage"].get("completion_tokens", 0)
            elif status == "no_transcript":
                miss += 1
            else:
                fail += 1
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] ok={done} fail={fail} miss={miss} "
                      f"elapsed={el/60:.0f}min eta={el/i*(len(todo)-i)/60:.0f}min "
                      f"models={sorted(models)}", flush=True)
    append_meta(run, {"done": done, "fail": fail, "no_transcript": miss,
                      "model_echo": sorted(models), "usage_total": usage_tot})
    print(f"RUN {run} COMPLETE: ok={done} fail={fail} no_transcript={miss}")

if __name__ == "__main__":
    if not API_KEY:
        sys.exit("no API key found in STT/.env")
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "test":
        cmd_test()
    elif mode == "run":
        cmd_run(sys.argv[2] if len(sys.argv) > 2 else "A")
    else:
        print(__doc__)
