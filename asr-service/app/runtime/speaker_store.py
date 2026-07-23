"""声纹库存储（data/speakers.db，SQLite stdlib + 内存质心矩阵检索）。

体例照抄 task_store.py（单连接 check_same_thread=False + auto_vacuum=INCREMENTAL
+ WAL + threading.Lock 串行化 + 删除后 incremental_vacuum），两点刻意差异：
- 错误语义相反：本库是管理 API 的事实源，读写失败一律 raise SpeakerStoreError
  （TaskStore 是旁路自吞）；唯一例外是 audit——审计是旁路，失败仅 WARN。
- 永不自动清理：无 TTL、无启动清理（声纹是长期积累资产），唯一删除途径 =
  delete_speaker（被遗忘权，物理回收）。
- 自动说话人认领：元数据更新、模板替换和 claim_key 幂等结果同事务提交，
  保留原 speaker_id；仅 source=auto 可执行。

检索模型：启动/写后全量重载内存质心矩阵 [N,192]，identify 为纯 numpy 点积
（千人级 <1ms）；重载构造完成后一次性替换引用，读侧无锁也无撕裂读。
"""
import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS speakers (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  note       TEXT,
  consent    INTEGER NOT NULL CHECK (consent = 1),
  source     TEXT NOT NULL DEFAULT 'manual',
  model_tag  TEXT NOT NULL,
  centroid   BLOB NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS templates (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  speaker_id TEXT NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
  vector     BLOB NOT NULL,
  dur_sec    REAL NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_templates_speaker ON templates(speaker_id);
CREATE TABLE IF NOT EXISTS speaker_claims (
  claim_key  TEXT PRIMARY KEY,
  speaker_id TEXT NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
  response   TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_speaker_claims_speaker
  ON speaker_claims(speaker_id);
CREATE TABLE IF NOT EXISTS audit_log (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  ts         TEXT NOT NULL,
  action     TEXT NOT NULL,
  speaker_id TEXT,
  detail     TEXT
);
"""


class SpeakerStoreError(Exception):
    """声纹库操作失败——路由层据此转 HTTP 错误（与 TaskStore 容错自吞语义相反）。"""


class SpeakerNotFoundError(SpeakerStoreError):
    """目标说话人/模板不存在——路由层映射 404（其余 SpeakerStoreError 一律 500）。"""


class SpeakerClaimConflictError(SpeakerStoreError):
    """认领目标或幂等键冲突——路由层映射为带冲突对象的 409。"""

    def __init__(self, code: str, message: str, *, speaker_id: str,
                 speaker_name: str, source: str):
        super().__init__(message)
        self.code = code
        self.speaker_id = speaker_id
        self.speaker_name = speaker_name
        self.source = source


class SpeakerStore:
    SCHEMA_VERSION = 2
    DIM = 192
    MAX_TEMPLATES = 16          # 每人模板上限（防滥用）
    MAX_CLAIM_KEY_LENGTH = 128
    _NORM_TOL = 1e-3            # L2 归一容差（入库向量须由引擎出口归一）

    def __init__(self, db_path: str, model_tag: str):
        self.db_path = db_path
        self.model_tag = model_tag
        self._lock = threading.Lock()
        self._cache_version = 0
        self._cache_dirty = True
        # identify 读侧无锁：(matrix, ids, names, sources) 装入单引用整体交换——
        # 多属性分写在字节码间隙可被读侧观察到撕裂（GIL 不保证多字节码序列原子）
        self._cache: tuple[
            np.ndarray, list[str], list[str], list[str]
        ] = (np.zeros((0, self.DIM), dtype=np.float32), [], [], [])

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # auto_vacuum 须在建表前设置才对新库生效
            self._conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")   # 级联删除依赖
            self._conn.executescript(_DDL)
            schema_row = self._conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if schema_row is None:
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                    (str(self.SCHEMA_VERSION),),
                )
            else:
                try:
                    schema_version = int(schema_row["value"])
                except (TypeError, ValueError) as e:
                    raise SpeakerStoreError("声纹库 schema_version 无效") from e
                if schema_version > self.SCHEMA_VERSION:
                    raise SpeakerStoreError(
                        f"声纹库版本过新: {schema_version} > {self.SCHEMA_VERSION}"
                    )
                if schema_version < self.SCHEMA_VERSION:
                    self._conn.execute(
                        "UPDATE meta SET value=? WHERE key='schema_version'",
                        (str(self.SCHEMA_VERSION),),
                    )
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('model_tag', ?)",
                (model_tag,),
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('created_at', ?)",
                (datetime.now().isoformat(),),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            raise SpeakerStoreError(f"声纹库初始化失败: {e}") from e
        with self._lock:
            self._reload_cache()
        logger.info(f"声纹库已启用: {db_path}（model_tag={model_tag}，"
                    f"{self.speaker_count} 人，永不自动清理）")

    # ─── 内部工具 ───

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat()

    def _validate_vector(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.DIM:
            raise SpeakerStoreError(f"向量维度须为 {self.DIM}，实得 {v.shape[0]}")
        norm = float(np.linalg.norm(v))
        if abs(norm - 1.0) > self._NORM_TOL:
            raise SpeakerStoreError(f"向量须 L2 归一（norm={norm:.4f}）")
        return v

    def _reload_cache(self):
        """重建内存质心矩阵。须持锁调用；构造完成后整体替换引用（读侧无撕裂）。"""
        self._cache_dirty = True
        try:
            rows = self._conn.execute(
                "SELECT id, name, source, centroid FROM speakers"
            ).fetchall()
        except sqlite3.Error as e:
            raise SpeakerStoreError(f"质心缓存重载失败: {e}") from e
        ids, names, sources, vecs = [], [], [], []
        for r in rows:
            ids.append(r["id"])
            names.append(r["name"])
            sources.append(r["source"])
            vecs.append(np.frombuffer(r["centroid"], dtype=np.float32))
        matrix = (np.stack(vecs) if vecs
                  else np.zeros((0, self.DIM), dtype=np.float32))
        self._cache = (matrix, ids, names, sources)
        self._cache_version += 1
        self._cache_dirty = False

    def _evict_from_cache(self, speaker_id: str):
        """按 id 摘除内存缓存单项（delete_speaker 重载失败时的兜底）。须持锁调用。"""
        matrix, ids, names, sources = self._cache
        if speaker_id not in ids:
            return
        keep = [i for i, x in enumerate(ids) if x != speaker_id]
        self._cache = (
            matrix[keep],
            [ids[i] for i in keep],
            [names[i] for i in keep],
            [sources[i] for i in keep],
        )
        self._cache_version += 1

    def _recompute_centroid(self, speaker_id: str):
        """模板均值 → L2 重归一 → 回写 speakers.centroid。须持锁调用；0 模板时保留旧质心。"""
        rows = self._conn.execute(
            "SELECT vector FROM templates WHERE speaker_id=?", (speaker_id,)
        ).fetchall()
        if not rows:
            return
        vecs = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
        centroid = vecs.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm
        self._conn.execute(
            "UPDATE speakers SET centroid=?, updated_at=? WHERE id=?",
            (centroid.astype(np.float32).tobytes(), self._now(), speaker_id),
        )

    def _claim_replay_locked(self, speaker_id: str,
                             claim_key: str) -> dict | None:
        """读取幂等认领结果并校验 key 归属。须持锁调用。"""
        row = self._conn.execute(
            "SELECT c.speaker_id, c.response, s.name, s.source"
            " FROM speaker_claims c JOIN speakers s ON s.id = c.speaker_id"
            " WHERE c.claim_key=?",
            (claim_key,),
        ).fetchone()
        if row is None:
            return None
        if row["speaker_id"] != speaker_id:
            raise SpeakerClaimConflictError(
                "claim_key_conflict",
                "claim_key 已用于其他说话人",
                speaker_id=row["speaker_id"],
                speaker_name=row["name"],
                source=row["source"],
            )
        try:
            response = json.loads(row["response"])
        except (TypeError, json.JSONDecodeError) as e:
            raise SpeakerStoreError(f"认领幂等记录损坏: {e}") from e
        if not isinstance(response, dict):
            raise SpeakerStoreError("认领幂等记录格式错误")
        return response

    def _require_auto_speaker_locked(self, speaker_id: str) -> sqlite3.Row:
        """读取并校验认领目标。须持锁调用，写事务内会再次执行。"""
        row = self._conn.execute(
            "SELECT id, name, source FROM speakers WHERE id=?",
            (speaker_id,),
        ).fetchone()
        if row is None:
            raise SpeakerNotFoundError(f"说话人不存在: {speaker_id}")
        if row["source"] != "auto":
            raise SpeakerClaimConflictError(
                "speaker_not_claimable",
                "仅自动登记的说话人可以被认领",
                speaker_id=row["id"],
                speaker_name=row["name"],
                source=row["source"],
            )
        return row

    # ─── 写路径（失败一律 raise SpeakerStoreError）───

    def enroll_speaker(self, name: str, note: str | None, vectors: list[np.ndarray],
                       durs: list[float], consent: bool, source: str = "manual") -> str:
        """登记说话人（单事务：speakers + N 条 templates）。返回 speaker_id（uuid4 hex）。"""
        if consent is not True:
            raise SpeakerStoreError("登记必须携带 consent=true（数据主体同意）")
        if not vectors or len(vectors) != len(durs):
            raise SpeakerStoreError("模板向量与时长数量不符或为空")
        if len(vectors) > self.MAX_TEMPLATES:
            raise SpeakerStoreError(f"模板数超过上限 {self.MAX_TEMPLATES}")
        validated = [self._validate_vector(v) for v in vectors]
        centroid = np.stack(validated).mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm

        speaker_id = uuid.uuid4().hex
        now = self._now()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO speakers(id, name, note, consent, source, model_tag,"
                    " centroid, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (speaker_id, name, note, 1, source, self.model_tag,
                     centroid.astype(np.float32).tobytes(), now, now),
                )
                self._conn.executemany(
                    "INSERT INTO templates(speaker_id, vector, dur_sec, created_at)"
                    " VALUES(?,?,?,?)",
                    [(speaker_id, v.tobytes(), float(d), now)
                     for v, d in zip(validated, durs)],
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise SpeakerStoreError(f"登记写入失败: {e}") from e
            self._reload_cache()
        self.audit("enroll" if source == "manual" else "auto_enroll",
                   speaker_id, {"name": name, "templates": len(validated)})
        return speaker_id

    def resolve_claim(self, speaker_id: str, claim_key: str) -> dict | None:
        """返回同 key 的原认领结果；首次认领则只校验目标当前可认领。

        该检查用于在音频预处理前快速返回重试或冲突。真正写入时仍会在同一
        事务内重查，避免并发认领的 TOCTOU。
        """
        if not claim_key or len(claim_key) > self.MAX_CLAIM_KEY_LENGTH:
            raise SpeakerStoreError("claim_key 无效")
        with self._lock:
            try:
                replay = self._claim_replay_locked(speaker_id, claim_key)
                if replay is not None:
                    # 首次请求可能已提交但缓存重载失败；重试时顺带自愈。
                    if self._cache_dirty:
                        self._reload_cache()
                    return replay
                self._require_auto_speaker_locked(speaker_id)
                return None
            except sqlite3.Error as e:
                raise SpeakerStoreError(f"认领状态读取失败: {e}") from e

    def claim_auto_speaker(
        self,
        speaker_id: str,
        name: str,
        note: str | None,
        vectors: list[np.ndarray],
        durs: list[float],
        claim_key: str,
        quality_hint: str | None = None,
    ) -> tuple[dict, bool]:
        """认领自动说话人并替换模板，返回 (响应, 是否首次写入)。

        speaker 元数据、旧模板删除、新模板插入和幂等结果在同一事务完成；
        speaker_id 与 created_at 保持不变。
        """
        replay = self.resolve_claim(speaker_id, claim_key)
        if replay is not None:
            return replay, False
        if not vectors or len(vectors) != len(durs):
            raise SpeakerStoreError("模板向量与时长数量不符或为空")
        if len(vectors) > self.MAX_TEMPLATES:
            raise SpeakerStoreError(f"模板数超过上限 {self.MAX_TEMPLATES}")

        validated = [self._validate_vector(v) for v in vectors]
        try:
            validated_durs = [float(dur) for dur in durs]
        except (TypeError, ValueError) as e:
            raise SpeakerStoreError("模板时长无效") from e
        centroid = np.stack(validated).mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm

        now = self._now()
        response: dict
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                replay = self._claim_replay_locked(speaker_id, claim_key)
                if replay is not None:
                    self._conn.rollback()
                    if self._cache_dirty:
                        self._reload_cache()
                    return replay, False
                self._require_auto_speaker_locked(speaker_id)

                self._conn.execute(
                    "DELETE FROM templates WHERE speaker_id=?",
                    (speaker_id,),
                )
                template_ids = []
                for vector, dur in zip(validated, validated_durs):
                    cur = self._conn.execute(
                        "INSERT INTO templates"
                        "(speaker_id, vector, dur_sec, created_at) VALUES(?,?,?,?)",
                        (speaker_id, vector.tobytes(), dur, now),
                    )
                    template_ids.append(int(cur.lastrowid))
                self._conn.execute(
                    "UPDATE speakers SET name=?, note=?, consent=1, source='manual',"
                    " centroid=?, updated_at=? WHERE id=?",
                    (name, note, centroid.astype(np.float32).tobytes(), now,
                     speaker_id),
                )
                response = {
                    "speaker_id": speaker_id,
                    "name": name,
                    "source": "manual",
                    "templates": len(template_ids),
                    "template_ids": template_ids,
                    "quality_hint": quality_hint,
                }
                self._conn.execute(
                    "INSERT INTO speaker_claims"
                    "(claim_key, speaker_id, response, created_at) VALUES(?,?,?,?)",
                    (claim_key, speaker_id,
                     json.dumps(response, ensure_ascii=False), now),
                )
                self._conn.commit()
                self._reload_cache()
            except SpeakerStoreError:
                self._conn.rollback()
                raise
            except sqlite3.Error as e:
                self._conn.rollback()
                raise SpeakerStoreError(f"说话人认领失败: {e}") from e

        self.audit(
            "claim",
            speaker_id,
            {"name": name, "templates": len(validated)},
        )
        return response, True

    def alloc_auto_name(self) -> str:
        """自动登记占位名：meta.auto_name_seq 持锁自增，序号只增不复用（与改名/删除解耦）。"""
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT value FROM meta WHERE key='auto_name_seq'").fetchone()
                seq = int(row["value"]) + 1 if row else 1
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('auto_name_seq', ?)",
                    (str(seq),),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                raise SpeakerStoreError(f"占位名序号分配失败: {e}") from e
        return f"说话人_{seq:02d}"

    def add_template(self, speaker_id: str, vector: np.ndarray, dur: float) -> None:
        """追加模板并重算质心。"""
        v = self._validate_vector(vector)
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM templates WHERE speaker_id=?",
                    (speaker_id,),
                ).fetchone()
                exists = self._conn.execute(
                    "SELECT 1 FROM speakers WHERE id=?", (speaker_id,)).fetchone()
                if exists is None:
                    raise SpeakerNotFoundError(f"说话人不存在: {speaker_id}")
                if row["n"] >= self.MAX_TEMPLATES:
                    raise SpeakerStoreError(f"模板数已达上限 {self.MAX_TEMPLATES}")
                self._conn.execute(
                    "INSERT INTO templates(speaker_id, vector, dur_sec, created_at)"
                    " VALUES(?,?,?,?)",
                    (speaker_id, v.tobytes(), float(dur), self._now()),
                )
                self._recompute_centroid(speaker_id)
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise SpeakerStoreError(f"模板追加失败: {e}") from e
            self._reload_cache()
        self.audit("add_template", speaker_id, {"dur_sec": float(dur)})

    def delete_template(self, speaker_id: str, template_id: int) -> int:
        """删除单条模板，返回剩余模板数；剩 0 时不删 speaker（由调用方提示）。"""
        with self._lock:
            try:
                cur = self._conn.execute(
                    "DELETE FROM templates WHERE id=? AND speaker_id=?",
                    (template_id, speaker_id),
                )
                if cur.rowcount == 0:
                    raise SpeakerNotFoundError(f"模板不存在: {speaker_id}/{template_id}")
                self._recompute_centroid(speaker_id)
                remaining = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM templates WHERE speaker_id=?",
                    (speaker_id,),
                ).fetchone()["n"]
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise SpeakerStoreError(f"模板删除失败: {e}") from e
            self._reload_cache()
        self.audit("delete_template", speaker_id, {"template_id": template_id})
        return int(remaining)

    def update_speaker(self, speaker_id: str, name: str | None = None,
                       note: str | None = None) -> None:
        """改名/备注（不影响 speaker_id 与模板）。"""
        sets, params = [], []
        if name is not None:
            sets.append("name=?")
            params.append(name)
        if note is not None:
            sets.append("note=?")
            params.append(note)
        if not sets:
            return
        sets.append("updated_at=?")
        params += [self._now(), speaker_id]
        with self._lock:
            try:
                cur = self._conn.execute(
                    f"UPDATE speakers SET {', '.join(sets)} WHERE id=?", tuple(params))
                self._conn.commit()
            except sqlite3.Error as e:
                raise SpeakerStoreError(f"说话人更新失败: {e}") from e
            if cur.rowcount == 0:
                raise SpeakerNotFoundError(f"说话人不存在: {speaker_id}")
            if name is not None:
                self._reload_cache()        # identify 返回 name，需要同步
        self.audit("update", speaker_id, {"renamed": name is not None})

    def delete_speaker(self, speaker_id: str) -> None:
        """硬删除（级联清 templates + incremental_vacuum 物理回收——被遗忘权）。"""
        with self._lock:
            try:
                cur = self._conn.execute(
                    "DELETE FROM speakers WHERE id=?", (speaker_id,))
                self._conn.commit()
                if cur.rowcount == 0:
                    raise SpeakerNotFoundError(f"说话人不存在: {speaker_id}")
                self._conn.execute("PRAGMA incremental_vacuum")
            except sqlite3.Error as e:
                self._conn.rollback()
                raise SpeakerStoreError(f"说话人删除失败: {e}") from e
            try:
                self._reload_cache()
            except SpeakerStoreError as e:
                # DELETE 已落库：重载失败不能留下幻影命中（被遗忘权），
                # 降级为手术摘除内存项；后续任意写操作的全量重载会自然纠偏
                self._evict_from_cache(speaker_id)
                logger.warning(f"删除后质心缓存重载失败，已内存摘除 {speaker_id}: {e}")
        self.audit("delete", speaker_id)

    def audit(self, action: str, speaker_id: str | None = None,
              detail: dict | None = None) -> None:
        """审计落库。★ 审计是旁路：写失败仅 WARN，不上抛、不阻断业务。"""
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO audit_log(ts, action, speaker_id, detail)"
                    " VALUES(?,?,?,?)",
                    (self._now(), action, speaker_id,
                     json.dumps(detail, ensure_ascii=False) if detail else None),
                )
                self._conn.commit()
        except sqlite3.Error as e:
            logger.warning(f"声纹库审计写入失败（业务不受影响）: {e}")

    # ─── 读路径 ───

    def list_speakers(self) -> list[dict]:
        """全部说话人摘要（不含 embedding 本体）。"""
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT s.id, s.name, s.note, s.source, s.created_at, s.updated_at,"
                    " COUNT(t.id) AS template_count"
                    " FROM speakers s LEFT JOIN templates t ON t.speaker_id = s.id"
                    " GROUP BY s.id ORDER BY s.created_at DESC"
                ).fetchall()
            except sqlite3.Error as e:
                raise SpeakerStoreError(f"说话人列表读取失败: {e}") from e
        return [dict(r) for r in rows]

    def get_speaker(self, speaker_id: str) -> dict | None:
        """单人详情（含模板摘要，不含向量本体）；未命中返回 None。"""
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT id, name, note, source, model_tag, created_at, updated_at"
                    " FROM speakers WHERE id=?", (speaker_id,)).fetchone()
                if row is None:
                    return None
                tpl = self._conn.execute(
                    "SELECT id, dur_sec, created_at FROM templates WHERE speaker_id=?"
                    " ORDER BY id", (speaker_id,)).fetchall()
            except sqlite3.Error as e:
                raise SpeakerStoreError(f"说话人详情读取失败: {e}") from e
        info = dict(row)
        info["templates"] = [dict(t) for t in tpl]
        return info

    # ─── 识别（纯内存，不触库；无锁读一致性靠"重载即整体替换引用"）───

    def identify(self, emb: np.ndarray, threshold: float = 0.45,
                 margin: float = 0.10, *,
                 include_source: bool = False) -> dict | None:
        """1:N 开集识别：top1 < threshold 或 top1-top2 < margin → None（unknown）。

        emb 须 L2 归一 [192]；threshold/margin 由调用方（Service）从 cfg 传入。
        include_source 供管理端识别接口区分 manual/auto；转写联动保持原响应。
        """
        matrix, ids, names, sources = self._cache  # 单属性读：快照原子，无撕裂
        if matrix.shape[0] == 0:
            return None
        scores = matrix @ np.asarray(emb, dtype=np.float32).reshape(-1)
        order = np.argsort(scores)
        top1 = float(scores[order[-1]])
        if top1 < threshold:
            return None
        # 库内仅 1 人时无第二名可比，margin 无定义——单靠 threshold 门控（有意设计）
        if matrix.shape[0] > 1:
            top2 = float(scores[order[-2]])
            if top1 - top2 < margin:
                return None             # 近邻打架：开集场景宁缺勿错
        idx = int(order[-1])
        result = {"speaker_id": ids[idx], "name": names[idx], "score": top1}
        if include_source:
            result["source"] = sources[idx]
        return result

    # ─── 启动检查 / 收尾 ───

    def check_model_tag(self, engine_tag: str) -> bool:
        """库内 meta.model_tag 与引擎 tag 一致性（失配 → enroll/identify 禁用，V4 语义）。"""
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT value FROM meta WHERE key='model_tag'").fetchone()
            except sqlite3.Error as e:
                raise SpeakerStoreError(f"model_tag 读取失败: {e}") from e
        return row is not None and row["value"] == engine_tag

    @property
    def cache_version(self) -> int:
        return self._cache_version

    @property
    def speaker_count(self) -> int:
        return len(self._cache[1])

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
