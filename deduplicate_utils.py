# deduplicate_utils.py
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict



# -------------------------------------------------------
# Step 1. Build lookup from database
# -------------------------------------------------------
def build_id_lookup(database: list[dict]) -> dict[int, dict]:
    """
    Build a mapping: faiss_id -> metadata for the corresponding clip or image.
    This allows us to translate from FAISS search results back to file/clip info.
    """
    lookup = {}
    for entry in database:
        media_type = entry.get("media_type")

        if media_type == "video":
            for clip in entry.get("clips", []):
                fid = clip.get("faiss_id")
                if fid is not None:
                    lookup[fid] = {
                        "file_id": entry["file_id"],
                        "file_name": entry["file_name"],
                        "clip_idx": clip.get("clip_idx"),
                        "faiss_id": fid,
                        "media_type": "video",
                        "prefix": entry.get("prefix", ""),
                        "duration": entry.get("duration", 0.0),
                        "variance": entry.get("quality", {}).get("variance", 0.0),
                        "start_s": clip.get("start_s"),
                        "end_s": clip.get("end_s"),
                    }

        elif media_type == "image":
            fid = entry.get("faiss_id")
            if fid is not None:
                lookup[fid] = {
                    "file_id": entry["file_id"],
                    "file_name": entry["file_name"],
                    "clip_idx": None,
                    "faiss_id": fid,
                    "media_type": "image",
                    "prefix": entry.get("prefix", ""),
                    "duration": 0.0,
                    "variance": entry.get("metadata", {})
                                       .get("quality", {})
                                       .get("variance", 0.0),
                    "start_s": None,
                    "end_s": None,
                }

    return lookup


# -------------------------------------------------------
# Step 2. Group embeddings with FAISS range search
# -------------------------------------------------------
def collect_groups(index, embeddings: np.ndarray, radius: float) -> list[list[int]]:
    """
    Use FAISS range search to find neighbors within the given radius.
    Build connected components (groups) of faiss_ids that are near each other.
    """
    faiss.normalize_L2(embeddings)
    lims, dists, labels = index.range_search(embeddings, radius)

    neighbors = defaultdict(set)

    # Build adjacency list
    for qi in range(len(embeddings)):
        start, end = lims[qi], lims[qi + 1]
        if start == end:
            continue
        qid = int(labels[start])
        for j in labels[start:end]:
            jid = int(j)
            if jid == -1 or jid == qid:
                continue
            neighbors[qid].add(jid)
            neighbors[jid].add(qid)

    # DFS to extract connected components
    visited = set()
    groups = []
    for node in neighbors:
        if node in visited:
            continue
        stack, comp = [node], []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            stack.extend(neighbors[cur] - visited)
        groups.append(comp)

    return groups


# -------------------------------------------------------
# Step 3. Write results as file-pair JSONL
# -------------------------------------------------------
def write_dedup_jsonl(groups: list[list[int]], lookup: dict[int, dict], output_path: Path):
    """
    Each JSONL line = one (original_file, duplicate_file) pair.
    All duplicate clip pairs between them are merged into one record.
    """

    pair_map = defaultdict(list)

    for group in groups:
        if len(group) <= 1:
            continue

        entries = [lookup[i] for i in group if i in lookup]
        if len(entries) <= 1:
            continue

        by_file = defaultdict(list)
        for e in entries:
            by_file[e["file_id"]].append(e)

        file_ids = list(by_file.keys())
        if len(file_ids) <= 1:
            continue

        for i in range(len(file_ids)):
            for j in range(i + 1, len(file_ids)):
                fid_a, fid_b = file_ids[i], file_ids[j]
                clips_a, clips_b = by_file[fid_a], by_file[fid_b]

                for ca in clips_a:
                    for cb in clips_b:
                        pair_map[(fid_a, fid_b)].append({
                            "original": {
                                "file_id": ca["file_id"],
                                "file_name": ca["file_name"],
                                "media_type": ca["media_type"],
                                "clip_idx": ca.get("clip_idx"),
                                "start_s": ca.get("start_s"),
                                "end_s": ca.get("end_s"),
                            },
                            "duplicate": {
                                "file_id": cb["file_id"],
                                "file_name": cb["file_name"],
                                "media_type": cb["media_type"],
                                "clip_idx": cb.get("clip_idx"),
                                "start_s": cb.get("start_s"),
                                "end_s": cb.get("end_s"),
                            }
                        })

    with open(output_path, "w", encoding="utf-8") as f:
        for (fid_a, fid_b), clip_pairs in pair_map.items():
            # grab one representative from each side
            first_orig = clip_pairs[0]["original"]
            first_dup = clip_pairs[0]["duplicate"]

            # look up durations from original lookup (per file)
            orig_durations = [e["duration"] for e in lookup.values() if e["file_id"] == fid_a]
            dup_durations  = [e["duration"] for e in lookup.values() if e["file_id"] == fid_b]

            record = {
                "original_file_id": fid_a,
                "original_file_name": first_orig["file_name"],
                "original_duration": orig_durations[0] if orig_durations else 0.0,
                "duplicate_file_id": fid_b,
                "duplicate_file_name": first_dup["file_name"],
                "duplicate_duration": dup_durations[0] if dup_durations else 0.0,
                "clips": clip_pairs,
            }
            f.write(json.dumps(record) + "\n")

######### api helpers 

import json
from pathlib import Path
from typing import Dict, List

def update_duplicate_flags(
    descriptions_path: Path,
    updates: Dict[int, bool]
) -> List[dict]:
    """
    Update 'duplicate' flags at top-level entries by file_id,
    persist back to the JSONL file, and return the updated list.

    Args:
        descriptions_path: path to descriptions.jsonl
        updates: mapping {file_id: duplicate_value}

    Returns:
        Updated list of description entries
    """
    # load jsonl
    entries: List[dict] = []
    with open(descriptions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # update
    for entry in entries:
        fid = entry.get("file_id")
        if fid in updates:
            entry["duplicate"] = updates[fid]

    # write back
    with open(descriptions_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entries