"""Text / number helpers shared by all matchers."""
import bisect
import itertools
import re
import unicodedata
from collections import defaultdict


def fold(s):
    return unicodedata.normalize("NFKD", str(s).upper()).encode("ascii", "ignore").decode()


def nz(s):
    """Normalised key: uppercase ASCII alnum only."""
    return re.sub(r"[^A-Z0-9]", "", fold(s))


def cents(x):
    return int(round(float(x) * 100))


def fmt(c):
    return f"{c / 100:.2f}"


NUM = re.compile(r"-?\d{1,3}(?:\.\d{3})*,\d{2}|-?\d+\.\d{2}|-?\d{1,3}(?:,\d{3})*\.\d{2}|-?\d+,\d{2}")


def pnum(s):
    """Parse 1.234,56 / 1,234.56 / 1234.56 / 1234,56."""
    if "," in s and s.rfind(",") > s.rfind("."):
        return float(s.replace(".", "").replace(",", "."))
    return float(s.replace(",", ""))


class PrefixIndex:
    """Truncation-aware party lookup: longest common prefix of (bank text from any word start) against the
    sorted normalised party names of the entity. Recall ~98.7 % on AR lines (agent 2)."""

    def __init__(self, rows):
        self.ent = {}
        tmp = defaultdict(lambda: defaultdict(set))
        for e, p, n in rows:
            k = nz(n)
            if len(k) >= 4:
                tmp[e][k].add(p)
        for e, d in tmp.items():
            self.ent[e] = (sorted(d), d)

    def query(self, e, text, minlcp=8, return_len=False):
        if e not in self.ent:
            return (set(), 0) if return_len else set()
        keys, d = self.ent[e]
        words = re.findall(r"[A-Z0-9]+", fold(text))
        best, bp = 0, None
        for i in range(len(words)):
            s = "".join(words[i:i + 8])[:48]
            if len(s) < 4:
                continue
            j = bisect.bisect_left(keys, s)
            for k in (j - 1, j):
                if 0 <= k < len(keys):
                    a = keys[k]
                    l = 0
                    m = min(len(a), len(s))
                    while l < m and a[l] == s[l]:
                        l += 1
                    if l > best:
                        best, bp = l, s[:l]
        out = set()
        if best >= minlcp:
            j = bisect.bisect_left(keys, bp)
            while j < len(keys) and keys[j].startswith(bp):
                if len(keys[j]) == best or best >= 12 or best >= 0.6 * len(keys[j]):
                    out |= d[keys[j]]
                j += 1
        return (out, best) if return_len else out


def mitm(vals, target, cap=2, maxn=22):
    """Meet-in-the-middle subset sum on integer cents. Returns (#solutions found up to cap, first solution idx list)."""
    n = len(vals)
    if n == 0 or n > maxn:
        return 0, None
    h = n // 2
    A, B = vals[:h], vals[h:]
    left = defaultdict(list)
    for r in range(len(A) + 1):
        for comb in itertools.combinations(range(len(A)), r):
            left[sum(A[i] for i in comb)].append(comb)
    cnt, sol = 0, None
    for r in range(len(B) + 1):
        for comb in itertools.combinations(range(len(B)), r):
            for lc in left.get(target - sum(B[i] for i in comb), ()):
                if not lc and not comb:
                    continue
                cnt += 1
                if sol is None:
                    sol = list(lc) + [h + i for i in comb]
                if cnt >= cap:
                    return cnt, sol
    return cnt, sol


def mitm_all(vals, target, cap=64, maxn=22):
    """All subsets (up to cap) of vals summing to target (meet in the middle). Returns list of index lists."""
    n = len(vals)
    if n == 0 or n > maxn:
        return []
    h = n // 2
    A, B = vals[:h], vals[h:]
    left = defaultdict(list)
    for r in range(len(A) + 1):
        for comb in itertools.combinations(range(len(A)), r):
            left[sum(A[i] for i in comb)].append(comb)
    out = []
    for r in range(len(B) + 1):
        for comb in itertools.combinations(range(len(B)), r):
            for lc in left.get(target - sum(B[i] for i in comb), ()):
                if not lc and not comb:
                    continue
                out.append(list(lc) + [h + i for i in comb])
                if len(out) >= cap:
                    return out
    return out
