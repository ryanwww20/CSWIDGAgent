"""Layer 0 actuator: drive ipywidgets *in the kernel* (no browser).

Setting `widget.value` in the kernel fires the same observe/callback chain a real
drag would, so we drive the genuine interaction logic headlessly and read the
resulting output from the notebook's Output widgets. Authoritative control list
comes from the live `ipywidgets.Widget.widgets` registry — independent of how the
author named variables, so this works on un-instrumented / wild notebooks.

Output capture targets ipywidgets `Output` widgets (where `interact`/callbacks
render). matplotlib figures land there as image/png under `%matplotlib inline`.
"""
from __future__ import annotations

import json
from pathlib import Path
from queue import Empty
from typing import Any

import nbformat

from actuator import Actuator
from schemas import WidgetSpec, widget_kind

# Setting matplotlib_close(False) keeps figures alive in pyplot's figure
# manager after a callback's plt.show(), so we can savefig the real rendered
# plot headlessly. Run before the notebook so the initial render survives.
_MPL_SETUP = (
    "try:\n"
    "    from matplotlib_inline.backend_inline import set_matplotlib_close as _smc\n"
    "    _smc(False)\n"
    "except Exception:\n"
    "    pass\n"
)

# Driver helpers injected into the demo's kernel. They print marker-prefixed
# JSON on stdout, which we parse from the iopub stream.
#
# Output capture: ipywidgets `Output.outputs` does NOT populate in a kernel with
# no frontend, so we capture matplotlib figures straight from pyplot's figure
# manager (robust + frontend-free) and fall back to Output text where present.
_KERNEL_PREAMBLE = r'''
import json as __json, base64 as __b64, hashlib as __hashlib, os as __os
import ipywidgets as __ipw
try:
    import matplotlib.pyplot as __plt
    from matplotlib_inline.backend_inline import set_matplotlib_close as __smc
    __smc(False)
except Exception:
    __plt = None

def __ipw_enum():
    specs = []
    for mid, w in list(__ipw.Widget.widgets.items()):
        t = type(w).__name__
        if t == "Output":
            continue
        spec = {"id": mid, "type": t,
                "name": (getattr(w, "description", "") or "").strip(),
                "value": None, "min": None, "max": None, "step": None,
                "options": None}
        if hasattr(w, "value"):
            try:
                __json.dumps(w.value); spec["value"] = w.value
            except Exception:
                spec["value"] = repr(w.value)
        for a in ("min", "max", "step"):
            if hasattr(w, a):
                try:
                    __json.dumps(getattr(w, a)); spec[a] = getattr(w, a)
                except Exception:
                    pass
        labels = getattr(w, "_options_labels", None)
        if labels is not None:
            spec["options"] = [str(x) for x in labels]
        elif hasattr(w, "options"):
            try:
                spec["options"] = [str(x) for x in list(w.options)]
            except Exception:
                pass
        specs.append(spec)
    print("__IPW_ENUM__" + __json.dumps(specs))

def __ipw_set(mid, value):
    ok = False
    try:
        __ipw.Widget.widgets[mid].value = value; ok = True
    except Exception as e:
        print("__IPW_ERR__" + repr(e))
    print("__IPW_SET__" + __json.dumps(ok))

def __ipw_set_index(mid, i):
    ok = False
    try:
        __ipw.Widget.widgets[mid].index = i; ok = True
    except Exception as e:
        print("__IPW_ERR__" + repr(e))
    print("__IPW_SET__" + __json.dumps(ok))

def __ipw_click(mid):
    ok = False
    try:
        __ipw.Widget.widgets[mid].click(); ok = True
    except Exception as e:
        print("__IPW_ERR__" + repr(e))
    print("__IPW_SET__" + __json.dumps(ok))

def __ipw_snapshot(outdir, tag):
    __os.makedirs(outdir, exist_ok=True)
    h = __hashlib.sha256(); items = []; idx = 0
    # 1) Primary: the current matplotlib figure (latest rendered by a callback).
    #    We do NOT clear figures before driving — a no-op set produces no new
    #    figure, so the latest figure is unchanged and hashes the same (a true
    #    "no change"). Older figures are closed after capture to bound memory.
    if __plt is not None:
        figs = __plt.get_fignums()
        if figs:
            try:
                fig = __plt.figure(figs[-1])
                p = __os.path.join(outdir, tag + "_" + str(idx) + ".png")
                fig.savefig(p, dpi=80, bbox_inches="tight")
                with open(p, "rb") as f:
                    raw = f.read()
                h.update(raw)
                items.append({"type": "image", "path": p}); idx += 1
            except Exception as e:
                print("__IPW_ERR__" + repr(e))
            for n in figs[:-1]:
                try:
                    __plt.close(n)
                except Exception:
                    pass
    # 2) Fallback: any text/stream an Output widget did capture.
    for mid, w in list(__ipw.Widget.widgets.items()):
        if type(w).__name__ != "Output":
            continue
        for out in list(getattr(w, "outputs", []) or []):
            ot = out.get("output_type")
            if ot in ("display_data", "execute_result"):
                data = out.get("data", {}) or {}
                if "image/png" in data:
                    raw = __b64.b64decode(data["image/png"]); h.update(raw)
                    p = __os.path.join(outdir, tag + "_o" + str(idx) + ".png")
                    with open(p, "wb") as f:
                        f.write(raw)
                    items.append({"type": "image", "path": p}); idx += 1
                elif "text/plain" in data:
                    txt = data["text/plain"]; h.update(txt.encode("utf-8"))
                    items.append({"type": "text", "text": txt[:300]})
            elif ot == "stream":
                txt = out.get("text", ""); h.update(txt.encode("utf-8"))
                items.append({"type": "stream", "text": txt[:300]})
    print("__IPW_SNAP__" + __json.dumps({"hash": h.hexdigest(), "items": items}))
'''


class KernelActuator(Actuator):
    def __init__(self, notebook_path: str, workdir: str,
                 cell_timeout: float = 120.0) -> None:
        self.notebook_path = Path(notebook_path)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cell_timeout = cell_timeout
        self.km = None
        self.kc = None
        self._specs: list[WidgetSpec] = []
        self._id2spec: dict[str, WidgetSpec] = {}
        self.cell_errors: list[str] = []
        # Per-action logs, refreshed by set_values()/click() and read by the
        # executor right after each action (text-demo output + robustness signal).
        self.last_stdout: str = ""
        self.last_stderr: str | None = None

    # --- lifecycle -------------------------------------------------------
    def start(self) -> None:
        from jupyter_client.manager import start_new_kernel
        self.km, self.kc = start_new_kernel(kernel_name="python3")
        self._exec(_MPL_SETUP, 30)  # keep initial render alive for baseline
        nb = nbformat.read(self.notebook_path, as_version=4)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code" or not cell.source.strip():
                continue
            _, _, err = self._exec(cell.source, self.cell_timeout)
            if err:
                self.cell_errors.append(f"cell{i}: {err.splitlines()[-1][:160]}")
        self._exec(_KERNEL_PREAMBLE, 30)

    def stop(self) -> None:
        try:
            if self.km is not None:
                self.km.shutdown_kernel(now=True)
        finally:
            if self.kc is not None:
                self.kc.stop_channels()

    # --- enumeration -----------------------------------------------------
    def enumerate(self) -> list[WidgetSpec]:
        out, _, _ = self._exec("__ipw_enum()", 30)
        raw = self._marker(out, "__IPW_ENUM__") or []
        specs: list[WidgetSpec] = []
        seen: dict[str, int] = {}
        for r in raw:
            kind = widget_kind(r["type"], bool(r.get("options")), r.get("value"))
            base = r["name"] or r["type"]
            n = seen.get(base, 0)
            seen[base] = n + 1
            name = base if n == 0 else f"{base}#{n}"
            drivable = kind in ("button", "selection", "numeric", "bool", "text")
            specs.append(WidgetSpec(
                id=r["id"], name=name, type=r["type"], driver="kernel", kind=kind,
                drivable=drivable, value=r.get("value"), min=r.get("min"),
                max=r.get("max"), step=r.get("step"), options=r.get("options")))
        self._specs = specs
        self._id2spec = {s.id: s for s in specs}
        self._name2spec = {s.name: s for s in specs}
        return specs

    # --- actuation -------------------------------------------------------
    def set_values(self, mapping: dict[str, Any]) -> dict[str, bool]:
        result: dict[str, bool] = {}
        self._reset_action_log()
        for name, value in mapping.items():
            spec = self._name2spec.get(name)
            if spec is None:
                result[name] = False
                continue
            if spec.kind == "selection":
                idx = self._option_index(spec, value)
                if idx is None:
                    result[name] = False
                    self._log_err(f"option {value!r} not valid for {name}")
                    continue
                code = f"__ipw_set_index({spec.id!r}, {idx!r})"
            else:
                code = f"__ipw_set({spec.id!r}, {value!r})"
            out, serr, err = self._exec(code, 60)
            result[name] = bool(self._marker(out, "__IPW_SET__"))
            self._log_action(out, serr, err)
        return result

    def click(self, name: str, repeat: int = 1) -> bool:
        spec = self._name2spec.get(name)
        self._reset_action_log()
        if spec is None:
            return False
        ok = True
        for _ in range(max(1, repeat)):
            out, serr, err = self._exec(f"__ipw_click({spec.id!r})", 60)
            ok = ok and bool(self._marker(out, "__IPW_SET__"))
            self._log_action(out, serr, err)
        return ok

    def snapshot(self, tag: str) -> tuple[str, list[str]]:
        code = f"__ipw_snapshot({str(self.workdir)!r}, {tag!r})"
        out, _, _ = self._exec(code, 60)
        snap = self._marker(out, "__IPW_SNAP__") or {"hash": "", "items": []}
        images = [it["path"] for it in snap["items"] if it.get("type") == "image"]
        return snap.get("hash", ""), images

    # --- per-action log capture -------------------------------------------
    def _reset_action_log(self) -> None:
        self.last_stdout = ""
        self.last_stderr = None
        self._stdout_parts: list[str] = []
        self._stderr_parts: list[str] = []

    def _log_err(self, msg: str) -> None:
        self._stderr_parts.append(msg)
        self.last_stderr = "\n".join(self._stderr_parts).strip() or None

    def _log_action(self, out: str, serr: str, err: str | None) -> None:
        """Fold one _exec's output into the per-action log: plain stdout (the
        demo's own prints — its output for text demos) and anything that signals
        a break (stderr stream, traceback, __IPW_ERR__ markers)."""
        plain = "\n".join(line for line in out.splitlines()
                          if not line.startswith("__IPW_"))
        if plain.strip():
            self._stdout_parts.append(plain.strip())
        for line in out.splitlines():
            if line.startswith("__IPW_ERR__"):
                self._stderr_parts.append(line[len("__IPW_ERR__"):])
        if serr and serr.strip():
            self._stderr_parts.append(serr.strip())
        if err:
            self._stderr_parts.append(err)
        self.last_stdout = "\n".join(self._stdout_parts).strip()
        self.last_stderr = "\n".join(self._stderr_parts).strip() or None

    # --- helpers ---------------------------------------------------------
    @staticmethod
    def _option_index(spec: WidgetSpec, value: Any) -> int | None:
        """Resolve a requested option to its index; None when it doesn't exist
        (an invalid plan step must FAIL visibly, not silently pick option 0)."""
        opts = spec.options or []
        sval = str(value)
        if sval in opts:
            return opts.index(sval)
        if isinstance(value, int) and 0 <= value < len(opts):
            return value
        return None

    def _exec(self, code: str, timeout: float) -> tuple[str, str, str | None]:
        """Run code in the kernel. Returns (stdout, stderr, error_traceback).

        stdout/stderr are kept separate so callback warnings (NaN/overflow land
        on the stderr stream) and tracebacks reach the robustness metric instead
        of being silently dropped.
        """
        assert self.kc is not None
        msg_id = self.kc.execute(code)
        stdout: list[str] = []
        stderr: list[str] = []
        error: str | None = None
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=timeout)
            except Empty:
                return "".join(stdout), "".join(stderr), "timeout"
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            mt, content = msg["msg_type"], msg["content"]
            if mt == "stream":
                if content.get("name") == "stderr":
                    stderr.append(content.get("text", ""))
                else:
                    stdout.append(content.get("text", ""))
            elif mt == "error":
                error = "\n".join(content.get("traceback", []))
            elif mt == "status" and content.get("execution_state") == "idle":
                break
        return "".join(stdout), "".join(stderr), error

    @staticmethod
    def _marker(stdout: str, marker: str) -> Any:
        for line in stdout.splitlines():
            if line.startswith(marker):
                try:
                    return json.loads(line[len(marker):])
                except json.JSONDecodeError:
                    return None
        return None
