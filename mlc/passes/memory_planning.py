"""Memory planning: pack every intermediate into one arena.

With static shapes and no autograd, the whole allocation pattern is known at
compile time. Each intermediate buffer is written by exactly one kernel and
read by a known set of later ones, so its live range is an interval on the
schedule. Two buffers whose intervals do not overlap can share memory.

That is interval-graph colouring. Unlike general graph colouring it is not
NP-hard to colour optimally, but *packing* is: we need contiguous byte ranges,
not abstract colours, which makes it a 2-D strip packing problem (an interval
on the time axis, a byte extent on the space axis). The standard heuristic --
place buffers largest first, at the lowest offset that clears every
conflicting buffer already placed -- gets within a few percent of the lower
bound in practice, and the lower bound is computed here too so the gap is
visible rather than assumed.

The payoff is not only peak memory. A fixed arena means every pointer a
kernel sees is the same on every call, which is the precondition for
capturing the schedule as a CUDA graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..config import Config
from ..ir.graph import Graph, Value
from ..ir.types import Buffer
from ..kernels import Schedule

#: Arena offsets are aligned so every buffer starts on a boundary the memory
#: system likes. 256 bytes covers vectorised loads and cuBLAS's expectations.
ALIGNMENT = 256


def _align(n: int, to: int = ALIGNMENT) -> int:
    return (n + to - 1) // to * to


@dataclass
class LiveRange:
    buffer: Buffer
    #: index of the kernel that writes it
    start: int
    #: index of the last kernel that reads it
    end: int
    size: int

    def overlaps(self, other: "LiveRange") -> bool:
        return not (self.end < other.start or other.end < self.start)


@dataclass
class AllocationPlan:
    """How the runtime should get memory for each buffer."""

    mode: str  # "arena" | "eager"
    arena_bytes: int = 0
    #: buffer name -> byte offset into the arena
    offsets: dict[str, int] = field(default_factory=dict)
    #: kernel index -> buffers to allocate just before it runs (eager mode)
    alloc_at: dict[int, list[Buffer]] = field(default_factory=dict)
    #: kernel index -> buffer names droppable just after it runs (eager mode)
    free_at: dict[int, list[str]] = field(default_factory=dict)
    live_ranges: dict[str, LiveRange] = field(default_factory=dict)
    #: the most bytes live at any one instant: the best any planner could do
    peak_bytes: int = 0
    #: what allocating everything up front would have cost
    total_bytes: int = 0

    @property
    def efficiency(self) -> float:
        """Arena size over the lower bound. 1.0 is optimal."""
        return self.arena_bytes / self.peak_bytes if self.peak_bytes else 1.0

    def summary(self) -> str:
        if self.mode != "arena":
            return (f"memory: unplanned, peak {self.peak_bytes / 1024:.1f} KiB "
                    f"of {self.total_bytes / 1024:.1f} KiB total")
        return (f"memory: arena {self.arena_bytes / 1024:.1f} KiB, "
                f"lower bound {self.peak_bytes / 1024:.1f} KiB "
                f"({self.efficiency:.2f}x), "
                f"unplanned would be {self.total_bytes / 1024:.1f} KiB")


# --------------------------------------------------------------------------
# Live ranges
# --------------------------------------------------------------------------

def compute_live_ranges(graph: Graph, schedule: Schedule) -> list[LiveRange]:
    """Live range of every plannable buffer, as kernel indices.

    Ranges are computed per *buffer*, not per value: several values can be
    views of one buffer, and the buffer stays live until the last of them is
    read. Getting this wrong at the value level would let the planner reuse
    memory that a later view still points into.
    """
    first_write: dict[str, int] = {}
    last_read: dict[str, int] = {}
    buffers: dict[str, Buffer] = {}

    for i, k in enumerate(schedule.kernels):
        for v in k.writes():
            buffers.setdefault(v.buffer.name, v.buffer)
            first_write.setdefault(v.buffer.name, i)
            last_read[v.buffer.name] = max(last_read.get(v.buffer.name, i), i)
        for v in k.reads():
            buffers.setdefault(v.buffer.name, v.buffer)
            last_read[v.buffer.name] = max(last_read.get(v.buffer.name, i), i)

    # Graph outputs stay live past the end of the schedule.
    end_of_time = len(schedule.kernels)
    for v in graph.outputs:
        last_read[v.buffer.name] = end_of_time
        buffers.setdefault(v.buffer.name, v.buffer)

    ranges: list[LiveRange] = []
    for name, buf in buffers.items():
        if not buf.plannable:
            continue
        start = first_write.get(name)
        if start is None:
            # Read but never written: an input or param, already excluded, or
            # a bug elsewhere. Skip rather than place it.
            continue
        ranges.append(LiveRange(buf, start, last_read.get(name, start), _align(buf.nbytes)))
    return ranges


def peak_live_bytes(ranges: list[LiveRange], n_kernels: int) -> int:
    """The most bytes simultaneously live: a lower bound on any arena."""
    if not ranges:
        return 0
    events: list[tuple[int, int]] = []
    for r in ranges:
        events.append((r.start, r.size))
        events.append((r.end + 1, -r.size))
    events.sort()
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------

def pack(ranges: list[LiveRange]) -> tuple[dict[str, int], int]:
    """Assign each range a byte offset. Greedy, largest first, lowest fit.

    Processing large buffers first matters: they are the ones that determine
    the arena size, and placing them while the arena is still empty lets the
    small ones fill the gaps around them afterwards.
    """
    placed: list[tuple[int, LiveRange]] = []  # (offset, range)
    offsets: dict[str, int] = {}
    total = 0

    for r in sorted(ranges, key=lambda r: (-r.size, r.start, r.buffer.name)):
        conflicts = sorted(
            ((off, p) for off, p in placed if p.overlaps(r)), key=lambda t: t[0]
        )
        offset = 0
        for off, p in conflicts:
            if off >= offset + r.size:
                break  # the gap below this buffer is wide enough
            offset = max(offset, _align(off + p.size))
        offsets[r.buffer.name] = offset
        placed.append((offset, r))
        total = max(total, offset + r.size)

    return offsets, total


def verify(ranges: list[LiveRange], offsets: dict[str, int]) -> list[str]:
    """Two buffers alive at the same time must never share a byte.

    This is the one property whose violation is a silent miscompile rather
    than a crash, so it is checked directly rather than inferred from the
    algorithm being correct.
    """
    problems: list[str] = []
    for i, a in enumerate(ranges):
        oa = offsets[a.buffer.name]
        for b in ranges[i + 1 :]:
            if not a.overlaps(b):
                continue
            ob = offsets[b.buffer.name]
            if oa < ob + b.size and ob < oa + a.size:
                problems.append(
                    f"{a.buffer.name}[{oa}:{oa + a.size}] live [{a.start},{a.end}] "
                    f"overlaps {b.buffer.name}[{ob}:{ob + b.size}] live [{b.start},{b.end}]"
                )
    return problems


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def plan_memory(graph: Graph, schedule: Schedule, cfg: Config) -> AllocationPlan:
    ranges = compute_live_ranges(graph, schedule)
    peak = peak_live_bytes(ranges, len(schedule.kernels))
    total = sum(r.size for r in ranges)

    if not cfg.memory_planning:
        plan = AllocationPlan(mode="eager", peak_bytes=peak, total_bytes=total)
        plan.live_ranges = {r.buffer.name: r for r in ranges}
        for r in ranges:
            plan.alloc_at.setdefault(r.start, []).append(r.buffer)
            if r.end < len(schedule.kernels):
                plan.free_at.setdefault(r.end, []).append(r.buffer.name)
        for b in graph.buffers():
            b.arena_offset = None
        schedule.arena_bytes = 0
        schedule.plan = plan
        return plan

    offsets, arena_bytes = pack(ranges)
    problems = verify(ranges, offsets)
    if problems:
        raise AssertionError("memory planner produced overlapping buffers:\n" + "\n".join(problems))

    for b in graph.buffers():
        b.arena_offset = offsets.get(b.name)
    plan = AllocationPlan(
        mode="arena",
        arena_bytes=arena_bytes,
        offsets=offsets,
        live_ranges={r.buffer.name: r for r in ranges},
        peak_bytes=peak,
        total_bytes=total,
    )
    schedule.arena_bytes = arena_bytes
    schedule.plan = plan
    return plan
