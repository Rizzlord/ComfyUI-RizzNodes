import { app } from "../../scripts/app.js";

const VIDEO_TRACKS = 3;
const AUDIO_TRACKS = 3;
const TOTAL_TRACKS = VIDEO_TRACKS + AUDIO_TRACKS;

const EDITOR_MIN_W = 680;
const EDITOR_MIN_H = 430;
const CLIP_MIN_SEC = 0.15;
const DEFAULT_CLIP_SEC = 3.0;
const MIN_TIMELINE_SEC = 6.0;
const SNAP_THRESHOLD_PX = 10;
const HIDDEN_WIDGET = "hidden";
const TRIM_HANDLE_W = 5;
const TRIM_HANDLE_PAD = 2;
const AUDIO_VOLUME_MAX = 2.0;
const AUDIO_VOL_HIT_PAD = 6;
const AUDIO_FADE_HANDLE_R = 5;
const TRANSITION_TYPES = [
    "Fade",
    "Smooth",
    "Dissolve",
    "Dip Black",
    "Dip White",
    "Slide Left",
    "Slide Right",
    "Wipe Left",
    "Wipe Right",
    "Zoom In",
    "Zoom Out",
];
const TRANSITION_TARGETS = ["In+Out", "In Only", "Out Only", "Clear"];

const COLORS = {
    panelBg: "rgba(20,22,28,0.96)",
    panelStroke: "rgba(112,130,160,0.45)",
    toolbarBg: "rgba(12,14,18,0.95)",
    rulerBg: "rgba(8,9,12,0.96)",
    trackVideo: "rgba(52,96,156,0.18)",
    trackAudio: "rgba(40,128,78,0.18)",
    trackBorder: "rgba(220,230,255,0.10)",
    textMuted: "rgba(225,235,255,0.62)",
    textStrong: "rgba(240,245,255,0.95)",
    playhead: "#ffad42",
    clipVideo: "rgba(82,167,255,0.88)",
    clipAudio: "rgba(95,219,149,0.85)",
    clipSel: "rgba(255,210,95,0.95)",
    buttonBg: "rgba(70,82,108,0.92)",
    buttonBgDanger: "rgba(142,62,62,0.95)",
    buttonText: "#f4f6ff",
};

const BUTTONS = [
    { id: "add_v1", label: "+V1", width: 48 },
    { id: "add_v2", label: "+V2", width: 48 },
    { id: "add_v3", label: "+V3", width: 48 },
    { id: "add_a1", label: "+A1", width: 48 },
    { id: "add_a2", label: "+A2", width: 48 },
    { id: "add_a3", label: "+A3", width: 48 },
    { id: "len_dec", label: "-5s", width: 52 },
    { id: "len_inc", label: "+5s", width: 52 },
    { id: "len_fit", label: "Fit", width: 48 },
    { id: "cut", label: "Cut", width: 60 },
    { id: "delete", label: "Delete", width: 70, danger: true },
];

function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

function toNum(v, d = 0) {
    const n = Number(v);
    return Number.isFinite(n) ? n : d;
}

function makeId(prefix) {
    return `${prefix}_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
}

function formatTime(sec) {
    sec = Math.max(0, toNum(sec, 0));
    const mins = Math.floor(sec / 60);
    const s = sec - mins * 60;
    if (mins > 0) {
        return `${mins}:${s.toFixed(1).padStart(4, "0")}`;
    }
    return `${s.toFixed(1)}s`;
}

function roundRect(ctx, x, y, w, h, r = 6) {
    if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, r);
        return;
    }
    ctx.beginPath();
    ctx.rect(x, y, w, h);
}

function isNodeSelected(node) {
    const selected = app.canvas?.selected_nodes;
    return !!(selected && selected[node.id]);
}

app.registerExtension({
    name: "rizznodes.timeline_editor",

    nodeCreated(node) {
        if (node.comfyClass !== "RizzTimelineEditor") return;

        const getWidget = (name) => {
            if (!Array.isArray(node.widgets)) return null;
            return node.widgets.find((w) => w?.name === name) || null;
        };

        const normalizeTransition = (raw) => {
            if (!raw || typeof raw !== "object") return null;
            const type = `${raw.type || "None"}`.trim();
            const duration = Math.max(0, toNum(raw.duration, 0));
            if (type === "None" || duration <= 0) return null;
            if (!TRANSITION_TYPES.includes(type)) return null;
            return { type, duration };
        };

        const state = {
            timelineLength: 10.0,
            playhead: 0.0,
            videoClips: [],
            audioClips: [],
        };

        const uiCache = {
            clipRects: [],
            buttons: [],
        };

        const selected = {
            media: null,
            id: null,
        };

        let drag = null;
        let hoverButtonId = null;
        let lastConnectionSig = "";
        let snapGuideTime = null;

        const hideTimelineWidget = () => {
            const w = getWidget("timeline_json");
            if (!w) return;
            w.hidden = true;
            w.type = HIDDEN_WIDGET;
            w.computeSize = () => [0, -4];
            w.draw = () => {};
            if (w.inputEl) w.inputEl.style.display = "none";
        };

        const getVisibleWidgets = () => {
            if (!node.widgets) return [];
            return node.widgets.filter((w) => !w.hidden && w.type !== HIDDEN_WIDGET);
        };

        const getWidgetBottom = () => {
            const widgets = getVisibleWidgets();
            const nodeW = Number.isFinite(node.size?.[0]) ? node.size[0] : 480;
            let maxBottom = 0;

            for (const w of widgets) {
                let h = 20;
                if (Number.isFinite(w.computedHeight)) h = w.computedHeight;
                else if (Number.isFinite(w.height)) h = w.height;
                else if (typeof w.computeSize === "function") {
                    const sz = w.computeSize(nodeW);
                    if (Array.isArray(sz) && Number.isFinite(sz[1])) h = sz[1];
                }
                const y = Number.isFinite(w.last_y) ? w.last_y : (Number.isFinite(w.y) ? w.y : 0);
                const b = y + h;
                if (Number.isFinite(b)) maxBottom = Math.max(maxBottom, b);
            }

            if (maxBottom < 40 && widgets.length > 0) {
                maxBottom = 35 + widgets.length * 24;
            }
            return maxBottom;
        };

        const ensureNodeSize = () => {
            const wBottom = getWidgetBottom();
            const minW = Math.max(EDITOR_MIN_W, node.size[0]);
            const minH = Math.max(node.size[1], wBottom + EDITOR_MIN_H);
            if (node.size[0] !== minW || node.size[1] !== minH) {
                node.setSize([minW, minH]);
            }
        };

        const normalizeClip = (raw, media, idx) => {
            if (!raw || typeof raw !== "object") return null;
            const src = clamp(Math.round(toNum(raw.src, 1)), 1, 3);
            const track = clamp(Math.round(toNum(raw.track, 0)), 0, 2);
            const start = Math.max(0, toNum(raw.start, 0));
            const trimIn = Math.max(0, toNum(raw.in, 0));
            const dur = Math.max(CLIP_MIN_SEC, toNum(raw.dur, DEFAULT_CLIP_SEC));
            const vol = Math.max(0, toNum(raw.volume, 1));
            return {
                id: `${raw.id || `${media}_${idx + 1}`}`,
                src,
                track,
                start,
                in: trimIn,
                dur,
                volume: vol,
                transition_in: normalizeTransition(raw.transition_in),
                transition_out: normalizeTransition(raw.transition_out),
                enabled: true,
            };
        };

        const loadFromWidget = () => {
            const w = getWidget("timeline_json");
            if (!w) return;

            let parsed = null;
            try {
                parsed = JSON.parse(w.value || "{}");
            } catch (e) {
                parsed = null;
            }

            if (!parsed || typeof parsed !== "object") {
                parsed = {
                    timeline_length: 10.0,
                    playhead: 0.0,
                    video_clips: [],
                    audio_clips: [],
                };
            }

            state.timelineLength = Math.max(MIN_TIMELINE_SEC, toNum(parsed.timeline_length, 10.0));
            state.playhead = clamp(toNum(parsed.playhead, 0.0), 0, state.timelineLength);
            state.videoClips = Array.isArray(parsed.video_clips)
                ? parsed.video_clips.map((c, idx) => normalizeClip(c, "video", idx)).filter(Boolean)
                : [];
            state.audioClips = Array.isArray(parsed.audio_clips)
                ? parsed.audio_clips.map((c, idx) => normalizeClip(c, "audio", idx)).filter(Boolean)
                : [];
        };

        const ensureTimelineBounds = () => {
            let maxEnd = 0;
            for (const c of state.videoClips) maxEnd = Math.max(maxEnd, c.start + c.dur);
            for (const c of state.audioClips) maxEnd = Math.max(maxEnd, c.start + c.dur);
            if (maxEnd + 0.4 > state.timelineLength) {
                state.timelineLength = Math.max(MIN_TIMELINE_SEC, maxEnd + 1.0);
            }
            state.playhead = clamp(state.playhead, 0, state.timelineLength);
        };

        const saveToWidget = () => {
            ensureTimelineBounds();
            const w = getWidget("timeline_json");
            if (!w) return;
            const payload = {
                timeline_length: state.timelineLength,
                playhead: state.playhead,
                video_clips: state.videoClips.map((c) => ({
                    id: c.id,
                    src: c.src,
                    track: c.track,
                    start: c.start,
                    in: c.in,
                    dur: c.dur,
                    volume: c.volume,
                    transition_in: c.transition_in || null,
                    transition_out: c.transition_out || null,
                    enabled: true,
                })),
                audio_clips: state.audioClips.map((c) => ({
                    id: c.id,
                    src: c.src,
                    track: c.track,
                    start: c.start,
                    in: c.in,
                    dur: c.dur,
                    volume: c.volume,
                    transition_in: c.transition_in || null,
                    transition_out: c.transition_out || null,
                    enabled: true,
                })),
            };
            w.value = JSON.stringify(payload);
            syncTimelineLengthWidget();
            node.setDirtyCanvas(true, true);
        };

        const getTransitionType = () => {
            const w = getWidget("transition_type");
            const value = `${w?.value || "Fade"}`;
            return TRANSITION_TYPES.includes(value) ? value : "Fade";
        };

        const getTransitionTarget = () => {
            const w = getWidget("transition_target");
            const value = `${w?.value || "In+Out"}`;
            return TRANSITION_TARGETS.includes(value) ? value : "In+Out";
        };

        const getTransitionDuration = () => {
            const w = getWidget("transition_duration");
            return Math.max(0.05, toNum(w?.value, 0.25));
        };

        const getTimelineLengthFromWidget = () => {
            const w = getWidget("timeline_length_sec");
            return Math.max(MIN_TIMELINE_SEC, toNum(w?.value, state.timelineLength));
        };

        const syncTimelineLengthWidget = () => {
            const w = getWidget("timeline_length_sec");
            if (!w) return;
            const rounded = Number(state.timelineLength.toFixed(3));
            if (toNum(w.value, rounded) !== rounded) {
                w.value = rounded;
            }
        };

        const applyTimelineLengthFromWidget = () => {
            state.timelineLength = getTimelineLengthFromWidget();
            state.playhead = clamp(state.playhead, 0, state.timelineLength);
            saveToWidget();
        };

        const resolveTransitionTargetClip = () => {
            let clip = getSelectedClip();
            if (clip) return clip;

            const found = findClipAtPlayhead();
            if (found) {
                selectClip(found.media, found.clip.id);
                return found.clip;
            }

            const fallback = state.videoClips[0] ? { clip: state.videoClips[0], media: "video" }
                : (state.audioClips[0] ? { clip: state.audioClips[0], media: "audio" } : null);
            if (!fallback) return null;
            selectClip(fallback.media, fallback.clip.id);
            return fallback.clip;
        };

        const clearSelectedTransition = () => {
            const clip = resolveTransitionTargetClip();
            if (!clip) return;
            clip.transition_in = null;
            clip.transition_out = null;
            saveToWidget();
        };

        const applySelectedTransition = () => {
            const clip = resolveTransitionTargetClip();
            if (!clip) return;

            const target = getTransitionTarget();
            if (target === "Clear") {
                clearSelectedTransition();
                return;
            }

            const duration = Math.max(0.01, Math.min(getTransitionDuration(), Math.max(0.01, clip.dur * 0.49)));
            const transition = { type: getTransitionType(), duration };

            if (target === "In+Out" || target === "In Only") {
                clip.transition_in = { ...transition };
            }
            if (target === "In+Out" || target === "Out Only") {
                clip.transition_out = { ...transition };
            }
            saveToWidget();
        };

        const ensureTransitionControls = () => {
            let wType = getWidget("transition_type");
            if (!wType) {
                wType = node.addWidget("combo", "transition_type", "Fade", () => {}, { values: TRANSITION_TYPES });
            }

            let wTarget = getWidget("transition_target");
            if (!wTarget) {
                wTarget = node.addWidget("combo", "transition_target", "In+Out", () => {}, { values: TRANSITION_TARGETS });
            }

            let wDuration = getWidget("transition_duration");
            if (!wDuration) {
                wDuration = node.addWidget("number", "transition_duration", 0.25, () => {}, {
                    min: 0.05,
                    max: 3.0,
                    step: 0.05,
                    precision: 2,
                });
            }

            let wApply = getWidget("add_transition");
            if (!wApply) {
                wApply = node.addWidget("button", "add_transition", null, () => {
                    applySelectedTransition();
                });
            }

            let wClear = getWidget("clear_transition");
            if (!wClear) {
                wClear = node.addWidget("button", "clear_transition", null, () => {
                    clearSelectedTransition();
                });
            }

            let wTimelineLen = getWidget("timeline_length_sec");
            if (!wTimelineLen) {
                wTimelineLen = node.addWidget("number", "timeline_length_sec", state.timelineLength, () => {}, {
                    min: MIN_TIMELINE_SEC,
                    max: 21600,
                    step: 0.1,
                    precision: 3,
                });
            }

            let wSetTimelineLen = getWidget("set_timeline_length");
            if (!wSetTimelineLen) {
                wSetTimelineLen = node.addWidget("button", "set_timeline_length", null, () => {
                    applyTimelineLengthFromWidget();
                });
            }

            if (wType) wType.value = `${wType.value || "Fade"}`;
            if (wTarget) wTarget.value = `${wTarget.value || "In+Out"}`;
            if (wDuration) wDuration.value = Math.max(0.05, toNum(wDuration.value, 0.25));
            syncTimelineLengthWidget();
        };

        const getTrackInfoFromRow = (row) => {
            if (row >= 0 && row < VIDEO_TRACKS) {
                return { media: "video", track: row, label: `V${row + 1}` };
            }
            if (row >= VIDEO_TRACKS && row < TOTAL_TRACKS) {
                const t = row - VIDEO_TRACKS;
                return { media: "audio", track: t, label: `A${t + 1}` };
            }
            return null;
        };

        const getLayout = () => {
            const pad = 10;
            const toolbarH = 30;
            const rulerH = 22;
            const top = getWidgetBottom() + 8;
            const left = pad;
            const width = Math.max(node.size[0] - pad * 2, 280);
            const height = Math.max(node.size[1] - top - pad, 260);

            const tracksTop = top + toolbarH + 8 + rulerH + 5;
            const tracksHeight = Math.max(120, height - toolbarH - 8 - rulerH - 5);
            const trackGap = 4;
            const trackH = Math.max(18, (tracksHeight - trackGap * (TOTAL_TRACKS - 1)) / TOTAL_TRACKS);

            const labelW = 42;
            const timeLeft = left + labelW + 6;
            const timeWidth = Math.max(80, width - labelW - 12);

            return {
                left,
                top,
                width,
                height,
                toolbarH,
                rulerH,
                tracksTop,
                tracksHeight,
                trackGap,
                trackH,
                labelW,
                timeLeft,
                timeWidth,
            };
        };

        const getDuration = () => Math.max(MIN_TIMELINE_SEC, state.timelineLength);

        const timeToX = (t, layout) => {
            const dur = getDuration();
            return layout.timeLeft + (clamp(t, 0, dur) / dur) * layout.timeWidth;
        };

        const xToTime = (x, layout) => {
            const dur = getDuration();
            const rel = clamp((x - layout.timeLeft) / layout.timeWidth, 0, 1);
            return rel * dur;
        };

        const rowFromY = (y, layout) => {
            for (let row = 0; row < TOTAL_TRACKS; row++) {
                const ry = layout.tracksTop + row * (layout.trackH + layout.trackGap);
                if (y >= ry && y <= ry + layout.trackH) return row;
            }
            return -1;
        };

        const findClip = (media, id) => {
            const arr = media === "video" ? state.videoClips : state.audioClips;
            return arr.find((c) => c.id === id) || null;
        };

        const getSelectedClip = () => {
            if (!selected.media || !selected.id) return null;
            return findClip(selected.media, selected.id);
        };

        const findClipAtPlayhead = () => {
            const t = state.playhead;
            const candidates = [];
            const pushFrom = (arr, media) => {
                for (const c of arr) {
                    const end = c.start + c.dur;
                    if (t >= c.start && t <= end) {
                        candidates.push({ clip: c, media });
                    }
                }
            };
            pushFrom(state.videoClips, "video");
            pushFrom(state.audioClips, "audio");
            if (candidates.length === 0) return null;
            candidates.sort((a, b) => {
                if (a.media !== b.media) return a.media === "video" ? -1 : 1;
                if (a.clip.track !== b.clip.track) return b.clip.track - a.clip.track;
                return b.clip.start - a.clip.start;
            });
            return candidates[0];
        };

        const selectClip = (media, id) => {
            selected.media = media;
            selected.id = id;
        };

        const clearSelection = () => {
            selected.media = null;
            selected.id = null;
        };

        const getMediaList = (media) => (media === "video" ? state.videoClips : state.audioClips);

        const isInputConnected = (name) => {
            if (!Array.isArray(node.inputs)) return false;
            const input = node.inputs.find((inp) => inp?.name === name);
            if (!input) return false;
            if (Array.isArray(input.links)) return input.links.length > 0;
            return input.link !== null && input.link !== undefined;
        };

        const findBestTrack = (media, start, dur) => {
            const list = getMediaList(media);
            let bestTrack = 0;
            let bestScore = Number.POSITIVE_INFINITY;

            for (let t = 0; t < 3; t++) {
                let overlaps = 0;
                for (const c of list) {
                    if (c.track !== t) continue;
                    const cEnd = c.start + c.dur;
                    const nEnd = start + dur;
                    if (start < cEnd && nEnd > c.start) overlaps += 1;
                }
                if (overlaps < bestScore) {
                    bestScore = overlaps;
                    bestTrack = t;
                    if (overlaps === 0) break;
                }
            }
            return bestTrack;
        };

        const addClip = (media, src, forcedTrack = null) => {
            const track = forcedTrack !== null ? clamp(forcedTrack, 0, 2) : findBestTrack(media, state.playhead, DEFAULT_CLIP_SEC);
            const clip = {
                id: makeId(media === "video" ? "v" : "a"),
                src: clamp(Math.round(src), 1, 3),
                track,
                start: state.playhead,
                in: 0,
                dur: DEFAULT_CLIP_SEC,
                volume: 1.0,
                transition_in: null,
                transition_out: null,
                enabled: true,
            };
            if (media === "video") state.videoClips.push(clip);
            else state.audioClips.push(clip);
            selectClip(media, clip.id);
            ensureTimelineBounds();
            saveToWidget();
        };

        const hasClipForSource = (media, src) => {
            const list = getMediaList(media);
            return list.some((c) => c.src === src);
        };

        const getConnectionSignature = () => {
            const parts = [];
            for (let i = 1; i <= 3; i++) {
                const vName = `video_${i}`;
                const aName = `audio_${i}`;
                const vInput = node.inputs?.find((inp) => inp?.name === vName);
                const aInput = node.inputs?.find((inp) => inp?.name === aName);
                const vLink = Array.isArray(vInput?.links) ? vInput.links.join(",") : `${vInput?.link ?? "null"}`;
                const aLink = Array.isArray(aInput?.links) ? aInput.links.join(",") : `${aInput?.link ?? "null"}`;
                parts.push(`${vName}:${vLink}`);
                parts.push(`${aName}:${aLink}`);
            }
            return parts.join("|");
        };

        const seedConnectedInputs = () => {
            let changed = false;
            const originalPlayhead = state.playhead;

            for (let i = 1; i <= 3; i++) {
                if (isInputConnected(`video_${i}`) && !hasClipForSource("video", i)) {
                    state.videoClips.push({
                        id: makeId("v"),
                        src: i,
                        track: i - 1,
                        start: 0,
                        in: 0,
                        dur: DEFAULT_CLIP_SEC,
                        volume: 1.0,
                        transition_in: null,
                        transition_out: null,
                        enabled: true,
                    });
                    changed = true;
                }
            }

            for (let i = 1; i <= 3; i++) {
                if (isInputConnected(`audio_${i}`) && !hasClipForSource("audio", i)) {
                    state.audioClips.push({
                        id: makeId("a"),
                        src: i,
                        track: i - 1,
                        start: 0,
                        in: 0,
                        dur: DEFAULT_CLIP_SEC,
                        volume: 1.0,
                        transition_in: null,
                        transition_out: null,
                        enabled: true,
                    });
                    changed = true;
                }
            }

            if (changed) {
                state.playhead = originalPlayhead;
                saveToWidget();
            }
        };

        const syncClipsWithConnectedInputs = () => {
            const connectedVideo = new Set();
            const connectedAudio = new Set();
            for (let i = 1; i <= 3; i++) {
                if (isInputConnected(`video_${i}`)) connectedVideo.add(i);
                if (isInputConnected(`audio_${i}`)) connectedAudio.add(i);
            }

            let changed = false;
            const selectedClip = getSelectedClip();
            const selectedWasRemoved = selectedClip && (
                (selected.media === "video" && !connectedVideo.has(selectedClip.src)) ||
                (selected.media === "audio" && !connectedAudio.has(selectedClip.src))
            );

            const prevVideoLen = state.videoClips.length;
            const prevAudioLen = state.audioClips.length;
            state.videoClips = state.videoClips.filter((c) => connectedVideo.has(c.src));
            state.audioClips = state.audioClips.filter((c) => connectedAudio.has(c.src));
            if (state.videoClips.length !== prevVideoLen || state.audioClips.length !== prevAudioLen) {
                changed = true;
            }

            if (selectedWasRemoved) {
                clearSelection();
                changed = true;
            }

            if (changed) saveToWidget();
        };

        const collectSnapTimes = (excludeMedia = null, excludeId = null) => {
            const times = [0];
            const addFrom = (clips, media) => {
                for (const c of clips) {
                    if (media === excludeMedia && c.id === excludeId) continue;
                    times.push(c.start);
                    times.push(c.start + c.dur);
                }
            };
            addFrom(state.videoClips, "video");
            addFrom(state.audioClips, "audio");
            return times;
        };

        const getSnapThresholdSec = (layout) => {
            if (!layout || layout.timeWidth <= 0) return 0;
            return (SNAP_THRESHOLD_PX * getDuration()) / layout.timeWidth;
        };

        const findNearestSnap = (targetTime, snapTimes, thresholdSec) => {
            let best = null;
            let bestDist = thresholdSec + 1e-9;
            for (const t of snapTimes) {
                const d = Math.abs(t - targetTime);
                if (d <= thresholdSec && d < bestDist) {
                    best = t;
                    bestDist = d;
                }
            }
            return best;
        };

        const getSnappedMoveStart = (rawStart, duration, layout, media, id) => {
            const snapTimes = collectSnapTimes(media, id);
            const threshold = getSnapThresholdSec(layout);
            const snapStart = findNearestSnap(rawStart, snapTimes, threshold);
            const snapEnd = findNearestSnap(rawStart + duration, snapTimes, threshold);

            let outStart = rawStart;
            let outGuide = null;
            let bestDist = threshold + 1e-9;

            if (snapStart !== null) {
                const d = Math.abs(rawStart - snapStart);
                if (d < bestDist) {
                    bestDist = d;
                    outStart = snapStart;
                    outGuide = snapStart;
                }
            }

            if (snapEnd !== null) {
                const d = Math.abs((rawStart + duration) - snapEnd);
                if (d < bestDist) {
                    bestDist = d;
                    outStart = snapEnd - duration;
                    outGuide = snapEnd;
                }
            }

            return { start: Math.max(0, outStart), guide: outGuide };
        };

        const getSnappedTime = (rawTime, layout, media, id) => {
            const snapTimes = collectSnapTimes(media, id);
            const threshold = getSnapThresholdSec(layout);
            const snapped = findNearestSnap(rawTime, snapTimes, threshold);
            if (snapped === null) return { time: rawTime, guide: null };
            return { time: snapped, guide: snapped };
        };

        const setTimelineLength = (seconds) => {
            state.timelineLength = Math.max(MIN_TIMELINE_SEC, toNum(seconds, MIN_TIMELINE_SEC));
            saveToWidget();
        };

        const fitTimelineLength = () => {
            let maxEnd = 0;
            for (const c of state.videoClips) maxEnd = Math.max(maxEnd, c.start + c.dur);
            for (const c of state.audioClips) maxEnd = Math.max(maxEnd, c.start + c.dur);
            state.timelineLength = Math.max(MIN_TIMELINE_SEC, maxEnd + 0.5);
            saveToWidget();
        };

        const deleteSelectedClip = () => {
            if (!selected.media || !selected.id) return;
            const arr = getMediaList(selected.media);
            const idx = arr.findIndex((c) => c.id === selected.id);
            if (idx >= 0) {
                arr.splice(idx, 1);
                clearSelection();
                saveToWidget();
            }
        };

        const cutSelectedClip = () => {
            const clip = getSelectedClip();
            if (!clip) return;
            const t = state.playhead;
            const start = clip.start;
            const end = clip.start + clip.dur;
            if (t <= start + CLIP_MIN_SEC || t >= end - CLIP_MIN_SEC) return;

            const leftDur = t - start;
            const rightDur = end - t;

            clip.dur = leftDur;
            const second = {
                ...clip,
                id: makeId(selected.media === "video" ? "v" : "a"),
                start: t,
                in: clip.in + leftDur,
                dur: rightDur,
            };

            getMediaList(selected.media).push(second);
            selectClip(selected.media, second.id);
            saveToWidget();
        };

        const applyButtonAction = (id) => {
            if (id === "add_v1") addClip("video", 1);
            else if (id === "add_v2") addClip("video", 2);
            else if (id === "add_v3") addClip("video", 3);
            else if (id === "add_a1") addClip("audio", 1);
            else if (id === "add_a2") addClip("audio", 2);
            else if (id === "add_a3") addClip("audio", 3);
            else if (id === "len_dec") setTimelineLength(state.timelineLength - 5.0);
            else if (id === "len_inc") setTimelineLength(state.timelineLength + 5.0);
            else if (id === "len_fit") fitTimelineLength();
            else if (id === "cut") cutSelectedClip();
            else if (id === "delete") deleteSelectedClip();
        };

        const collectClipRects = (layout) => {
            const out = [];
            const dur = getDuration();
            const pxPerSec = layout.timeWidth / dur;
            const clipH = Math.max(12, layout.trackH - 6);

            const pushMedia = (arr, media) => {
                for (const clip of arr) {
                    const row = media === "video" ? clip.track : clip.track + VIDEO_TRACKS;
                    const y = layout.tracksTop + row * (layout.trackH + layout.trackGap) + (layout.trackH - clipH) * 0.5;
                    const x = layout.timeLeft + clip.start * pxPerSec;
                    const w = Math.max(8, clip.dur * pxPerSec);
                    out.push({ media, clip, row, x, y, w, h: clipH });
                }
            };

            pushMedia(state.videoClips, "video");
            pushMedia(state.audioClips, "audio");
            out.sort((a, b) => (a.row - b.row) || (a.clip.start - b.clip.start));
            return out;
        };

        const getTransitionDurationValue = (transition) => {
            if (!transition || typeof transition !== "object") return 0;
            return Math.max(0, toNum(transition.duration, 0));
        };

        const getAudioVolumeLineY = (clipRect) => {
            const vol = clamp(toNum(clipRect.clip.volume, 1.0), 0, AUDIO_VOLUME_MAX);
            const t = vol / AUDIO_VOLUME_MAX;
            return clipRect.y + clipRect.h - t * clipRect.h;
        };

        const getAudioFadeHandlePos = (clipRect, layout) => {
            const pxPerSec = layout.timeWidth / Math.max(getDuration(), 0.0001);
            const maxPx = clipRect.w * 0.49;
            const inPx = Math.min(maxPx, getTransitionDurationValue(clipRect.clip.transition_in) * pxPerSec);
            const outPx = Math.min(maxPx, getTransitionDurationValue(clipRect.clip.transition_out) * pxPerSec);
            const y = getAudioVolumeLineY(clipRect);
            return {
                inX: clipRect.x + inPx,
                outX: clipRect.x + clipRect.w - outPx,
                y,
            };
        };

        const getClipRectById = (media, id, layout) => {
            const rects = collectClipRects(layout);
            return rects.find((r) => r.media === media && r.clip.id === id) || null;
        };

        const hitTestButton = (mx, my) => {
            for (const b of uiCache.buttons) {
                if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) return b;
            }
            return null;
        };

        const hitTestAudioControl = (mx, my, layout) => {
            for (let i = uiCache.clipRects.length - 1; i >= 0; i--) {
                const r = uiCache.clipRects[i];
                if (r.media !== "audio") continue;
                if (mx < r.x || mx > r.x + r.w || my < r.y || my > r.y + r.h) continue;

                const handles = getAudioFadeHandlePos(r, layout);
                const dxIn = mx - handles.inX;
                const dyIn = my - handles.y;
                if (dxIn * dxIn + dyIn * dyIn <= (AUDIO_FADE_HANDLE_R + 3) * (AUDIO_FADE_HANDLE_R + 3)) {
                    return { mode: "audio_fade_in", media: "audio", id: r.clip.id };
                }

                const dxOut = mx - handles.outX;
                const dyOut = my - handles.y;
                if (dxOut * dxOut + dyOut * dyOut <= (AUDIO_FADE_HANDLE_R + 3) * (AUDIO_FADE_HANDLE_R + 3)) {
                    return { mode: "audio_fade_out", media: "audio", id: r.clip.id };
                }

                const ly = getAudioVolumeLineY(r);
                const safeLeft = r.x + Math.max(10, TRIM_HANDLE_W + TRIM_HANDLE_PAD + 4);
                const safeRight = r.x + r.w - Math.max(10, TRIM_HANDLE_W + TRIM_HANDLE_PAD + 4);
                if (mx >= safeLeft && mx <= safeRight && Math.abs(my - ly) <= AUDIO_VOL_HIT_PAD) {
                    return { mode: "audio_volume", media: "audio", id: r.clip.id };
                }
            }
            return null;
        };

        const hitTestClip = (mx, my) => {
            for (let i = uiCache.clipRects.length - 1; i >= 0; i--) {
                const r = uiCache.clipRects[i];
                if (mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h) {
                    const edgePad = Math.max(10, TRIM_HANDLE_W + TRIM_HANDLE_PAD + 3);
                    let zone = "body";
                    if (mx <= r.x + edgePad) zone = "left";
                    else if (mx >= r.x + r.w - edgePad) zone = "right";
                    return { ...r, zone };
                }
            }
            return null;
        };

        const isInRuler = (mx, my, layout) => {
            return (
                mx >= layout.timeLeft &&
                mx <= layout.timeLeft + layout.timeWidth &&
                my >= layout.top + layout.toolbarH + 8 &&
                my <= layout.top + layout.toolbarH + 8 + layout.rulerH
            );
        };

        const updatePlayheadFromX = (mx, layout) => {
            state.playhead = xToTime(mx, layout);
        };

        const drawButton = (ctx, b, active = false) => {
            const bg = b.danger ? COLORS.buttonBgDanger : COLORS.buttonBg;
            const fill = active ? "rgba(255,255,255,0.22)" : bg;
            roundRect(ctx, b.x, b.y, b.w, b.h, 6);
            ctx.fillStyle = fill;
            ctx.fill();
            ctx.strokeStyle = "rgba(255,255,255,0.18)";
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = COLORS.buttonText;
            ctx.font = "bold 11px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(b.label, b.x + b.w / 2, b.y + b.h / 2 + 0.5);
        };

        const drawRulerTicks = (ctx, layout) => {
            const dur = getDuration();
            const idealStep = dur / 10;
            const steps = [0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
            let step = steps[steps.length - 1];
            for (const s of steps) {
                if (s >= idealStep) {
                    step = s;
                    break;
                }
            }

            ctx.strokeStyle = "rgba(210,220,255,0.20)";
            ctx.fillStyle = COLORS.textMuted;
            ctx.font = "10px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "top";

            for (let t = 0; t <= dur + 0.0001; t += step) {
                const x = timeToX(t, layout);
                ctx.beginPath();
                ctx.moveTo(x, layout.top + layout.toolbarH + 8);
                ctx.lineTo(x, layout.top + layout.toolbarH + 8 + layout.rulerH);
                ctx.stroke();
                ctx.fillText(formatTime(t), x, layout.top + layout.toolbarH + 10);
            }
        };

        const drawUI = (ctx) => {
            hideTimelineWidget();
            ensureNodeSize();
            const sig = getConnectionSignature();
            if (sig !== lastConnectionSig) {
                lastConnectionSig = sig;
                syncClipsWithConnectedInputs();
                seedConnectedInputs();
            }

            const layout = getLayout();
            uiCache.clipRects = collectClipRects(layout);
            uiCache.buttons = [];

            roundRect(ctx, layout.left - 1, layout.top - 1, layout.width + 2, layout.height + 2, 8);
            ctx.fillStyle = COLORS.panelBg;
            ctx.fill();
            ctx.strokeStyle = COLORS.panelStroke;
            ctx.lineWidth = 1;
            ctx.stroke();

            roundRect(ctx, layout.left + 2, layout.top + 2, layout.width - 4, layout.toolbarH, 6);
            ctx.fillStyle = COLORS.toolbarBg;
            ctx.fill();

            let bx = layout.left + 8;
            const by = layout.top + 5;
            for (const spec of BUTTONS) {
                const b = {
                    ...spec,
                    x: bx,
                    y: by,
                    h: layout.toolbarH - 6,
                };
                uiCache.buttons.push(b);
                drawButton(ctx, b, hoverButtonId === b.id);
                bx += b.w + 6;
            }

            ctx.fillStyle = COLORS.textMuted;
            ctx.font = "11px sans-serif";
            ctx.textAlign = "right";
            ctx.textBaseline = "middle";
            ctx.fillText(`Playhead: ${formatTime(state.playhead)} / ${formatTime(getDuration())}`, layout.left + layout.width - 8, by + (layout.toolbarH - 6) / 2);

            roundRect(ctx, layout.left + 2, layout.top + layout.toolbarH + 8, layout.width - 4, layout.rulerH, 6);
            ctx.fillStyle = COLORS.rulerBg;
            ctx.fill();
            drawRulerTicks(ctx, layout);

            for (let row = 0; row < TOTAL_TRACKS; row++) {
                const y = layout.tracksTop + row * (layout.trackH + layout.trackGap);
                const info = getTrackInfoFromRow(row);
                const isVideo = info?.media === "video";
                roundRect(ctx, layout.left + 2, y, layout.width - 4, layout.trackH, 5);
                ctx.fillStyle = isVideo ? COLORS.trackVideo : COLORS.trackAudio;
                ctx.fill();
                ctx.strokeStyle = COLORS.trackBorder;
                ctx.lineWidth = 1;
                ctx.stroke();

                ctx.fillStyle = COLORS.textStrong;
                ctx.font = "bold 11px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(info?.label || "", layout.left + layout.labelW * 0.5, y + layout.trackH * 0.5);
            }

            const pxPerSec = layout.timeWidth / Math.max(getDuration(), 0.0001);
            for (const r of uiCache.clipRects) {
                const isSel = selected.media === r.media && selected.id === r.clip.id;
                const fill = isSel ? COLORS.clipSel : (r.media === "video" ? COLORS.clipVideo : COLORS.clipAudio);
                roundRect(ctx, r.x, r.y, r.w, r.h, 6);
                ctx.fillStyle = fill;
                ctx.fill();
                ctx.strokeStyle = isSel ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.32)";
                ctx.lineWidth = isSel ? 1.8 : 1;
                ctx.stroke();

                // Visual trim grabbers at start/end
                ctx.fillStyle = "rgba(10,12,16,0.75)";
                ctx.fillRect(r.x + TRIM_HANDLE_PAD, r.y + 1, TRIM_HANDLE_W, r.h - 2);
                ctx.fillRect(r.x + r.w - TRIM_HANDLE_W - TRIM_HANDLE_PAD, r.y + 1, TRIM_HANDLE_W, r.h - 2);

                const tIn = r.clip.transition_in;
                if (tIn?.duration > 0) {
                    const tw = Math.min(r.w * 0.48, Math.max(3, tIn.duration * pxPerSec));
                    ctx.fillStyle = "rgba(255,255,255,0.22)";
                    ctx.fillRect(r.x + 1, r.y + 1, tw, Math.min(8, r.h - 2));
                }

                const tOut = r.clip.transition_out;
                if (tOut?.duration > 0) {
                    const tw = Math.min(r.w * 0.48, Math.max(3, tOut.duration * pxPerSec));
                    ctx.fillStyle = "rgba(255,255,255,0.22)";
                    ctx.fillRect(r.x + r.w - tw - 1, r.y + 1, tw, Math.min(8, r.h - 2));
                }

                if (r.media === "audio") {
                    const volY = getAudioVolumeLineY(r);
                    const handles = getAudioFadeHandlePos(r, layout);

                    ctx.strokeStyle = "rgba(255,255,255,0.8)";
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(r.x + 6, volY);
                    ctx.lineTo(r.x + r.w - 6, volY);
                    ctx.stroke();

                    ctx.beginPath();
                    ctx.fillStyle = "rgba(250,250,255,0.95)";
                    ctx.arc(handles.inX, handles.y, AUDIO_FADE_HANDLE_R, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.beginPath();
                    ctx.arc(handles.outX, handles.y, AUDIO_FADE_HANDLE_R, 0, Math.PI * 2);
                    ctx.fill();
                }

                ctx.fillStyle = "rgba(18,18,22,0.92)";
                ctx.font = "bold 10px sans-serif";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                const volText = r.media === "audio" ? `  vol:${toNum(r.clip.volume, 1).toFixed(2)}` : "";
                const label = `${r.media === "video" ? "V" : "A"}${r.clip.src}  ${r.clip.dur.toFixed(2)}s${volText}`;
                ctx.fillText(label, r.x + 6, r.y + r.h * 0.5);
            }

            const phX = timeToX(state.playhead, layout);
            const phTop = layout.top + layout.toolbarH + 8;
            const phBottom = layout.tracksTop + layout.tracksHeight;

            if (snapGuideTime !== null) {
                const sx = timeToX(snapGuideTime, layout);
                ctx.save();
                ctx.strokeStyle = "rgba(255,255,255,0.55)";
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.moveTo(sx, phTop);
                ctx.lineTo(sx, phBottom);
                ctx.stroke();
                ctx.restore();
            }

            ctx.strokeStyle = COLORS.playhead;
            ctx.lineWidth = 1.6;
            ctx.beginPath();
            ctx.moveTo(phX, phTop);
            ctx.lineTo(phX, phBottom);
            ctx.stroke();

            ctx.fillStyle = COLORS.playhead;
            ctx.beginPath();
            ctx.moveTo(phX, phTop);
            ctx.lineTo(phX - 5, phTop - 8);
            ctx.lineTo(phX + 5, phTop - 8);
            ctx.closePath();
            ctx.fill();

            ctx.fillStyle = COLORS.textMuted;
            ctx.font = "10px sans-serif";
            ctx.textAlign = "left";
            ctx.textBaseline = "alphabetic";
            ctx.fillText("Audio: drag line for volume, side circles for fades. Clips: edge grabbers trim, snap enabled.", layout.left + 4, layout.top + layout.height - 4);
        };

        const startDrag = (mode, data = {}) => {
            drag = { mode, ...data };
            snapGuideTime = null;
        };

        const stopDrag = () => {
            drag = null;
            snapGuideTime = null;
        };

        const isPrimaryButtonPressed = (e) => {
            if (!e) return true;
            if (typeof e.buttons === "number") return (e.buttons & 1) === 1;
            if (typeof e.which === "number") return e.which === 1;
            return true;
        };

        const origOnDrawForeground = node.onDrawForeground;
        node.onDrawForeground = function (ctx) {
            if (origOnDrawForeground) origOnDrawForeground.apply(this, arguments);
            drawUI(ctx);
        };

        const origOnMouseDown = node.onMouseDown;
        node.onMouseDown = function (e, localPos) {
            const mx = localPos[0];
            const my = localPos[1];
            const layout = getLayout();

            const hitButton = hitTestButton(mx, my);
            if (hitButton) {
                applyButtonAction(hitButton.id);
                return true;
            }

            const audioCtrl = hitTestAudioControl(mx, my, layout);
            if (audioCtrl) {
                selectClip(audioCtrl.media, audioCtrl.id);
                const clip = findClip(audioCtrl.media, audioCtrl.id);
                startDrag(audioCtrl.mode, {
                    media: audioCtrl.media,
                    id: audioCtrl.id,
                    startX: mx,
                    startY: my,
                    origVolume: clip ? toNum(clip.volume, 1.0) : 1.0,
                });
                node.setDirtyCanvas(true, true);
                return true;
            }

            const hitClip = hitTestClip(mx, my);
            if (hitClip) {
                selectClip(hitClip.media, hitClip.clip.id);
                const mode = hitClip.zone === "left" ? "resize_left" : hitClip.zone === "right" ? "resize_right" : "move";
                startDrag(mode, {
                    media: hitClip.media,
                    id: hitClip.clip.id,
                    startX: mx,
                    startY: my,
                    origStart: hitClip.clip.start,
                    origDur: hitClip.clip.dur,
                    origIn: hitClip.clip.in,
                    origTrack: hitClip.clip.track,
                });
                node.setDirtyCanvas(true, true);
                return true;
            }

            if (isInRuler(mx, my, layout)) {
                updatePlayheadFromX(mx, layout);
                clearSelection();
                startDrag("scrub", { startX: mx });
                saveToWidget();
                return true;
            }

            if (origOnMouseDown) return origOnMouseDown.apply(this, arguments);
            return false;
        };

        const origOnMouseMove = node.onMouseMove;
        node.onMouseMove = function (e, localPos) {
            const mx = localPos[0];
            const my = localPos[1];
            const layout = getLayout();

            const hoveredButton = hitTestButton(mx, my);
            const newHover = hoveredButton ? hoveredButton.id : null;
            if (newHover !== hoverButtonId) {
                hoverButtonId = newHover;
                node.setDirtyCanvas(true, true);
            }

            if (!drag) {
                if (origOnMouseMove) return origOnMouseMove.apply(this, arguments);
                return false;
            }

            if (!isPrimaryButtonPressed(e)) {
                stopDrag();
                node.setDirtyCanvas(true, true);
                return true;
            }

            if (drag.mode === "scrub") {
                updatePlayheadFromX(mx, layout);
                snapGuideTime = null;
                saveToWidget();
                return true;
            }

            if (drag.mode === "audio_volume" || drag.mode === "audio_fade_in" || drag.mode === "audio_fade_out") {
                const clip = findClip("audio", drag.id);
                const rect = getClipRectById("audio", drag.id, layout);
                if (!clip || !rect) {
                    stopDrag();
                    return true;
                }

                if (drag.mode === "audio_volume") {
                    const norm = clamp((rect.y + rect.h - my) / Math.max(rect.h, 1), 0, 1);
                    clip.volume = norm * AUDIO_VOLUME_MAX;
                } else {
                    const pxPerSec = layout.timeWidth / Math.max(getDuration(), 0.0001);
                    const maxFade = Math.max(0, clip.dur * 0.49);
                    let fadeDur = 0;
                    if (drag.mode === "audio_fade_in") {
                        fadeDur = clamp((mx - rect.x) / Math.max(pxPerSec, 0.0001), 0, maxFade);
                        clip.transition_in = fadeDur <= 0.02 ? null : { type: "Fade", duration: Number(fadeDur.toFixed(4)) };
                    } else {
                        fadeDur = clamp((rect.x + rect.w - mx) / Math.max(pxPerSec, 0.0001), 0, maxFade);
                        clip.transition_out = fadeDur <= 0.02 ? null : { type: "Fade", duration: Number(fadeDur.toFixed(4)) };
                    }
                }

                snapGuideTime = null;
                ensureTimelineBounds();
                saveToWidget();
                return true;
            }

            const clip = findClip(drag.media, drag.id);
            if (!clip) {
                stopDrag();
                return true;
            }

            const secPerPx = getDuration() / layout.timeWidth;
            const deltaSec = (mx - drag.startX) * secPerPx;

            if (drag.mode === "move") {
                const rawStart = Math.max(0, drag.origStart + deltaSec);
                const snapped = getSnappedMoveStart(rawStart, clip.dur, layout, drag.media, drag.id);
                clip.start = snapped.start;
                snapGuideTime = snapped.guide;
                const row = rowFromY(my, layout);
                const info = getTrackInfoFromRow(row);
                if (info && info.media === drag.media) {
                    clip.track = info.track;
                }
            } else if (drag.mode === "resize_right") {
                const rawDur = Math.max(CLIP_MIN_SEC, drag.origDur + deltaSec);
                const rawEnd = clip.start + rawDur;
                const snapped = getSnappedTime(rawEnd, layout, drag.media, drag.id);
                const newEnd = Math.max(clip.start + CLIP_MIN_SEC, snapped.time);
                clip.dur = Math.max(CLIP_MIN_SEC, newEnd - clip.start);
                snapGuideTime = snapped.guide;
            } else if (drag.mode === "resize_left") {
                let newStart = drag.origStart + deltaSec;
                const maxStart = drag.origStart + drag.origDur - CLIP_MIN_SEC;
                const snapped = getSnappedTime(newStart, layout, drag.media, drag.id);
                newStart = clamp(snapped.time, 0, maxStart);
                snapGuideTime = snapped.guide;

                let shift = newStart - drag.origStart;
                let newIn = drag.origIn + shift;
                if (newIn < 0) {
                    newStart -= newIn;
                    newIn = 0;
                    shift = newStart - drag.origStart;
                }

                clip.start = newStart;
                clip.in = newIn;
                clip.dur = Math.max(CLIP_MIN_SEC, drag.origDur - shift);
            } else {
                snapGuideTime = null;
            }

            ensureTimelineBounds();
            saveToWidget();
            return true;
        };

        const origOnMouseUp = node.onMouseUp;
        node.onMouseUp = function () {
            stopDrag();
            if (origOnMouseUp) return origOnMouseUp.apply(this, arguments);
            return false;
        };

        const origOnDblClick = node.onDblClick;
        node.onDblClick = function (e, localPos) {
            const mx = localPos[0];
            const my = localPos[1];
            const layout = getLayout();
            const row = rowFromY(my, layout);
            const info = getTrackInfoFromRow(row);
            if (info) {
                state.playhead = xToTime(mx, layout);
                addClip(info.media, 1, info.track);
                return true;
            }
            if (origOnDblClick) return origOnDblClick.apply(this, arguments);
            return false;
        };

        const handleHotkey = (ev) => {
            if (!ev) return;
            if (!isNodeSelected(node) && !selected.id) return false;

            const key = (ev.key || "").toLowerCase();
            if (key === "delete" || key === "backspace") {
                deleteSelectedClip();
                return true;
            }
            if (key === "c" && !ev.ctrlKey && !ev.metaKey) {
                cutSelectedClip();
                return true;
            }
            if (key === "1" || key === "2" || key === "3") {
                const clip = getSelectedClip();
                if (clip) {
                    clip.src = Number(key);
                    saveToWidget();
                    return true;
                }
            }
            return false;
        };

        const keyHandler = (ev) => {
            if (!handleHotkey(ev)) return;
            ev.preventDefault();
            ev.stopPropagation();
        };

        const origOnKeyDown = node.onKeyDown;
        node.onKeyDown = function (ev) {
            if (handleHotkey(ev)) return true;
            if (origOnKeyDown) return origOnKeyDown.apply(this, arguments);
            return false;
        };

        window.addEventListener("keydown", keyHandler, true);
        const forceStopDrag = () => {
            if (!drag) return;
            stopDrag();
            node.setDirtyCanvas(true, true);
        };
        window.addEventListener("mouseup", forceStopDrag, true);
        window.addEventListener("pointerup", forceStopDrag, true);
        window.addEventListener("blur", forceStopDrag, true);

        const origOnRemoved = node.onRemoved;
        node.onRemoved = function () {
            window.removeEventListener("keydown", keyHandler, true);
            window.removeEventListener("mouseup", forceStopDrag, true);
            window.removeEventListener("pointerup", forceStopDrag, true);
            window.removeEventListener("blur", forceStopDrag, true);
            if (origOnRemoved) return origOnRemoved.apply(this, arguments);
            return undefined;
        };

        const origOnConfigure = node.onConfigure;
        node.onConfigure = function () {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            ensureTransitionControls();
            loadFromWidget();
            hideTimelineWidget();
            ensureNodeSize();
            node.setDirtyCanvas(true, true);
        };

        const origOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function () {
            if (origOnConnectionsChange) origOnConnectionsChange.apply(this, arguments);
            lastConnectionSig = getConnectionSignature();
            syncClipsWithConnectedInputs();
            seedConnectedInputs();
            node.setDirtyCanvas(true, true);
        };

        setTimeout(() => {
            ensureTransitionControls();
            loadFromWidget();
            hideTimelineWidget();
            ensureNodeSize();
            lastConnectionSig = getConnectionSignature();
            syncClipsWithConnectedInputs();
            seedConnectedInputs();
            saveToWidget();
        }, 160);
    },
});
