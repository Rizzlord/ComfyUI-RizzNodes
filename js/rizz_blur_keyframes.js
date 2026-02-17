import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EDITOR_MIN_W = 320;
const EDITOR_MIN_H = 240;
const TIMELINE_H = 40;
const CANVAS_PADDING = 10;
const CANVAS_TOP_GAP = 8;
const KF_DOT_RADIUS = 7;
const KF_DOT_RADIUS_TIMELINE = 5;
const BLUR_CIRCLE_COLOR = "rgba(60,140,255,0.35)";
const BLUR_CIRCLE_STROKE = "rgba(60,140,255,0.85)";
const INPAINT_CIRCLE_COLOR = "rgba(255,80,60,0.30)";
const INPAINT_CIRCLE_STROKE = "rgba(255,80,60,0.85)";
const KF_COLOR_ACTIVE = "#ff6b35";
const KF_COLOR_INACTIVE = "#4ea8ff";
const KF_COLOR_HOVER = "#ffc107";
const TIMELINE_BG = "rgba(30,30,30,0.85)";
const TIMELINE_TRACK = "rgba(80,80,80,0.6)";
const PLAYHEAD_COLOR = "#ff6b35";

app.registerExtension({
    name: "rizznodes.blur_keyframes",

    nodeCreated(node) {
        if (node.comfyClass !== "RizzBlurSpot") return;

        const getWidget = (name) => {
            if (!Array.isArray(node.widgets)) return null;
            return node.widgets.find(w => w?.name === name) || null;
        };

        let keyframes = [];
        let currentFrame = 0;
        let totalFrames = 100;
        let videoWidth = 640;
        let videoHeight = 360;
        let selectedKfIndex = -1;
        let draggingKf = -1;
        let draggingTimeline = false;
        let hoverKfIndex = -1;
        let previewImage = null;
        let previewLoaded = false;
        let lastVideoPath = null;

        const loadKeyframesFromWidget = () => {
            const w = getWidget("keyframes_json");
            if (!w) return;
            try {
                const parsed = JSON.parse(w.value);
                if (Array.isArray(parsed)) {
                    keyframes = parsed;
                    keyframes.sort((a, b) => a.frame - b.frame);
                }
            } catch (e) {
                keyframes = [];
            }
        };

        const saveKeyframesToWidget = () => {
            const w = getWidget("keyframes_json");
            if (!w) return;
            keyframes.sort((a, b) => a.frame - b.frame);
            w.value = JSON.stringify(keyframes);
        };

        const hideKeyframesWidget = () => {
            const w = getWidget("keyframes_json");
            if (!w) return;
            w.hidden = true;
            w.type = "hidden";
            w.computeSize = () => [0, -4];
            w.draw = () => { };
            if (w.inputEl) {
                w.inputEl.style.display = "none";
            }
        };

        const getVisibleWidgets = () => {
            if (!node.widgets) return [];
            return node.widgets.filter(w => !w.hidden && w.type !== "hidden");
        };

        const getEditorArea = () => {
            const visible = getVisibleWidgets();
            let widgetBottom = 0;
            for (const w of visible) {
                let h = 20;
                if (Number.isFinite(w.computedHeight)) h = w.computedHeight;
                else if (Number.isFinite(w.last_y) && Number.isFinite(w.computedHeight)) {
                    h = w.computedHeight;
                }
                const y = Number.isFinite(w.last_y) ? w.last_y : (Number.isFinite(w.y) ? w.y : 0);
                const bottom = y + h;
                if (Number.isFinite(bottom)) widgetBottom = Math.max(widgetBottom, bottom);
            }

            if (widgetBottom < 30 && visible.length > 0) {
                widgetBottom = 30 + visible.length * 24;
            }

            const top = widgetBottom + CANVAS_TOP_GAP;
            const left = CANVAS_PADDING;
            const right = node.size[0] - CANVAS_PADDING;
            const bottom = node.size[1] - CANVAS_PADDING;
            const w = Math.max(right - left, 1);
            const h = Math.max(bottom - top - TIMELINE_H - 8, 1);

            return { top, left, w, h };
        };

        const getTimelineArea = () => {
            const ea = getEditorArea();
            return {
                left: ea.left,
                top: ea.top + ea.h + 8,
                w: ea.w,
                h: TIMELINE_H
            };
        };

        const resolveVideoPath = () => {
            const videoInput = node.inputs?.find(i => i.name === "video");
            if (!videoInput || !videoInput.link) return null;

            const linkInfo = app.graph.links[videoInput.link];
            if (!linkInfo) return null;

            const sourceNode = app.graph.getNodeById(linkInfo.origin_id);
            if (!sourceNode) return null;

            if (sourceNode.widgets) {
                for (const w of sourceNode.widgets) {
                    if (w.name === "video_path" || w.name === "path") {
                        if (typeof w.value === "string" && w.value.length > 2) return w.value;
                    }
                }
            }

            const outputSlot = linkInfo.origin_slot;
            if (sourceNode.outputs && sourceNode.outputs[outputSlot]) {
                const outputName = sourceNode.outputs[outputSlot].name;
                if (outputName === "video_path" || outputName === "path") {
                    if (sourceNode._last_output && typeof sourceNode._last_output === "string") {
                        return sourceNode._last_output;
                    }
                }
            }

            if (sourceNode.widgets_values) {
                for (const val of sourceNode.widgets_values) {
                    if (typeof val === "string" && /\.(mp4|mkv|mov|webm|avi)$/i.test(val)) {
                        return val;
                    }
                }
            }

            if (sourceNode.widgets) {
                for (const w of sourceNode.widgets) {
                    if (typeof w.value === "string" && /\.(mp4|mkv|mov|webm|avi)$/i.test(w.value)) {
                        return w.value;
                    }
                }
            }

            return null;
        };

        const tryLoadPreview = async () => {
            const videoInput = node.inputs?.find(i => i.name === "video");
            if (!videoInput || !videoInput.link) {
                previewImage = null;
                previewLoaded = false;
                lastVideoPath = null;
                return;
            }

            const videoPath = resolveVideoPath();
            if (!videoPath) return;
            if (videoPath === lastVideoPath && previewLoaded) return;
            lastVideoPath = videoPath;

            try {
                const resp = await api.fetchApi("/rizz/video_first_frame", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ path: videoPath })
                });

                if (!resp.ok) return;

                const vw = parseInt(resp.headers.get("X-Video-Width")) || 640;
                const vh = parseInt(resp.headers.get("X-Video-Height")) || 360;
                const vf = parseInt(resp.headers.get("X-Video-Frames")) || 100;
                videoWidth = vw;
                videoHeight = vh;
                totalFrames = vf;

                const blob = await resp.blob();
                const url = URL.createObjectURL(blob);
                const img = new Image();
                img.onload = () => {
                    previewImage = img;
                    previewLoaded = true;
                    node.setDirtyCanvas(true, true);
                };
                img.src = url;
            } catch (e) {
                console.warn("[RizzBlurSpot] preview load failed:", e);
            }
        };

        const ensureNodeSize = () => {
            const minW = Math.max(node.size[0], EDITOR_MIN_W + CANVAS_PADDING * 2);
            const ea = getEditorArea();
            const widgetBottom = ea.top - CANVAS_TOP_GAP;
            const minH = widgetBottom + CANVAS_TOP_GAP + EDITOR_MIN_H + TIMELINE_H + 16 + CANVAS_PADDING;
            const h = Math.max(node.size[1], minH);
            if (node.size[0] !== minW || node.size[1] !== h) {
                node.setSize([minW, h]);
            }
        };

        loadKeyframesFromWidget();

        setTimeout(() => {
            hideKeyframesWidget();
            ensureNodeSize();
            tryLoadPreview();
            node.setDirtyCanvas(true, true);
        }, 300);

        const origOnDrawForeground = node.onDrawForeground;
        node.onDrawForeground = function (ctx) {
            if (origOnDrawForeground) origOnDrawForeground.apply(this, arguments);

            hideKeyframesWidget();

            const ea = getEditorArea();
            const ta = getTimelineArea();

            ctx.save();
            ctx.fillStyle = "rgba(20,20,25,0.92)";
            ctx.strokeStyle = "rgba(60,140,255,0.3)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(ea.left - 2, ea.top - 2, ea.w + 4, ea.h + 4, 6);
            ctx.fill();
            ctx.stroke();

            ctx.beginPath();
            ctx.rect(ea.left, ea.top, ea.w, ea.h);
            ctx.clip();

            if (previewImage && previewLoaded) {
                const scale = Math.min(ea.w / previewImage.naturalWidth, ea.h / previewImage.naturalHeight);
                const dw = previewImage.naturalWidth * scale;
                const dh = previewImage.naturalHeight * scale;
                const dx = ea.left + (ea.w - dw) / 2;
                const dy = ea.top + (ea.h - dh) / 2;
                ctx.globalAlpha = 0.4;
                ctx.drawImage(previewImage, dx, dy, dw, dh);
                ctx.globalAlpha = 1.0;
            } else {
                ctx.fillStyle = "rgba(255,255,255,0.06)";
                ctx.fillRect(ea.left, ea.top, ea.w, ea.h);
                ctx.fillStyle = "rgba(255,255,255,0.3)";
                ctx.font = "12px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("Connect a video to see preview", ea.left + ea.w / 2, ea.top + ea.h / 2);
            }

            const blurSizeW = getWidget("blur_size");
            const blurRadius = blurSizeW ? blurSizeW.value : 100;

            const canvasScaleX = ea.w / videoWidth;
            const canvasScaleY = ea.h / videoHeight;
            const circleCanvasR = blurRadius * Math.min(canvasScaleX, canvasScaleY);

            const interpKf = _interpolateKfAtFrame(currentFrame);
            if (interpKf) {
                const cx = ea.left + interpKf.x * ea.w;
                const cy = ea.top + interpKf.y * ea.h;

                const modeW = getWidget("mode");
                const isInpaint = modeW && modeW.value === "Watermark Removal";

                ctx.beginPath();
                ctx.arc(cx, cy, circleCanvasR, 0, Math.PI * 2);
                ctx.fillStyle = isInpaint ? INPAINT_CIRCLE_COLOR : BLUR_CIRCLE_COLOR;
                ctx.fill();
                ctx.strokeStyle = isInpaint ? INPAINT_CIRCLE_STROKE : BLUR_CIRCLE_STROKE;
                ctx.lineWidth = 2;
                ctx.stroke();
            }

            for (let i = 0; i < keyframes.length; i++) {
                const kf = keyframes[i];
                const kx = ea.left + kf.x * ea.w;
                const ky = ea.top + kf.y * ea.h;
                const r = KF_DOT_RADIUS;

                let color = KF_COLOR_INACTIVE;
                if (i === selectedKfIndex) color = KF_COLOR_ACTIVE;
                else if (i === hoverKfIndex) color = KF_COLOR_HOVER;

                ctx.beginPath();
                ctx.arc(kx, ky, r, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 1.5;
                ctx.stroke();

                ctx.fillStyle = "#fff";
                ctx.font = "bold 9px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText(`${kf.frame}`, kx, ky - r - 3);
            }

            if (keyframes.length >= 2) {
                ctx.strokeStyle = "rgba(255,255,255,0.15)";
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                for (let i = 0; i < keyframes.length; i++) {
                    const kx = ea.left + keyframes[i].x * ea.w;
                    const ky = ea.top + keyframes[i].y * ea.h;
                    if (i === 0) ctx.moveTo(kx, ky);
                    else ctx.lineTo(kx, ky);
                }
                ctx.stroke();
                ctx.setLineDash([]);
            }

            ctx.restore();

            ctx.save();
            ctx.fillStyle = TIMELINE_BG;
            ctx.beginPath();
            ctx.roundRect(ta.left, ta.top, ta.w, ta.h, 6);
            ctx.fill();

            const trackY = ta.top + ta.h / 2;
            const trackLeft = ta.left + 10;
            const trackRight = ta.left + ta.w - 10;
            const trackW = trackRight - trackLeft;

            ctx.strokeStyle = TIMELINE_TRACK;
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(trackLeft, trackY);
            ctx.lineTo(trackRight, trackY);
            ctx.stroke();

            for (let i = 0; i < keyframes.length; i++) {
                const kf = keyframes[i];
                const tx = trackLeft + (kf.frame / Math.max(totalFrames - 1, 1)) * trackW;
                let color = KF_COLOR_INACTIVE;
                if (i === selectedKfIndex) color = KF_COLOR_ACTIVE;

                ctx.beginPath();
                ctx.arc(tx, trackY, KF_DOT_RADIUS_TIMELINE, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 1;
                ctx.stroke();
            }

            const playheadX = trackLeft + (currentFrame / Math.max(totalFrames - 1, 1)) * trackW;
            ctx.beginPath();
            ctx.moveTo(playheadX, ta.top + 4);
            ctx.lineTo(playheadX - 5, ta.top + 12);
            ctx.lineTo(playheadX + 5, ta.top + 12);
            ctx.closePath();
            ctx.fillStyle = PLAYHEAD_COLOR;
            ctx.fill();

            ctx.strokeStyle = PLAYHEAD_COLOR;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(playheadX, ta.top + 12);
            ctx.lineTo(playheadX, ta.top + ta.h - 4);
            ctx.stroke();

            ctx.fillStyle = "rgba(255,255,255,0.7)";
            ctx.font = "10px sans-serif";
            ctx.textAlign = "left";
            ctx.fillText(`Frame: ${currentFrame} / ${totalFrames}`, ta.left + 10, ta.top + ta.h - 6);

            ctx.textAlign = "right";
            ctx.fillText(`${keyframes.length} keyframe${keyframes.length !== 1 ? "s" : ""}`, ta.left + ta.w - 10, ta.top + ta.h - 6);

            ctx.restore();
        };

        function _interpolateKfAtFrame(frame) {
            if (keyframes.length === 0) return null;
            if (keyframes.length === 1) return { x: keyframes[0].x, y: keyframes[0].y };

            if (frame <= keyframes[0].frame) return { x: keyframes[0].x, y: keyframes[0].y };
            if (frame >= keyframes[keyframes.length - 1].frame) {
                return { x: keyframes[keyframes.length - 1].x, y: keyframes[keyframes.length - 1].y };
            }

            let before = keyframes[0], after = keyframes[keyframes.length - 1];
            for (let i = 0; i < keyframes.length - 1; i++) {
                if (keyframes[i].frame <= frame && keyframes[i + 1].frame >= frame) {
                    before = keyframes[i];
                    after = keyframes[i + 1];
                    break;
                }
            }

            const span = after.frame - before.frame;
            if (span <= 0) return { x: before.x, y: before.y };
            let t = (frame - before.frame) / span;

            const interpW = getWidget("interpolation");
            const mode = interpW ? interpW.value : "linear";
            if (mode === "ease_in") t = t * t;
            else if (mode === "ease_out") t = 1 - (1 - t) * (1 - t);
            else if (mode === "ease_in_out") t = 3 * t * t - 2 * t * t * t;

            return {
                x: before.x + (after.x - before.x) * t,
                y: before.y + (after.y - before.y) * t
            };
        }

        function findKfAtCanvasPos(mx, my) {
            const ea = getEditorArea();
            for (let i = keyframes.length - 1; i >= 0; i--) {
                const kx = ea.left + keyframes[i].x * ea.w;
                const ky = ea.top + keyframes[i].y * ea.h;
                const dx = mx - kx, dy = my - ky;
                if (dx * dx + dy * dy <= (KF_DOT_RADIUS + 4) * (KF_DOT_RADIUS + 4)) {
                    return i;
                }
            }
            return -1;
        }

        function frameFromTimelineX(mx) {
            const ta = getTimelineArea();
            const trackLeft = ta.left + 10;
            const trackRight = ta.left + ta.w - 10;
            const trackW = trackRight - trackLeft;
            const t = Math.max(0, Math.min(1, (mx - trackLeft) / trackW));
            return Math.round(t * Math.max(totalFrames - 1, 1));
        }

        function isInEditorArea(mx, my) {
            const ea = getEditorArea();
            return mx >= ea.left && mx <= ea.left + ea.w && my >= ea.top && my <= ea.top + ea.h;
        }

        function isInTimelineArea(mx, my) {
            const ta = getTimelineArea();
            return mx >= ta.left && mx <= ta.left + ta.w && my >= ta.top && my <= ta.top + ta.h;
        }

        const origOnMouseDown = node.onMouseDown;
        node.onMouseDown = function (e, localPos) {
            const mx = localPos[0], my = localPos[1];

            if (isInTimelineArea(mx, my)) {
                draggingTimeline = true;
                currentFrame = frameFromTimelineX(mx);
                node.setDirtyCanvas(true, true);
                return true;
            }

            if (isInEditorArea(mx, my)) {
                const idx = findKfAtCanvasPos(mx, my);
                if (idx >= 0) {
                    selectedKfIndex = idx;
                    draggingKf = idx;
                    currentFrame = keyframes[idx].frame;
                    node.setDirtyCanvas(true, true);
                    return true;
                } else {
                    const ea = getEditorArea();
                    const xNorm = (mx - ea.left) / ea.w;
                    const yNorm = (my - ea.top) / ea.h;
                    keyframes.push({ frame: currentFrame, x: xNorm, y: yNorm });
                    keyframes.sort((a, b) => a.frame - b.frame);
                    selectedKfIndex = keyframes.findIndex(k => k.frame === currentFrame && k.x === xNorm && k.y === yNorm);
                    saveKeyframesToWidget();
                    node.setDirtyCanvas(true, true);
                    return true;
                }
            }

            if (origOnMouseDown) return origOnMouseDown.apply(this, arguments);
        };

        const origOnMouseMove = node.onMouseMove;
        node.onMouseMove = function (e, localPos) {
            const mx = localPos[0], my = localPos[1];
            const ea = getEditorArea();

            if (draggingTimeline) {
                currentFrame = frameFromTimelineX(mx);
                node.setDirtyCanvas(true, true);
                return true;
            }

            if (draggingKf >= 0 && draggingKf < keyframes.length) {
                keyframes[draggingKf].x = Math.max(0, Math.min(1, (mx - ea.left) / ea.w));
                keyframes[draggingKf].y = Math.max(0, Math.min(1, (my - ea.top) / ea.h));
                saveKeyframesToWidget();
                node.setDirtyCanvas(true, true);
                return true;
            }

            if (isInEditorArea(mx, my)) {
                const oldHover = hoverKfIndex;
                hoverKfIndex = findKfAtCanvasPos(mx, my);
                if (oldHover !== hoverKfIndex) node.setDirtyCanvas(true, true);
            } else {
                if (hoverKfIndex >= 0) {
                    hoverKfIndex = -1;
                    node.setDirtyCanvas(true, true);
                }
            }

            if (origOnMouseMove) return origOnMouseMove.apply(this, arguments);
        };

        const origOnMouseUp = node.onMouseUp;
        node.onMouseUp = function (e, localPos) {
            if (draggingKf >= 0) {
                saveKeyframesToWidget();
            }
            draggingKf = -1;
            draggingTimeline = false;

            if (origOnMouseUp) return origOnMouseUp.apply(this, arguments);
        };

        node.onDblClick = function (e, localPos) {
            const mx = localPos[0], my = localPos[1];
            if (isInEditorArea(mx, my)) {
                const idx = findKfAtCanvasPos(mx, my);
                if (idx >= 0) {
                    keyframes.splice(idx, 1);
                    selectedKfIndex = -1;
                    saveKeyframesToWidget();
                    node.setDirtyCanvas(true, true);
                    return true;
                }
            }
        };

        const origOnConfigure = node.onConfigure;
        node.onConfigure = function () {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            loadKeyframesFromWidget();
            hideKeyframesWidget();
            ensureNodeSize();
            setTimeout(() => tryLoadPreview(), 300);
        };

        const origOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function () {
            if (origOnConnectionsChange) origOnConnectionsChange.apply(this, arguments);
            lastVideoPath = null;
            setTimeout(() => tryLoadPreview(), 500);
        };

        const origOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            if (origOnExecuted) origOnExecuted.apply(this, arguments);
            lastVideoPath = null;
            tryLoadPreview();
        };
    }
});
